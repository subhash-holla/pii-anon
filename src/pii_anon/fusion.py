"""Fusion strategies for merging PII findings from multiple engines.

When multiple detection engines run on the same payload, fusion strategies
merge their findings to produce consensus results. Different strategies
prioritize different goals:

- **Union (High Recall)**: emit all findings (no deduplication).
- **Weighted Consensus**: merge by location, weight by engine reliability.
- **Calibrated Majority**: require N engines to agree (precision-focused).
- **Intersection Consensus**: strict agreement (highest precision).

Custom strategies can be registered via ``register_fusion_strategy()``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable

from pii_anon.errors import FusionError
from pii_anon.types import EngineFinding, EnsembleFinding, FusionAuditRecord, FusionMode

FindingKey = tuple[str, str | None, int | None, int | None, str]
AuditKey = tuple[str, str | None, int | None, int | None]


class FusionStrategy:
    """Abstract base for fusion strategies.

    Subclasses implement ``merge()`` to combine raw engine findings into
    deduplicated ensemble findings.
    """
    strategy_id = "base"

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge raw engine findings into ensemble findings.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Deduplicated ensemble findings.
        """
        raise NotImplementedError


class UnionHighRecallFusion(FusionStrategy):
    """Union-based fusion: emit all findings without merging.

    Maximizes recall by including every engine finding, even if only one
    engine detected it. Useful when any detection is valuable and false
    positives can be filtered downstream.

    **Trade-off**: High recall, potentially low precision.
    """
    strategy_id = "union_high_recall"

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Wrap each raw finding as an ensemble finding.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            One ensemble finding per raw finding (no deduplication).
        """
        out: list[EnsembleFinding] = []
        for finding in findings:
            out.append(
                EnsembleFinding(
                    entity_type=finding.entity_type,
                    confidence=finding.confidence,
                    engines=[finding.engine_id],
                    field_path=finding.field_path,
                    span_start=finding.span_start,
                    span_end=finding.span_end,
                    explanation=finding.explanation,
                    language=finding.language,
                )
            )
        return out


def _overlap_iou(s1: int, e1: int, s2: int, e2: int) -> float:
    """Return Intersection-over-Union of two span ranges."""
    intersection = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return intersection / union if union > 0 else 0.0


# Key for grouping overlapping spans: (entity_type, field_path, language)
_OverlapGroupKey = tuple[str, str | None, str]


def _cluster_overlapping_spans(
    findings: list[EngineFinding],
    *,
    iou_threshold: float = 0.5,
) -> list[list[EngineFinding]]:
    """Cluster findings whose spans overlap above a threshold.

    Findings of the same entity type, field, and language are clustered when
    their span IoU exceeds ``iou_threshold``.  Non-overlapping findings each
    form their own single-element cluster.  Within each cluster the span used
    for the merged result is the one with the highest weighted confidence so
    that the best engine's boundaries win.
    """
    by_group: dict[_OverlapGroupKey, list[EngineFinding]] = defaultdict(list)
    for f in findings:
        key = (f.entity_type, f.field_path, f.language)
        by_group[key].append(f)

    clusters: list[list[EngineFinding]] = []
    for group in by_group.values():
        # Sort by span_start for sweep-line clustering
        sorted_group = sorted(group, key=lambda f: (f.span_start or 0, f.span_end or 0))
        used = [False] * len(sorted_group)
        for i, anchor in enumerate(sorted_group):
            if used[i]:
                continue
            cluster = [anchor]
            used[i] = True
            a_start = anchor.span_start or 0
            a_end = anchor.span_end or 0
            for j in range(i + 1, len(sorted_group)):
                if used[j]:
                    continue
                b = sorted_group[j]
                b_start = b.span_start or 0
                b_end = b.span_end or 0
                if b_start >= a_end + 5:
                    # Past the anchor span — no more overlaps possible
                    break
                if _overlap_iou(a_start, a_end, b_start, b_end) >= iou_threshold:
                    cluster.append(b)
                    used[j] = True
                    # Expand anchor envelope to union of spans
                    a_start = min(a_start, b_start)
                    a_end = max(a_end, b_end)
            clusters.append(cluster)
    return clusters


class WeightedConsensusFusion(FusionStrategy):
    """Confidence-weighted consensus: merge findings from same location.

    Groups findings by location (entity type, field, span), then computes
    a weighted average confidence. Engine weights reflect relative reliability.

    When multiple engines detect the same entity at slightly different
    boundaries (e.g. regex finds "123-45-6789" at (15, 26) while Presidio
    finds it at (15, 27)), an overlap-aware pre-clustering step merges them
    into a single finding using the best engine's span boundaries.

    **Trade-off**: Balanced precision and recall. Recovers high-confidence
    matches even if only one trusted engine found them.

    Parameters
    ----------
    weights : dict[str, float] | None
        Per-engine weights. Default weight is 1.0. Higher weight = more
        influential engine.
    iou_threshold : float
        Minimum Intersection-over-Union for two spans to be considered
        overlapping (default 0.5).
    """
    strategy_id = "weighted_consensus"

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        iou_threshold: float = 0.5,
    ) -> None:
        self.weights = weights or {}
        self.iou_threshold = iou_threshold

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge findings by location, weighted by engine reliability.

        Overlapping spans of the same entity type are clustered first,
        then each cluster is reduced to a single ensemble finding using
        weighted-average confidence and the highest-confidence engine's
        span boundaries.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Findings deduplicated by location with weighted-average confidence.
        """
        clusters = _cluster_overlapping_spans(findings, iou_threshold=self.iou_threshold)

        merged: list[EnsembleFinding] = []
        for cluster in clusters:
            weighted_sum = 0.0
            total_weight = 0.0
            engines: list[str] = []
            # Collect all span boundaries for majority voting.
            # When engines disagree on exact boundaries, use the most common
            # start/end pair rather than the single highest-confidence engine's
            # boundaries.  This avoids the problem where a high-weight engine
            # with slightly-off boundaries overrides a correct detection from
            # another engine, turning a true positive into a false negative in
            # exact-match evaluation.
            start_votes: dict[int, float] = {}
            end_votes: dict[int, float] = {}
            for item in cluster:
                weight = self.weights.get(item.engine_id, 1.0)
                weighted_sum += item.confidence * weight
                total_weight += weight
                engines.append(item.engine_id)
                s = item.span_start or 0
                e = item.span_end or 0
                start_votes[s] = start_votes.get(s, 0.0) + weight
                end_votes[e] = end_votes.get(e, 0.0) + weight

            # Pick the boundary with the most weighted votes
            best_span_start = max(start_votes, key=lambda k: start_votes[k])
            best_span_end = max(end_votes, key=lambda k: end_votes[k])

            representative = cluster[0]
            merged.append(
                EnsembleFinding(
                    entity_type=representative.entity_type,
                    confidence=(weighted_sum / total_weight) if total_weight else 0.0,
                    engines=sorted(set(engines)),
                    field_path=representative.field_path,
                    span_start=best_span_start,
                    span_end=best_span_end,
                    language=representative.language,
                )
            )
        return merged


class CalibratedMajorityFusion(FusionStrategy):
    """Consensus-threshold fusion: require N engines to agree.

    Builds on weighted consensus but filters results to only those found by
    at least N engines. Increases precision at potential cost of recall.

    **Trade-off**: Precision-focused. Good when multiple independent engines
    provide stronger signal than single high-confidence detection.

    Parameters
    ----------
    min_consensus : int
        Minimum number of engines required (default 2).
    """
    strategy_id = "calibrated_majority"

    def __init__(self, min_consensus: int = 2) -> None:
        self.min_consensus = min_consensus

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge findings by location and filter by consensus threshold.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Findings found by at least ``min_consensus`` engines, with
            weighted-average confidence.
        """
        weighted = WeightedConsensusFusion().merge(findings)
        return [finding for finding in weighted if len(finding.engines) >= self.min_consensus]


class IntersectionConsensusFusion(FusionStrategy):
    """Strict intersection: only emit findings from all engines.

    Highest precision. Only reports entities that **all** configured engines
    independently detected. Excludes findings from subset of engines.

    **Trade-off**: Highest precision, lowest recall. Suitable when false
    positives are costly and missed detections are acceptable.

    Parameters
    ----------
    min_consensus : int
        Minimum number of engines required (default/min 2).
    """
    strategy_id = "intersection_consensus"

    def __init__(self, min_consensus: int = 2) -> None:
        self.min_consensus = max(2, min_consensus)

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Merge findings by location and require strict intersection.

        Only returns findings detected by at least ``min_consensus`` engines.
        Uses the minimum confidence across all engines for conservative scoring.

        Parameters
        ----------
        findings : list[EngineFinding]
            Raw findings from all engines.

        Returns
        -------
        list[EnsembleFinding]
            Findings found by at least ``min_consensus`` engines, with
            minimum confidence across contributors.
        """
        grouped: dict[FindingKey, list[EngineFinding]] = defaultdict(list)
        for finding in findings:
            key = (
                finding.entity_type,
                finding.field_path,
                finding.span_start,
                finding.span_end,
                finding.language,
            )
            grouped[key].append(finding)

        merged: list[EnsembleFinding] = []
        for (entity_type, field_path, span_start, span_end, language), bucket in grouped.items():
            engines = sorted({item.engine_id for item in bucket})
            if len(engines) < self.min_consensus:
                continue
            merged.append(
                EnsembleFinding(
                    entity_type=entity_type,
                    confidence=min(item.confidence for item in bucket),
                    engines=engines,
                    field_path=field_path,
                    span_start=span_start,
                    span_end=span_end,
                    explanation="intersection consensus",
                    language=language,
                )
            )
        return merged


CustomFusionFactory = Callable[[dict[str, float], int], FusionStrategy]
_CUSTOM_FACTORIES: dict[str, CustomFusionFactory] = {}


def register_fusion_strategy(mode: str, factory: CustomFusionFactory) -> None:
    """Register a custom fusion strategy.

    Allows users to implement domain-specific fusion logic and make it
    available to the orchestrator.

    Parameters
    ----------
    mode : str
        Strategy name (e.g., "my_custom_fusion").
    factory : CustomFusionFactory
        Callable that accepts ``(weights: dict[str, float], min_consensus: int)``
        and returns a ``FusionStrategy`` instance.

    Notes
    -----
    Custom strategies are discovered by ``available_fusion_modes()`` and
    can be selected in ``ProcessingProfileSpec.mode``.

    Example
    -------
    >>> def my_factory(weights, min_consensus):
    ...     return MyCustomFusion(weights, min_consensus)
    >>> register_fusion_strategy("my_custom", my_factory)
    """
    _CUSTOM_FACTORIES[mode] = factory


def available_fusion_modes() -> list[str]:
    """List all available fusion strategies (built-in and registered).

    Returns
    -------
    list[str]
        Sorted list of strategy names.
    """
    builtin = [
        "union_high_recall",
        "weighted_consensus",
        "calibrated_majority",
        "intersection_consensus",
    ]
    return sorted(set(builtin + list(_CUSTOM_FACTORIES.keys())))


def build_fusion(mode: FusionMode, *, weights: dict[str, float], min_consensus: int) -> FusionStrategy:
    """Instantiate a fusion strategy by name.

    Selects and configures a fusion strategy based on the mode string.
    Applies engine weights and consensus thresholds as appropriate.

    Parameters
    ----------
    mode : FusionMode
        Strategy name (e.g., "weighted_consensus", "calibrated_majority").
    weights : dict[str, float]
        Per-engine weights (passed to strategies that support it).
    min_consensus : int
        Consensus threshold (passed to strategies that support it).

    Returns
    -------
    FusionStrategy
        Configured strategy instance.

    Raises
    ------
    FusionError
        If mode is not recognized.
    """
    if mode == "union_high_recall":
        return UnionHighRecallFusion()
    if mode == "calibrated_majority":
        return CalibratedMajorityFusion(min_consensus=min_consensus)
    if mode == "intersection_consensus":
        return IntersectionConsensusFusion(min_consensus=min_consensus)
    if mode == "weighted_consensus":
        return WeightedConsensusFusion(weights=weights)
    if mode in _CUSTOM_FACTORIES:
        return _CUSTOM_FACTORIES[mode](weights, min_consensus)
    raise FusionError(f"Unknown fusion mode `{mode}`")


def build_fusion_audit(
    strategy: FusionStrategy,
    merged: list[EnsembleFinding],
    source_findings: list[EngineFinding],
) -> list[FusionAuditRecord]:
    """Generate audit records documenting fusion decisions.

    Creates one audit record per ensemble finding, linking it to the raw
    findings that contributed to it. Enables traceability for compliance
    and debugging.

    Parameters
    ----------
    strategy : FusionStrategy
        The strategy used to perform the fusion.
    merged : list[EnsembleFinding]
        The ensemble findings produced by fusion.
    source_findings : list[EngineFinding]
        The raw engine findings that were merged.

    Returns
    -------
    list[FusionAuditRecord]
        Audit records with lineage information.
    """
    by_key: dict[AuditKey, list[EngineFinding]] = defaultdict(list)
    for item in source_findings:
        key = (item.entity_type, item.field_path, item.span_start, item.span_end)
        by_key[key].append(item)

    audits: list[FusionAuditRecord] = []
    for finding in merged:
        key = (finding.entity_type, finding.field_path, finding.span_start, finding.span_end)
        source = by_key.get(key, [])
        engines = sorted({entry.engine_id for entry in source}) or finding.engines
        audits.append(
            FusionAuditRecord(
                strategy=strategy.strategy_id,
                entity_type=finding.entity_type,
                field_path=finding.field_path,
                span_start=finding.span_start,
                span_end=finding.span_end,
                source_engines=engines,
                source_count=len(source),
                fused_confidence=finding.confidence,
                notes=["derived_from_source_findings"],
            )
        )
    return audits
