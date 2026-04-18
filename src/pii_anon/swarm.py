"""Swarm fusion strategy: four-layer pipeline with Dawid-Skene aggregation.

Layer 1 — Regex fast-pass: accept high-confidence regex findings immediately.
Layer 2 — Prune redundant engine findings (Jaccard-based).
Layer 3 — Dawid-Skene Bayesian aggregation + temperature scaling + meta-learner.
Layer 4 — Corroboration filter + checksum validation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pii_anon.fusion import FusionStrategy, _cluster_overlapping_spans, _overlap_iou
from pii_anon.types import EngineFinding, EnsembleFinding

logger = logging.getLogger(__name__)

# Entity types where regex checksum/format validation provides high precision.
STRUCTURED_TYPES = frozenset({
    "EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "IBAN", "IP_ADDRESS",
    "MAC_ADDRESS", "ROUTING_NUMBER", "VIN", "DRIVERS_LICENSE",
    "PASSPORT", "BANK_ACCOUNT", "EMPLOYEE_ID", "LICENSE_PLATE",
    "MEDICAL_RECORD_NUMBER", "NATIONAL_ID", "CRYPTO_WALLET",
})

# Entity types that benefit from multi-engine corroboration in Layer 4.
#
# A finding survives the corroboration gate when either (a) at least
# ``corroboration_min`` engines agree on it, or (b) its meta-score beats
# ``corroboration_override_threshold`` (a single highly-confident engine
# can override).  Types *not* listed here skip the gate entirely — that
# is correct for deterministic structured formats (IBAN, SSN — Luhn /
# mod-97 checksums are stronger than multi-engine votes) but a precision
# hazard for semantic / regex-ambiguous types.
#
# ``EMAIL_ADDRESS`` and ``CREDIT_CARD`` were added after the v10 paper
# evaluation flagged them as the lowest-precision swarm outputs (0.46 /
# 0.48).  Their regex surface is permissive enough that NER engines
# produce many single-engine false positives (e.g. MAC addresses matched
# as emails, partial card fragments matched as CREDIT_CARD) which slip
# past the meta-learner.  Requiring multi-engine agreement clamps those
# down without hurting recall on the common case.
SEMANTIC_TYPES = frozenset({
    "PERSON_NAME", "ORGANIZATION", "LOCATION", "DATE_OF_BIRTH",
    "ADDRESS", "USERNAME", "PHONE_NUMBER",
    "EMAIL_ADDRESS", "CREDIT_CARD",
})


def _default_artifacts_dir() -> Path:
    """Resolve the default swarm artifacts directory.

    Checks (in order):
    1. ``PII_ANON_SWARM_ARTIFACTS`` env var
    2. ``~/.pii_anon/swarm/``
    """
    env = os.environ.get("PII_ANON_SWARM_ARTIFACTS")
    if env:
        return Path(env)
    return Path.home() / ".pii_anon" / "swarm"


@dataclass
class SwarmConfig:
    """Configuration for the swarm pipeline."""

    fast_pass_threshold: float = 0.90
    iou_threshold: float = 0.3
    corroboration_min: int = 2
    corroboration_override_threshold: float = 0.85
    similarity_threshold: float = 0.85
    max_engines: int = 4
    meta_learner_path: str | None = None
    ds_params_path: str | None = None
    calibration_path: str | None = None
    emission_threshold: float = 0.50
    #: Engine IDs that must survive the Layer 2 Jaccard pruner regardless
    #: of entity-type overlap with higher-ranked engines.  ``regex-oss``
    #: is always pinned implicitly because checksum validators are
    #: stronger than set-cover heuristics; add your own engine IDs here
    #: to guarantee they participate in the fusion even when their
    #: detected type set is similar to an existing engine.  See
    #: :doc:`/extend-swarm` for the "plug your own engine into the swarm"
    #: workflow.
    force_include_engines: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Auto-discover trained artifacts from the default location."""
        artifacts_dir = _default_artifacts_dir()
        if self.ds_params_path is None:
            candidate = artifacts_dir / "ds_params.json"
            if candidate.exists():
                self.ds_params_path = str(candidate)
        if self.calibration_path is None:
            candidate = artifacts_dir / "temperature.json"
            if candidate.exists():
                self.calibration_path = str(candidate)
        if self.meta_learner_path is None:
            candidate = artifacts_dir / "xgboost_model.ubj"
            if candidate.exists():
                self.meta_learner_path = str(candidate)

    @classmethod
    def from_json(cls, path: str | Path) -> SwarmConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class SpanCandidate:
    """Internal representation of a candidate span during aggregation."""

    entity_type: str
    span_start: int
    span_end: int
    field_path: str | None
    engine_findings: dict[str, EngineFinding] = field(default_factory=dict)
    ds_confidence: float = 0.0
    meta_score: float = 0.0
    corroboration_count: int = 0


# ---------------------------------------------------------------------------
# Temperature Scaler
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """Per-engine temperature scaling: calibrated_conf = sigmoid(logit / T)."""

    def __init__(self, temperatures: dict[str, float] | None = None) -> None:
        self._temps: dict[str, float] = temperatures or {}

    def scale(self, engine_id: str, raw_confidence: float) -> float:
        t = self._temps.get(engine_id, 1.0)
        if t <= 0:
            t = 1.0
        logit = math.log(max(raw_confidence, 1e-9) / max(1.0 - raw_confidence, 1e-9))
        scaled_logit = logit / t
        return 1.0 / (1.0 + math.exp(-scaled_logit))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._temps, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> TemperatureScaler:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(temperatures=data)


# ---------------------------------------------------------------------------
# Informativeness Scorer
# ---------------------------------------------------------------------------

class InformativenessScorer:
    """Score engines by the informativeness of their confidence distributions.

    Engines with fixed confidence (variance ~0) get low informativeness;
    engines with calibrated, variable confidence get high informativeness.
    """

    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores: dict[str, float] = scores or {}

    def score(self, engine_id: str) -> float:
        return self._scores.get(engine_id, 0.5)

    @classmethod
    def from_engine_findings(
        cls, findings_by_engine: dict[str, list[float]],
    ) -> InformativenessScorer:
        scores: dict[str, float] = {}
        for engine_id, confidences in findings_by_engine.items():
            if len(confidences) < 2:
                scores[engine_id] = 0.1
                continue
            mean = sum(confidences) / len(confidences)
            variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
            scores[engine_id] = min(1.0, max(0.1, math.sqrt(variance) * 5.0))
        return cls(scores=scores)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._scores, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> InformativenessScorer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(scores=data)


# ---------------------------------------------------------------------------
# Dawid-Skene Aggregator (pure Python, no crowd-kit)
# ---------------------------------------------------------------------------

class DawidSkeneAggregator:
    """Bayesian aggregation using pre-trained Dawid-Skene confusion matrices.

    At training time, the EM algorithm learns per-engine confusion matrices
    and class priors.  At inference time, only a single Bayesian forward
    pass is needed (no EM iteration), so latency is O(spans * engines * types).
    """

    def __init__(
        self,
        confusion_matrices: dict[str, dict[str, dict[str, float]]] | None = None,
        class_priors: dict[str, float] | None = None,
    ) -> None:
        self._confusion: dict[str, dict[str, dict[str, float]]] = confusion_matrices or {}
        self._priors: dict[str, float] = class_priors or {}
        # Pre-computed frozenset of prior keys for fast union during inference.
        self._prior_types: frozenset[str] = frozenset(self._priors)

    @property
    def is_trained(self) -> bool:
        return bool(self._confusion) and bool(self._priors)

    def infer(self, engine_votes: dict[str, str]) -> tuple[str, float]:
        """Infer the most likely true label given engine observations.

        Parameters
        ----------
        engine_votes
            Mapping of engine_id -> observed entity type for this span.

        Returns
        -------
        tuple[str, float]
            (best_label, confidence) where confidence is the posterior probability.
        """
        if not self.is_trained:
            # Fallback: majority vote with uniform confidence.
            from collections import Counter
            counts = Counter(engine_votes.values())
            best = counts.most_common(1)[0]
            return best[0], best[1] / max(len(engine_votes), 1)

        # Union of trained prior types with any additional types engines voted for.
        all_types = set(self._prior_types)
        for etype in engine_votes.values():
            all_types.add(etype)

        log_posteriors: dict[str, float] = {}
        for true_type in all_types:
            log_p = math.log(max(self._priors.get(true_type, 1e-6), 1e-12))
            for engine_id, observed_type in engine_votes.items():
                cm = self._confusion.get(engine_id)
                if cm is None:
                    continue
                row = cm.get(true_type)
                if row is None:
                    log_p += math.log(1e-6)
                else:
                    log_p += math.log(max(row.get(observed_type, 1e-6), 1e-12))
            log_posteriors[true_type] = log_p

        # Normalize via log-sum-exp.
        max_log = max(log_posteriors.values())
        exp_sum = sum(math.exp(lp - max_log) for lp in log_posteriors.values())
        log_norm = max_log + math.log(exp_sum)

        best_type = max(log_posteriors, key=log_posteriors.__getitem__)
        confidence = math.exp(log_posteriors[best_type] - log_norm)
        return best_type, confidence

    @classmethod
    def train_em(
        cls,
        annotations: list[dict[str, str]],
        *,
        max_iter: int = 50,
        tol: float = 1e-4,
    ) -> DawidSkeneAggregator:
        """Train confusion matrices using the EM algorithm.

        Parameters
        ----------
        annotations
            List of items, where each item is a dict of engine_id -> observed label.
            Items without an observation for an engine are skipped for that engine.
        """
        if not annotations:
            return cls()

        engines = sorted({eid for ann in annotations for eid in ann})
        all_labels = sorted({label for ann in annotations for label in ann.values()})
        n_labels = len(all_labels)
        n_items = len(annotations)
        label_idx = {lab: idx for idx, lab in enumerate(all_labels)}

        # Initialize: majority vote for class assignments.
        class_probs = [[0.0] * n_labels for _ in range(n_items)]
        for i, ann in enumerate(annotations):
            counts: dict[str, int] = {}
            for label in ann.values():
                counts[label] = counts.get(label, 0) + 1
            total = sum(counts.values())
            for label, count in counts.items():
                class_probs[i][label_idx[label]] = count / total

        # EM iterations.
        confusion: dict[str, list[list[float]]] = {}
        priors = [1.0 / n_labels] * n_labels

        for iteration in range(max_iter):
            # M-step: update confusion matrices and priors.
            old_confusion = dict(confusion)  # shallow copy for convergence check

            confusion = {}
            for engine_id in engines:
                cm = [[1e-6] * n_labels for _ in range(n_labels)]  # Laplace smoothing
                for i, ann in enumerate(annotations):
                    if engine_id not in ann:
                        continue
                    obs_idx = label_idx[ann[engine_id]]
                    for t in range(n_labels):
                        cm[t][obs_idx] += class_probs[i][t]
                # Normalize rows.
                for t in range(n_labels):
                    row_sum = sum(cm[t])
                    if row_sum > 0:
                        cm[t] = [v / row_sum for v in cm[t]]
                confusion[engine_id] = cm

            # Update priors.
            for t in range(n_labels):
                priors[t] = sum(class_probs[i][t] for i in range(n_items)) / n_items

            # E-step: update class assignments.
            for i, ann in enumerate(annotations):
                log_probs = [0.0] * n_labels
                for t in range(n_labels):
                    log_probs[t] = math.log(max(priors[t], 1e-12))
                    for engine_id, label in ann.items():
                        obs_idx = label_idx[label]
                        engine_cm = confusion.get(engine_id)
                        if engine_cm is not None:
                            log_probs[t] += math.log(max(engine_cm[t][obs_idx], 1e-12))

                # Normalize via log-sum-exp.
                max_lp = max(log_probs)
                exp_probs = [math.exp(lp - max_lp) for lp in log_probs]
                prob_total: float = sum(exp_probs)
                class_probs[i] = [ep / prob_total for ep in exp_probs]

            # Convergence check.
            if old_confusion:
                max_delta = 0.0
                for eid in engines:
                    if eid in old_confusion and eid in confusion:
                        for t in range(n_labels):
                            for o in range(n_labels):
                                delta = abs(confusion[eid][t][o] - old_confusion[eid][t][o])
                                max_delta = max(max_delta, delta)
                if max_delta < tol:
                    logger.info("Dawid-Skene converged at iteration %d (delta=%.6f)", iteration, max_delta)
                    break

        # Convert to dict-of-dict format.
        cm_dict: dict[str, dict[str, dict[str, float]]] = {}
        for engine_id, cm in confusion.items():
            cm_dict[engine_id] = {}
            for t in range(n_labels):
                cm_dict[engine_id][all_labels[t]] = {
                    all_labels[o]: cm[t][o] for o in range(n_labels)
                }

        prior_dict = {all_labels[t]: priors[t] for t in range(n_labels)}
        return cls(confusion_matrices=cm_dict, class_priors=prior_dict)

    def save(self, path: str | Path) -> None:
        data = {"confusion_matrices": self._confusion, "class_priors": self._priors}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> DawidSkeneAggregator:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            confusion_matrices=data.get("confusion_matrices", {}),
            class_priors=data.get("class_priors", {}),
        )


# ---------------------------------------------------------------------------
# Engine Redundancy Pruning
# ---------------------------------------------------------------------------

def _jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


def _prune_redundant_findings(
    findings: list[EngineFinding],
    *,
    similarity_threshold: float = 0.85,
    max_engines: int = 4,
    force_include_engines: tuple[str, ...] | frozenset[str] = (),
) -> list[EngineFinding]:
    """Remove findings from redundant engines using greedy set-cover.

    Engines are ranked by the number of distinct entity types they detected.
    Engines whose detected type set has Jaccard similarity >= threshold with
    an already-selected engine are pruned.

    Pinned engines — ``regex-oss`` always, plus any listed in
    *force_include_engines* — bypass the Jaccard check entirely.  This is
    the extension hook for users who plug a custom detector into the
    swarm (see :doc:`/extend-swarm`): name it in ``SwarmConfig.force_include_engines``
    and its findings survive Layer 2 even when they overlap with an
    existing engine.
    """
    # Group findings by engine.
    by_engine: dict[str, list[EngineFinding]] = {}
    for f in findings:
        by_engine.setdefault(f.engine_id, []).append(f)

    if len(by_engine) <= 1:
        return findings

    pinned = frozenset({"regex-oss", *force_include_engines})

    # Compute each engine's detected entity type set.
    engine_types: dict[str, set[str]] = {
        eid: {f.entity_type for f in fs} for eid, fs in by_engine.items()
    }

    # Pass 1: pin the always-include engines up-front.  Doing this
    # *before* the coverage loop means the ``max_engines`` cap cannot
    # consume a pinned engine's slot and leave it pruned — a user who
    # pinned their detector must always see it in the fused output.
    selected: list[str] = [eid for eid in engine_types if eid in pinned]

    # Pass 2: greedy selection over the remaining engines, ranked by
    # distinct-type count descending.  Pinned engines are excluded since
    # they are already in.
    remaining = sorted(
        (eid for eid in engine_types if eid not in pinned),
        key=lambda e: len(engine_types[e]),
        reverse=True,
    )

    # ``max_engines`` caps only the *non-pinned* slots so a caller that
    # pins many engines cannot accidentally smuggle unbounded extras
    # past the cap.  We track non-pinned selections explicitly rather
    # than subtracting set sizes, which miscounts when the caller pins
    # more engines than ``max_engines``.
    non_pinned_selected = 0
    for engine_id in remaining:
        if non_pinned_selected >= max_engines:
            break
        # Check redundancy with already-selected engines.
        redundant = False
        for sel_id in selected:
            sim = _jaccard_similarity(engine_types[engine_id], engine_types[sel_id])
            if sim >= similarity_threshold:
                redundant = True
                break
        if not redundant:
            selected.append(engine_id)
            non_pinned_selected += 1

    selected_set = set(selected)
    return [f for f in findings if f.engine_id in selected_set]


# ---------------------------------------------------------------------------
# Logistic Fallback (when XGBoost is unavailable)
# ---------------------------------------------------------------------------

def _logistic_fallback_score(
    ds_confidence: float,
    corroboration_count: int,
    regex_detected: bool,
    is_structured: bool,
) -> float:
    """Simple logistic scoring function used when XGBoost is not installed."""
    x = (
        2.0 * ds_confidence
        + 0.5 * min(corroboration_count, 4)
        + (0.8 if regex_detected else 0.0)
        + (0.3 if is_structured else 0.0)
        - 2.0
    )
    return 1.0 / (1.0 + math.exp(-x))


# ---------------------------------------------------------------------------
# Swarm Fusion Strategy
# ---------------------------------------------------------------------------

class SwarmFusionStrategy(FusionStrategy):
    """Four-layer swarm fusion with Dawid-Skene aggregation and meta-learner.

    This is the primary fusion strategy for pii-anon-swarm.  It implements:

    Layer 1 — Regex fast-pass (high-confidence regex accepted immediately).
    Layer 2 — Redundant engine pruning (Jaccard-based).
    Layer 3 — Dawid-Skene aggregation + temperature scaling + meta-learner.
    Layer 4 — Corroboration filter + checksum validation.
    """

    strategy_id = "swarm"

    def __init__(
        self,
        config: SwarmConfig | None = None,
        config_path: str | Path | None = None,
        ds_aggregator: DawidSkeneAggregator | None = None,
        temperature_scaler: TemperatureScaler | None = None,
        informativeness_scorer: InformativenessScorer | None = None,
        meta_learner: Any | None = None,
    ) -> None:
        if config is None:
            config = SwarmConfig()
        if config_path is not None:
            config = SwarmConfig.from_json(config_path)
        self._config = config

        # Load trained artifacts if paths are provided.
        self._ds = ds_aggregator
        if self._ds is None and config.ds_params_path:
            try:
                self._ds = DawidSkeneAggregator.load(config.ds_params_path)
            except Exception:
                logger.warning("Failed to load DS params from %s", config.ds_params_path)
        if self._ds is None:
            self._ds = DawidSkeneAggregator()

        self._temp_scaler = temperature_scaler
        if self._temp_scaler is None and config.calibration_path:
            try:
                cal_path = Path(config.calibration_path)
                temp_path = cal_path.parent / "temperature.json"
                if temp_path.exists():
                    self._temp_scaler = TemperatureScaler.load(temp_path)
            except Exception:
                logger.warning("Failed to load temperature params")
        if self._temp_scaler is None:
            self._temp_scaler = TemperatureScaler()

        self._informativeness = informativeness_scorer
        if self._informativeness is None and config.calibration_path:
            try:
                cal_path = Path(config.calibration_path)
                info_path = cal_path.parent / "informativeness.json"
                if info_path.exists():
                    self._informativeness = InformativenessScorer.load(info_path)
            except Exception:
                pass
        if self._informativeness is None:
            self._informativeness = InformativenessScorer()

        # All three are guaranteed non-None at this point (see fallback assignments above).

        self._meta_learner = meta_learner
        if self._meta_learner is None and config.meta_learner_path:
            try:
                from pii_anon.swarm_learner import XGBoostMetaLearner
                self._meta_learner = XGBoostMetaLearner.load(config.meta_learner_path)
            except Exception:
                logger.info("Meta-learner not available; using logistic fallback")

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        """Run the four-layer swarm pipeline."""
        if not findings:
            return []

        cfg = self._config
        fast_pass_results: list[EnsembleFinding] = []
        remaining: list[EngineFinding] = []

        # ── Layer 1: Regex fast-pass ──────────────────────────────────────
        for f in findings:
            if f.engine_id == "regex-oss" and f.confidence >= cfg.fast_pass_threshold:
                fast_pass_results.append(EnsembleFinding(
                    entity_type=f.entity_type,
                    confidence=f.confidence,
                    engines=[f.engine_id],
                    field_path=f.field_path,
                    span_start=f.span_start,
                    span_end=f.span_end,
                    explanation=f"swarm:fast_pass (conf={f.confidence:.2f})",
                ))
            else:
                remaining.append(f)

        if not remaining:
            return fast_pass_results

        # ── Layer 2: Prune redundant engine findings ──────────────────────
        remaining = _prune_redundant_findings(
            remaining,
            similarity_threshold=cfg.similarity_threshold,
            max_engines=cfg.max_engines,
            force_include_engines=cfg.force_include_engines,
        )

        # ── Layer 3: Learned aggregation ──────────────────────────────────
        # 3a. Cluster overlapping spans.
        clusters = _cluster_overlapping_spans(remaining, iou_threshold=cfg.iou_threshold)

        aggregated: list[SpanCandidate] = []
        for cluster in clusters:
            candidate = self._build_candidate(cluster)
            if candidate is not None:
                aggregated.append(candidate)

        # 3b-d. Temperature scale, Dawid-Skene, informativeness, meta-learner.
        total_engines = len({f.engine_id for f in remaining})
        for candidate in aggregated:
            self._aggregate_candidate(candidate, total_engines=total_engines)

        # ── Layer 4: Validation & post-processing ─────────────────────────
        results: list[EnsembleFinding] = []
        for candidate in aggregated:
            if candidate.meta_score < cfg.emission_threshold:
                continue

            # Corroboration filter for semantic types.
            if candidate.entity_type in SEMANTIC_TYPES:
                if (candidate.corroboration_count < cfg.corroboration_min
                        and candidate.meta_score < cfg.corroboration_override_threshold):
                    continue

            results.append(EnsembleFinding(
                entity_type=candidate.entity_type,
                confidence=candidate.meta_score,
                engines=list(candidate.engine_findings.keys()),
                field_path=candidate.field_path,
                span_start=candidate.span_start,
                span_end=candidate.span_end,
                explanation=(
                    f"swarm:ds={candidate.ds_confidence:.2f}"
                    f" meta={candidate.meta_score:.2f}"
                    f" engines={candidate.corroboration_count}"
                ),
            ))

        # Deduplicate fast-pass results vs. Layer 3 results.
        results = self._deduplicate_fast_pass(fast_pass_results, results)
        return results

    def _build_candidate(self, cluster: list[EngineFinding]) -> SpanCandidate | None:
        """Convert a cluster of overlapping findings into a SpanCandidate."""
        assert self._informativeness is not None
        if not cluster:
            return None

        engine_findings: dict[str, EngineFinding] = {}
        start_votes: dict[int, float] = {}
        end_votes: dict[int, float] = {}
        type_votes: dict[str, float] = {}

        # Single pass: deduplicate, vote on boundaries and types.
        for f in cluster:
            if f.engine_id not in engine_findings or f.confidence > engine_findings[f.engine_id].confidence:
                engine_findings[f.engine_id] = f

        for f in engine_findings.values():
            w = self._informativeness.score(f.engine_id) * f.confidence
            if f.span_start is not None:
                start_votes[f.span_start] = start_votes.get(f.span_start, 0.0) + w
            if f.span_end is not None:
                end_votes[f.span_end] = end_votes.get(f.span_end, 0.0) + w
            type_votes[f.entity_type] = type_votes.get(f.entity_type, 0.0) + w

        best_start = max(start_votes, key=start_votes.__getitem__) if start_votes else 0
        best_end = max(end_votes, key=end_votes.__getitem__) if end_votes else 0
        best_type = max(type_votes, key=type_votes.__getitem__) if type_votes else cluster[0].entity_type

        return SpanCandidate(
            entity_type=best_type,
            span_start=best_start,
            span_end=best_end,
            field_path=cluster[0].field_path,
            engine_findings=engine_findings,
            corroboration_count=len(engine_findings),
        )

    def _aggregate_candidate(self, candidate: SpanCandidate, total_engines: int) -> None:
        """Run Layer 3 aggregation on a single candidate."""
        assert self._temp_scaler is not None
        assert self._ds is not None
        # 3b. Temperature-scale confidences.  The input ``EngineFinding``
        # objects belong to the caller (they may be retained for audit
        # / logging / retry), so we replace the dict entries with
        # copies rather than mutating the originals — double-scaling on
        # a retry would be a silent correctness bug.
        candidate.engine_findings = {
            engine_id: dataclasses.replace(
                f,
                confidence=self._temp_scaler.scale(engine_id, f.confidence),
            )
            for engine_id, f in candidate.engine_findings.items()
        }

        # 3c. Dawid-Skene inference.
        engine_votes = {eid: f.entity_type for eid, f in candidate.engine_findings.items()}
        ds_type, ds_conf = self._ds.infer(engine_votes)
        candidate.ds_confidence = ds_conf
        if ds_type != candidate.entity_type and ds_conf > 0.8:
            candidate.entity_type = ds_type

        # 3e. Meta-learner or logistic fallback.
        if self._meta_learner is not None:
            try:
                candidate.meta_score = self._meta_learner.predict_candidate(
                    candidate, total_engines=total_engines,
                    informativeness_scorer=self._informativeness,
                )
                return
            except Exception:
                pass

        # Logistic fallback.
        regex_detected = "regex-oss" in candidate.engine_findings
        is_structured = candidate.entity_type in STRUCTURED_TYPES
        candidate.meta_score = _logistic_fallback_score(
            ds_confidence=candidate.ds_confidence,
            corroboration_count=candidate.corroboration_count,
            regex_detected=regex_detected,
            is_structured=is_structured,
        )

    def _deduplicate_fast_pass(
        self,
        fast_pass: list[EnsembleFinding],
        layer3: list[EnsembleFinding],
    ) -> list[EnsembleFinding]:
        """Remove Layer 3 findings that overlap with fast-pass results."""
        if not fast_pass:
            return layer3

        fp_spans: set[tuple[str | None, int | None, int | None]] = set()
        for ef in fast_pass:
            fp_spans.add((ef.field_path, ef.span_start, ef.span_end))

        deduped: list[EnsembleFinding] = []
        for ef in layer3:
            key = (ef.field_path, ef.span_start, ef.span_end)
            if key not in fp_spans:
                # Check IoU overlap with any fast-pass span.
                overlaps = False
                for fp in fast_pass:
                    if (fp.field_path == ef.field_path
                            and fp.span_start is not None and fp.span_end is not None
                            and ef.span_start is not None and ef.span_end is not None):
                        if _overlap_iou(fp.span_start, fp.span_end, ef.span_start, ef.span_end) > 0.3:
                            overlaps = True
                            break
                if not overlaps:
                    deduped.append(ef)

        return fast_pass + deduped
