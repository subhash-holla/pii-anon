"""Dominance verification: prove F1(ensemble) >= max(F1(expert_i)) per entity type.

Formal Argument
---------------
Given K experts for entity type T and the MoE floor guarantee:

1. The best expert B for T always appears in the routing
   (``performance_floor=True`` in ``MoERouter``).
2. B's softmax weight >= 1/K (when all K experts have equal strength).
3. Non-routed experts receive ``floor_weight > 0`` (never dropped).
4. Therefore, every entity found by B is present in the ensemble output
   (union guarantee).
5. Since recall = TP / (TP + FN) and every TP of B is a TP of the
   ensemble, we have ``recall(ensemble) >= recall(B)``.
6. Additional experts may add FP, but the calibrated weights suppress
   low-quality experts via lower MoE weight, limiting FP inflation.
7. The verifier empirically checks this on labeled data.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DominanceViolation:
    """A single entity type where the ensemble underperforms an expert."""

    entity_type: str
    best_expert_id: str
    best_expert_f1: float
    ensemble_f1: float
    gap: float  # best_expert_f1 - ensemble_f1


@dataclass
class DominanceReport:
    """Results of the dominance verification.

    Attributes
    ----------
    passed : bool
        ``True`` if the dominance property holds for all entity types.
    violations : list[DominanceViolation]
        Entity types where the ensemble underperformed.
    entity_type_results : dict[str, dict[str, float]]
        Per-entity-type F1 for each engine and the ensemble.
    """

    passed: bool
    violations: list[DominanceViolation] = field(default_factory=list)
    entity_type_results: dict[str, dict[str, float]] = field(default_factory=dict)


class DominanceVerifier:
    """Empirically verify the F1 dominance guarantee.

    Runs each engine individually and the full ensemble against labeled
    data, then checks that ``F1(ensemble) >= max(F1(expert_i))`` for
    every entity type.

    Parameters
    ----------
    orchestrator : PIIOrchestrator
        The orchestrator with all engines configured.
    engine_registry : EngineRegistry
        Registry of engines to test individually.
    """

    def __init__(
        self,
        orchestrator: Any,
        engine_registry: Any,
    ) -> None:
        self._orchestrator = orchestrator
        self._engine_registry = engine_registry

    def verify(
        self,
        dataset: str = "pii_anon_benchmark_v1",
        *,
        max_samples: int | None = None,
    ) -> DominanceReport:
        """Run dominance verification.

        Parameters
        ----------
        dataset : str
            Benchmark dataset name.
        max_samples : int | None
            Limit records processed.

        Returns
        -------
        DominanceReport
            Verification results with pass/fail and any violations.
        """
        from pii_anon.benchmarks import load_benchmark_dataset
        from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

        records = load_benchmark_dataset(dataset)
        if max_samples is not None:
            records = records[:max_samples]

        # 1. Compute per-engine per-entity F1
        engine_f1: dict[str, dict[str, float]] = {}
        engines = self._engine_registry.list_engines(include_disabled=False)

        for engine in engines:
            tp: dict[str, int] = defaultdict(int)
            fp: dict[str, int] = defaultdict(int)
            fn: dict[str, int] = defaultdict(int)

            for record in records:
                context = {"language": getattr(record, "language", "en"), "policy_mode": "balanced"}
                try:
                    findings = engine.detect({"text": record.text}, context)
                except Exception:
                    continue

                pred_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
                for f in findings:
                    if f.span_start is not None and f.span_end is not None:
                        pred_by_type[f.entity_type].add((f.span_start, f.span_end))

                gold_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
                for label in record.labels:
                    etype = str(label.get("entity_type", "UNKNOWN"))
                    gold_by_type[etype].add((int(label.get("start", 0)), int(label.get("end", 0))))

                for etype in set(pred_by_type.keys()) | set(gold_by_type.keys()):
                    preds = pred_by_type.get(etype, set())
                    golds = gold_by_type.get(etype, set())
                    tp[etype] += len(preds & golds)
                    fp[etype] += len(preds - golds)
                    fn[etype] += len(golds - preds)

            f1_by_type: dict[str, float] = {}
            for etype in set(tp.keys()) | set(fn.keys()):
                t, f, n = tp[etype], fp[etype], fn[etype]
                prec = t / max(1, t + f)
                rec = t / max(1, t + n)
                f1_by_type[etype] = 2 * prec * rec / max(1e-9, prec + rec)

            engine_f1[engine.adapter_id] = f1_by_type

        # 2. Compute ensemble F1
        ensemble_tp: dict[str, int] = defaultdict(int)
        ensemble_fp: dict[str, int] = defaultdict(int)
        ensemble_fn: dict[str, int] = defaultdict(int)

        profile = ProcessingProfileSpec(
            profile_id="dominance-verify",
            mode="mixture_of_experts",
            audit_enabled=False,
        )
        segmentation = SegmentationPlan(enabled=False)

        for record in records:
            result = self._orchestrator.run(
                {"text": record.text},
                profile=ProcessingProfileSpec(
                    profile_id=profile.profile_id,
                    mode=profile.mode,
                    language=getattr(record, "language", "en"),
                    audit_enabled=False,
                ),
                segmentation=segmentation,
                scope="dominance-verify",
                token_version=1,
            )
            finding_rows = result.get("ensemble_findings", [])

            ens_pred_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
            for finding in finding_rows:
                span = finding.get("span", {})
                start = span.get("start")
                end = span.get("end")
                if start is not None and end is not None:
                    ens_pred_by_type[str(finding.get("entity_type", "UNKNOWN"))].add((int(start), int(end)))

            ens_gold_by_type: dict[str, set[tuple[int, int]]] = defaultdict(set)
            for label in record.labels:
                etype = str(label.get("entity_type", "UNKNOWN"))
                ens_gold_by_type[etype].add((int(label.get("start", 0)), int(label.get("end", 0))))

            for etype in set(ens_pred_by_type.keys()) | set(ens_gold_by_type.keys()):
                preds = ens_pred_by_type.get(etype, set())
                golds = ens_gold_by_type.get(etype, set())
                ensemble_tp[etype] += len(preds & golds)
                ensemble_fp[etype] += len(preds - golds)
                ensemble_fn[etype] += len(golds - preds)

        ensemble_f1: dict[str, float] = {}
        for etype in set(ensemble_tp.keys()) | set(ensemble_fn.keys()):
            t, f, n = ensemble_tp[etype], ensemble_fp[etype], ensemble_fn[etype]
            prec = t / max(1, t + f)
            rec = t / max(1, t + n)
            ensemble_f1[etype] = 2 * prec * rec / max(1e-9, prec + rec)

        # 3. Check dominance: F1(ensemble) >= max(F1(expert_i)) per entity type
        all_entity_types = set(ensemble_f1.keys())
        for engine_results in engine_f1.values():
            all_entity_types.update(engine_results.keys())

        violations: list[DominanceViolation] = []
        entity_type_results: dict[str, dict[str, float]] = {}

        for etype in sorted(all_entity_types):
            row: dict[str, float] = {"ensemble": ensemble_f1.get(etype, 0.0)}
            best_id = "none"
            best_f1 = 0.0
            for engine_id, f1_map in engine_f1.items():
                f1_val = f1_map.get(etype, 0.0)
                row[engine_id] = f1_val
                if f1_val > best_f1:
                    best_f1 = f1_val
                    best_id = engine_id
            entity_type_results[etype] = row

            ens_f1 = ensemble_f1.get(etype, 0.0)
            if ens_f1 < best_f1 - 1e-6:  # small epsilon for float comparison
                violations.append(
                    DominanceViolation(
                        entity_type=etype,
                        best_expert_id=best_id,
                        best_expert_f1=round(best_f1, 4),
                        ensemble_f1=round(ens_f1, 4),
                        gap=round(best_f1 - ens_f1, 4),
                    )
                )

        return DominanceReport(
            passed=len(violations) == 0,
            violations=violations,
            entity_type_results=entity_type_results,
        )
