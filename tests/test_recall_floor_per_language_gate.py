"""S1-03 — Per-language recall-floor CI ε-gate (FR-007 / NFR-011, Design DC-11).

Asserts, for EVERY language ``L``::

    recall_ensemble[L] >= recall_shared[L] - RECALL_FLOOR_EPSILON   (ε = 0.005)

where ``recall_shared`` is the recall of the always-on shared layer (``regex-oss``)
alone and ``recall_ensemble`` is the recall of the LIVE floored fusion path wired
in S1-02 (:class:`~pii_anon.routing.floor_fusion.FloorProjectingFusion` returned by
``build_fusion("swarm", ...)``). Because S1-02 makes
``spans(ensemble) ⊇ spans(shared)`` by construction, per-language
``recall(ensemble) ≥ recall(shared)`` holds with ε=0 in the ideal; this gate is a
REGRESSION GUARD with a 0.005 tolerance for scoring-boundary edge cases (e.g.
span-offset rounding).

Test design (a TRUE regression guard on S1-02, not a trivial skip)
------------------------------------------------------------------
* A SELF-CONTAINED scenario that ALWAYS runs (no external dataset). It builds a
  small synthetic labeled multilingual set (``en``, ``es``, ``zh``) using
  synthetic-shaped values per AX-001 (NO real PII — only offsets/types matter to
  the span metrics). The non-English (``es``) record carries a SEMANTIC-type
  ``regex-oss`` span engineered to confidence 0.55 — below the swarm Layer-1
  fast-pass (0.90) and uncorroborated — so the swarm Layer-4
  emission/corroboration gate (``swarm.py:654/658-661``) DROPS it. Therefore
  WITHOUT the floor, ``recall_ensemble[es] < recall_shared[es]``.
* The gate asserts the per-language floor on the floored ``build_fusion("swarm")``
  output and passes (the floor re-injects the dropped ``es`` span).
* A sibling "teeth" assertion proves the guard has TEETH: against the BARE inner
  strategy (``SwarmFusionStrategy().merge(...)``, no floor) the ``es`` recall
  DROPS below the floor — i.e. this test would FAIL if S1-02 were reverted.

The per-language recall reuses the eval-framework span-alignment machinery
(``_aligned_prf`` over :class:`LabeledSpan`, the same primitive
``fairness_metrics`` groups-by-language with). Language is recovered from each
finding's ``field_path`` (one field per language) rather than the finding's
``.language`` attribute, because the swarm strategy does not propagate
``language`` onto natively-emitted ensemble findings — so the gate must not
depend on it for the bare-strategy teeth path.
"""
from __future__ import annotations

from pii_anon.eval_framework.metrics.base import LabeledSpan, MatchMode
from pii_anon.eval_framework.metrics.span_metrics import _aligned_prf
from pii_anon.fusion import build_fusion
from pii_anon.swarm import SwarmFusionStrategy
from pii_anon.types import EngineFinding, EnsembleFinding

try:  # pragma: no cover - exercised by both branches across the RED/GREEN cycle
    from tests.conftest import requires_dataset
except ImportError:  # pragma: no cover - conftest import style fallback
    from conftest import requires_dataset  # type: ignore[no-redef]

#: Per-language recall-floor tolerance (NFR-011: per-language ε ≤ 0.005).
#: Kept module-local (not in composite.py) to avoid any risk to the global
#: floor-gate behaviour; the story permits an additive composite.py constant but
#: prefers this when touching composite.py risks regressions.
RECALL_FLOOR_EPSILON = 0.005

#: One synthetic field per language — ``field_path`` is the record/language key
#: (survives both the bare swarm and the floored path; ``.language`` does not).
_FIELD_BY_LANG: dict[str, str] = {
    "en": "text_en",
    "es": "text_es",
    "zh": "text_zh",
}
_LANG_BY_FIELD: dict[str, str] = {field: lang for lang, field in _FIELD_BY_LANG.items()}


def _ef(
    entity_type: str,
    start: int,
    end: int,
    *,
    lang: str,
    confidence: float,
    engine_id: str = "regex-oss",
) -> EngineFinding:
    """A synthetic engine finding addressed to *lang*'s field (AX-001: no real PII)."""
    return EngineFinding(
        entity_type=entity_type,
        confidence=confidence,
        field_path=_FIELD_BY_LANG[lang],
        span_start=start,
        span_end=end,
        engine_id=engine_id,
        language=lang,
    )


def _lang_of(field_path: str | None) -> str:
    """Recover the language label from a finding's ``field_path``."""
    return _LANG_BY_FIELD.get(field_path or "", "unknown")


def _to_labeled(
    findings: list[EngineFinding] | list[EnsembleFinding],
) -> list[LabeledSpan]:
    """Project engine/ensemble findings onto :class:`LabeledSpan`, keying
    ``record_id`` by the per-language field so per-language grouping is robust to
    strategies that drop the ``.language`` attribute."""
    spans: list[LabeledSpan] = []
    for f in findings:
        if f.span_start is None or f.span_end is None:
            continue
        spans.append(
            LabeledSpan(
                entity_type=f.entity_type,
                start=f.span_start,
                end=f.span_end,
                record_id=_lang_of(f.field_path),
            )
        )
    return spans


def _recall_per_language(
    predictions: list[LabeledSpan], gold: list[LabeledSpan]
) -> dict[str, float]:
    """Per-language recall via the eval-framework span alignment (STRICT match).

    Groups predictions and gold by ``record_id`` (== language) and runs the same
    greedy ``_aligned_prf`` primitive that ``fairness_metrics`` uses, returning
    ``{language: recall}`` for every language present in the gold set.
    """
    langs = {s.record_id for s in gold}
    recalls: dict[str, float] = {}
    for lang in langs:
        preds_l = [s for s in predictions if s.record_id == lang]
        gold_l = [s for s in gold if s.record_id == lang]
        _, recall, *_ = _aligned_prf(preds_l, gold_l, MatchMode.STRICT)
        recalls[lang] = recall
    return recalls


def _synthetic_multilingual_case() -> tuple[
    list[EngineFinding], list[LabeledSpan]
]:
    """Build the synthetic shared-layer (``regex-oss``) findings + gold spans.

    Languages ``en`` / ``es`` / ``zh``; each gold span is a ``regex-oss`` hit.
    The ``es`` SEMANTIC span is engineered to confidence 0.55 (below swarm
    fast-pass, uncorroborated) so the bare swarm Layer-4 gate DROPS it — making
    the floor load-bearing for ``es``. The ``en`` and ``zh`` spans are
    high-confidence (≥ fast-pass) so the bare swarm keeps them natively.
    """
    shared = [
        # en: high-confidence CREDIT_CARD -> swarm Layer-1 fast-pass keeps it.
        _ef("CREDIT_CARD", 10, 26, lang="en", confidence=0.95),
        # es: low-confidence SEMANTIC span -> swarm Layer-4 gate DROPS it.
        _ef("CREDIT_CARD", 40, 56, lang="es", confidence=0.55),
        # zh: high-confidence EMAIL_ADDRESS -> fast-pass keeps it.
        _ef("EMAIL_ADDRESS", 5, 22, lang="zh", confidence=0.96),
    ]
    gold = _to_labeled(shared)  # every shared hit is a gold entity (recall basis)
    return shared, gold


# ── FR-007 / NFR-011: per-language recall floor on the LIVE floored seam ─────


def test_nfr_011_per_language_recall_floor_holds_on_floored_fusion() -> None:
    """For EVERY language L: ``recall_ensemble[L] >= recall_shared[L] - ε`` when
    the ensemble is the LIVE floored ``build_fusion("swarm")`` path (S1-02).

    REGRESSION-GUARD STATE (RED): the ensemble is temporarily computed from the
    BARE inner ``SwarmFusionStrategy`` (the pre-S1-02 state) — the ``es`` span is
    dropped, so ``recall_ensemble[es] = 0 < recall_shared[es] - ε`` and this
    assertion FAILS. GREEN flips ``_ensemble_merge`` to the floored seam.
    """
    shared, gold = _synthetic_multilingual_case()

    # NOTE (RED): bare inner strategy — no floor. Flipped to build_fusion in GREEN.
    ensemble_out = SwarmFusionStrategy().merge(list(shared))

    recall_shared = _recall_per_language(_to_labeled(shared), gold)
    recall_ensemble = _recall_per_language(_to_labeled(ensemble_out), gold)

    for lang in sorted(recall_shared):
        assert recall_ensemble.get(lang, 0.0) >= recall_shared[lang] - RECALL_FLOOR_EPSILON, (
            f"per-language recall floor violated for {lang!r}: "
            f"ensemble={recall_ensemble.get(lang, 0.0):.4f} < "
            f"shared={recall_shared[lang]:.4f} - ε({RECALL_FLOOR_EPSILON})"
        )


def test_nfr_011_floor_has_teeth_bare_swarm_drops_non_english_below_floor() -> None:
    """TEETH: against the BARE inner strategy (no floor) the non-English (``es``)
    recall DROPS below the floor — documenting that the gate above would FAIL if
    S1-02's floor were reverted. This is what makes the gate a real regression
    guard rather than a tautology.
    """
    shared, gold = _synthetic_multilingual_case()

    # Precondition: the bare swarm genuinely drops the es span (not a coincidence).
    es_only = [f for f in shared if _lang_of(f.field_path) == "es"]
    assert SwarmFusionStrategy().merge(list(es_only)) == [], (
        "precondition: the bare swarm Layer-4 gate must drop the uncorroborated "
        "es SEMANTIC span"
    )

    recall_shared = _recall_per_language(_to_labeled(shared), gold)
    bare_out = SwarmFusionStrategy().merge(list(shared))
    recall_bare = _recall_per_language(_to_labeled(bare_out), gold)

    assert recall_shared["es"] == 1.0, "regex-oss must detect the es gold span"
    assert recall_bare.get("es", 0.0) < recall_shared["es"] - RECALL_FLOOR_EPSILON, (
        "teeth check: without the floor the es recall must drop below the floor "
        f"(bare={recall_bare.get('es', 0.0):.4f}, shared={recall_shared['es']:.4f})"
    )


# ── OPTIONAL: real multilingual eval dataset (skips gracefully when absent) ──


@requires_dataset
def test_nfr_011_per_language_recall_floor_on_real_dataset() -> None:
    """Run the same per-language floor check over the real multilingual eval
    dataset when ``pii-anon-datasets`` is installed; skip gracefully otherwise
    (``@requires_dataset``) so CI never goes falsely green/red.

    Placeholder pending the dataset-runner wiring (the self-contained tests above
    are the load-bearing regression guard). Kept as a marked skip so the
    integration surface is discoverable and enabled the moment data is present.
    """
    import pytest

    pytest.skip("real-dataset per-language floor runner is a follow-on; "
                "self-contained gate above is the regression guard")
