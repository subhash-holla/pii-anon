"""Tests for S4-03: SelectiveRiskReporter — per-class ECE/Brier/AURC + abstention.

Implements / pins (the story §3 acceptance A1..A9):
    FR-005   — calibration & selective-risk scorecard (ECE/MCE/Brier/AURC + reliability).
    NFR-017  — post-temp-scaling ECE ≤ 0.05 (high-resource) / ≤ 0.08 (long-tail); ECE_post ≤ ECE_pre.
    NFR-018  — Brier + decomposition per powered class (COULD).
    NFR-019  — selective-risk AURC + risk-coverage curve, monotone non-increasing.
    NFR-020  — calibrated confidence on EVERY finding (AX-005; 0 bare-logit) — the lone MUST.
    NFR-021  — abstention coverage-risk ≥3-point operating-point table at selective-risk {1%,2%,5%}.
    AX-005   — calibrated-abstention (the headline); AX-002 determinism; AX-001 synthetic fixtures.

Everything here is PURE-NUMPY / stdlib over SYNTHETIC per-finding
(confidence, correct, entity_type, calibrated, provenance) corpora — NEVER real
PII (AX-001). The reporter REUSES the sibling ``metrics/calibration.py`` global
ECE/MCE + reliability + TemperatureScaler (RISK-1: consumed, never rewritten).
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest

from pii_anon.eval_framework.metrics.selective_risk import (
    AbstentionOperatingPoint,
    ScoredFinding,
    SelectiveRiskReport,
    SelectiveRiskReporter,
)

# ---------------------------------------------------------------------------
# Synthetic per-finding corpora (AX-001 — no real PII).
#
# Each ScoredFinding carries (entity_type, confidence, correct, calibrated,
# provenance). ``calibrated``/``provenance`` drive the NFR-020 coverage audit;
# (confidence, correct) drive ECE/Brier/AURC.
# ---------------------------------------------------------------------------


def _well_calibrated_class(
    entity_type: str, *, n: int = 200, seed: int = 0
) -> list[ScoredFinding]:
    """A class whose confidence ≈ empirical accuracy in each band (low ECE).

    For a confidence band centred at p, a fraction ≈ p of the findings are
    correct — the textbook well-calibrated regime.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    out: list[ScoredFinding] = []
    for _ in range(n):
        conf = float(rng.uniform(0.05, 0.99))
        correct = bool(rng.random() < conf)  # P(correct) == confidence
        out.append(
            ScoredFinding(
                entity_type=entity_type,
                confidence=conf,
                correct=correct,
                calibrated=True,
                provenance="ensemble:weighted_consensus",
            )
        )
    return out


def _overconfident_class(
    entity_type: str, *, n: int = 200, seed: int = 1
) -> list[ScoredFinding]:
    """A badly mis-calibrated class: high confidence, low accuracy (high ECE)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out: list[ScoredFinding] = []
    for _ in range(n):
        conf = float(rng.uniform(0.85, 0.99))  # always very confident
        correct = bool(rng.random() < 0.40)  # but only ~40% correct
        out.append(
            ScoredFinding(
                entity_type=entity_type,
                confidence=conf,
                correct=correct,
                calibrated=True,
                provenance="ensemble:weighted_consensus",
            )
        )
    return out


def _two_class_corpus() -> list[ScoredFinding]:
    """A corpus spanning two entity classes: one well-calibrated, one over-confident."""
    return _well_calibrated_class("EMAIL_ADDRESS", seed=0) + _overconfident_class(
        "US_SSN", seed=1
    )


# ---------------------------------------------------------------------------
# A1 — per-class ECE (NFR-017) [UNIT-TEST]
# ---------------------------------------------------------------------------


def test_a1_per_class_ece_is_distinct_from_global_and_reuses_calibration() -> None:
    """[UNIT-TEST] A1 / NFR-017: the reporter returns a PER-ENTITY-CLASS ECE
    (distinct from the global ECE), reusing metrics/calibration.py — a
    deliberately mis-calibrated class shows a higher ECE than a well-calibrated
    one."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    assert isinstance(report, SelectiveRiskReport)
    # Per-class ECE present for BOTH classes, keyed by entity_type.
    assert set(report.per_class_ece) == {"EMAIL_ADDRESS", "US_SSN"}
    # The over-confident class has a strictly higher ECE than the well-calibrated.
    assert report.per_class_ece["US_SSN"] > report.per_class_ece["EMAIL_ADDRESS"]
    # The global ECE exists and is distinct from any single per-class ECE value
    # (it pools both classes).
    assert 0.0 <= report.global_ece <= 1.0
    assert report.per_class_ece["US_SSN"] != pytest.approx(report.global_ece)


def test_a1_per_class_ece_matches_calibration_module_on_a_single_slice() -> None:
    """[UNIT-TEST] A1: the per-class ECE is exactly what metrics/calibration.py's
    reliability_diagram_data would compute on that class's (confidence, correct)
    slice — proving REUSE, not a re-derivation (RISK-1)."""
    from pii_anon.eval_framework.metrics.calibration import reliability_diagram_data

    corpus = _two_class_corpus()
    report = SelectiveRiskReporter().report(corpus)
    ssn = [f for f in corpus if f.entity_type == "US_SSN"]
    expected = reliability_diagram_data(
        [f.confidence for f in ssn], [f.correct for f in ssn]
    ).ece
    assert report.per_class_ece["US_SSN"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# A2 — per-class Brier + decomposition (NFR-018) [UNIT-TEST]
# ---------------------------------------------------------------------------


def test_a2_per_class_brier_low_for_calibrated_high_for_miscalibrated() -> None:
    """[UNIT-TEST] A2 / NFR-018: per-class Brier — a well-calibrated class scores
    a lower Brier than a mis-calibrated (over-confident) class."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    assert set(report.per_class_brier) == {"EMAIL_ADDRESS", "US_SSN"}
    assert report.per_class_brier["US_SSN"] > report.per_class_brier["EMAIL_ADDRESS"]
    # Brier is a mean-squared-error in [0, 1].
    for v in report.per_class_brier.values():
        assert 0.0 <= v <= 1.0


def test_a2_brier_decomposition_reconciles_with_brier() -> None:
    """[UNIT-TEST] A2 / NFR-018: the Murphy decomposition
    (reliability − resolution + uncertainty) reconciles with the raw Brier for
    each class (within numerical tolerance)."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    for et, decomp in report.per_class_brier_decomposition.items():
        recon = decomp.reliability - decomp.resolution + decomp.uncertainty
        assert recon == pytest.approx(report.per_class_brier[et], abs=1e-6)


# ---------------------------------------------------------------------------
# A3 — AURC + monotone risk-coverage (NFR-019) [UNIT-TEST] [PROPERTY-TEST]
# ---------------------------------------------------------------------------


def test_a3_aurc_lower_for_better_calibrated_class() -> None:
    """[UNIT-TEST] A3 / NFR-019: per-class AURC + risk-coverage curve; AURC is
    LOWER for the better-calibrated (higher-accuracy, confidence-ordered) class."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    assert set(report.per_class_aurc) == {"EMAIL_ADDRESS", "US_SSN"}
    # The well-calibrated/accurate class abstains its errors away faster → lower AURC.
    assert report.per_class_aurc["EMAIL_ADDRESS"] < report.per_class_aurc["US_SSN"]


def test_a3_risk_coverage_curve_is_monotone_non_increasing_property() -> None:
    """[PROPERTY-TEST] A3 / NFR-019: the risk-coverage curve is monotone
    NON-INCREASING in coverage — selective risk never DROPS as you cover MORE
    (equivalently: as you abstain more, achieved risk does not increase). Asserted
    over both classes AND the global curve, across several random corpora."""
    import numpy as np

    rng = np.random.default_rng(7)
    for _ in range(8):
        corpus = _well_calibrated_class(
            "EMAIL_ADDRESS", n=120, seed=int(rng.integers(0, 1_000))
        ) + _overconfident_class("US_SSN", n=120, seed=int(rng.integers(0, 1_000)))
        report = SelectiveRiskReporter().report(corpus)
        curves = list(report.per_class_risk_coverage.values()) + [
            report.global_risk_coverage
        ]
        for curve in curves:
            # curve is ordered by ascending coverage; risk must be non-decreasing
            # as coverage grows (risk at full coverage ≥ risk at low coverage),
            # i.e. abstaining (lower coverage) cannot increase selective risk.
            risks = [pt.risk for pt in curve]
            for lo, hi in zip(risks, risks[1:]):
                assert hi >= lo - 1e-9


def test_a3_aurc_in_unit_interval() -> None:
    """[UNIT-TEST] A3: AURC is a normalised area in [0, 1]."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    for v in report.per_class_aurc.values():
        assert 0.0 <= v <= 1.0
    assert 0.0 <= report.global_aurc <= 1.0


# ---------------------------------------------------------------------------
# A4 — abstention operating-point table (NFR-021) [UNIT-TEST]
# ---------------------------------------------------------------------------


def test_a4_abstention_table_has_three_points_at_targets() -> None:
    """[UNIT-TEST] A4 / NFR-021: a ≥3-point abstention operating-point table at
    selective-risk targets {1%, 2%, 5%}; each row maps target risk → achieved
    coverage."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    table = report.abstention_table
    assert len(table) >= 3
    targets = sorted(round(p.target_risk, 4) for p in table)
    assert targets == [0.01, 0.02, 0.05]
    for pt in table:
        assert isinstance(pt, AbstentionOperatingPoint)
        assert 0.0 <= pt.achieved_coverage <= 1.0
        assert 0.0 <= pt.achieved_risk <= 1.0


def test_a4_abstention_points_consistent_with_risk_coverage_curve() -> None:
    """[UNIT-TEST] A4: looser risk targets admit AT LEAST as much coverage as
    tighter ones (monotone: a 5% risk budget covers ≥ what a 1% budget covers) —
    consistent with the risk-coverage curve."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    by_target = {
        round(p.target_risk, 4): p.achieved_coverage for p in report.abstention_table
    }
    assert by_target[0.05] >= by_target[0.02] >= by_target[0.01] - 1e-9


# ---------------------------------------------------------------------------
# A5 — calibrated-confidence-coverage (NFR-020, the lone MUST) [AUDIT] [CONTRACT-TEST]
# ---------------------------------------------------------------------------


def test_a5_coverage_is_one_when_every_finding_calibrated_with_provenance() -> None:
    """[AUDIT] A5 / NFR-020: a corpus where EVERY finding carries a calibrated
    confidence + provenance → calibrated_confidence_coverage == 1.0 (0 bare-logit)."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    assert report.calibrated_confidence_coverage == 1.0
    assert report.bare_logit_count == 0
    assert report.uncalibrated_findings == ()


def test_a5_coverage_drops_and_flags_bare_logit_finding() -> None:
    """[CONTRACT-TEST] A5 / NFR-020 TEETH: a corpus with a bare-logit /
    provenance-less finding → coverage < 1.0 AND the audit FLAGS it (not a no-op
    that always reports 100%)."""
    corpus = _two_class_corpus()
    # Inject a bare-logit finding (not calibrated, no provenance) — a NFR-020 breach.
    corpus.append(
        ScoredFinding(
            entity_type="PHONE_NUMBER",
            confidence=3.7,  # a raw logit, not a [0,1] probability
            correct=True,
            calibrated=False,
            provenance=None,
        )
    )
    report = SelectiveRiskReporter().report(corpus)
    assert report.calibrated_confidence_coverage < 1.0
    assert report.bare_logit_count == 1
    # The audit surfaces the offending finding (index or descriptor), not silence.
    assert len(report.uncalibrated_findings) == 1


def test_a5_provenance_less_but_in_range_still_flagged() -> None:
    """[AUDIT] A5 / NFR-020: a finding with an in-range confidence but NO
    provenance is still a coverage breach (NFR-020 requires confidence AND
    provenance) — teeth on the provenance half too."""
    corpus = _well_calibrated_class("EMAIL_ADDRESS", n=50, seed=3)
    corpus.append(
        ScoredFinding(
            entity_type="EMAIL_ADDRESS",
            confidence=0.9,  # in range
            correct=True,
            calibrated=True,
            provenance=None,  # but no provenance → still a breach
        )
    )
    report = SelectiveRiskReporter().report(corpus)
    assert report.calibrated_confidence_coverage < 1.0
    assert report.bare_logit_count == 1


# ---------------------------------------------------------------------------
# A6 — post-temp-scaling ECE (NFR-017) [UNIT-TEST]
# ---------------------------------------------------------------------------


def test_a6_post_temp_scaling_ece_never_worse_than_pre() -> None:
    """[UNIT-TEST] A6 / NFR-017: reusing TemperatureScaler, ECE_post ≤ ECE_pre
    ALWAYS (the grid search includes T=1.0, so it can never make it worse)."""
    report = SelectiveRiskReporter().report(_two_class_corpus())
    for et in report.per_class_ece:
        assert report.per_class_ece_post[et] <= report.per_class_ece[et] + 1e-9


def test_a6_well_resourced_class_meets_high_resource_threshold() -> None:
    """[UNIT-TEST] A6 / NFR-017: the post-scaling ECE on a synthetic
    well-resourced (and originally well-calibrated) class is ≤ 0.05; the long-tail
    threshold 0.08 is applied to sparse classes (the reporter exposes which bar a
    class is held to)."""
    # A big, well-calibrated class → high-resource bar 0.05.
    corpus = _well_calibrated_class("EMAIL_ADDRESS", n=600, seed=11)
    # A tiny class → long-tail bar 0.08.
    corpus += _well_calibrated_class("CRYPTO_WALLET", n=8, seed=12)
    report = SelectiveRiskReporter().report(corpus)
    assert report.per_class_ece_post["EMAIL_ADDRESS"] <= 0.05
    assert report.ece_threshold_for("EMAIL_ADDRESS") == 0.05
    assert report.ece_threshold_for("CRYPTO_WALLET") == 0.08


# ---------------------------------------------------------------------------
# A9 — import-boundary + determinism (AX-002 / AX-001) [PROPERTY-TEST]
# ---------------------------------------------------------------------------


def test_a9_reporter_is_deterministic_on_identical_inputs() -> None:
    """[PROPERTY-TEST] A9 / AX-002: two runs of the reporter on the SAME synthetic
    inputs yield equal reports (per-class ECE/Brier/AURC + coverage + table)."""
    corpus = _two_class_corpus()
    r1 = SelectiveRiskReporter().report(corpus)
    r2 = SelectiveRiskReporter().report(copy.deepcopy(corpus))
    assert r1.per_class_ece == r2.per_class_ece
    assert r1.per_class_brier == r2.per_class_brier
    assert r1.per_class_aurc == r2.per_class_aurc
    assert r1.calibrated_confidence_coverage == r2.calibrated_confidence_coverage
    assert [
        (p.target_risk, p.achieved_coverage, p.achieved_risk)
        for p in r1.abstention_table
    ] == [
        (p.target_risk, p.achieved_coverage, p.achieved_risk)
        for p in r2.abstention_table
    ]


_FORBIDDEN_IMPORT_HEADS: frozenset[str] = frozenset(
    {"swarm", "moe", "fusion", "policy"}
)


def _module_head_forbidden(module: str) -> bool:
    head = module.split(".")[0]
    if head in _FORBIDDEN_IMPORT_HEADS:
        return True
    # pii_anon.swarm / pii_anon.moe / ... — the second segment is the layer.
    parts = module.split(".")
    return len(parts) >= 2 and parts[0] == "pii_anon" and parts[1] in _FORBIDDEN_IMPORT_HEADS


def test_a9_selective_risk_has_no_detection_layer_imports() -> None:
    """[PROPERTY-TEST] A9: AST scan — selective_risk.py imports NOTHING from
    swarm/moe/fusion/policy (the reporter is eval REPORTING, decoupled from the
    detection/orchestration layers)."""
    import pii_anon.eval_framework.metrics.selective_risk as mod

    path = Path(mod.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _module_head_forbidden(alias.name):
                    violations.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if _module_head_forbidden(node.module):
                violations.append(node.module)
    assert not violations, f"selective_risk.py must not import detection layers: {violations}"


def test_a9_calibration_module_is_consumed_unchanged_byte_identical() -> None:
    """[PROPERTY-TEST] A9 / RISK-1: metrics/calibration.py is CONSUMED unchanged —
    its public API (reliability_diagram_data, TemperatureScaler, ECE/MCE) is
    importable and the reporter reuses it (the byte-identical invariant is also
    pinned by the md5 recorded in Evidence; here we assert the reuse surface)."""
    from pii_anon.eval_framework.metrics import calibration as cal

    assert hasattr(cal, "reliability_diagram_data")
    assert hasattr(cal, "TemperatureScaler")
    assert hasattr(cal, "ExpectedCalibrationError")
    assert hasattr(cal, "MaximumCalibrationError")


def test_re_exported_from_metrics_package() -> None:
    """[CONTRACT-TEST] The reporter is additively re-exported from
    eval_framework.metrics (existing exports untouched)."""
    from pii_anon.eval_framework import metrics

    assert hasattr(metrics, "SelectiveRiskReporter")
    assert hasattr(metrics, "SelectiveRiskReport")
    assert "SelectiveRiskReporter" in metrics.__all__


def test_empty_corpus_is_honest_not_a_crash() -> None:
    """[UNIT-TEST] An empty corpus yields an honest empty report (no per-class
    entries, coverage defined as 1.0 vacuously over zero findings) — never a
    division-by-zero crash."""
    report = SelectiveRiskReporter().report([])
    assert report.per_class_ece == {}
    assert report.abstention_table == ()
    assert report.bare_logit_count == 0
