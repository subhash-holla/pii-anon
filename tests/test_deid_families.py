"""Tests for S4-01: distinct anon-vs-pseudo reversibility scorers + no-merge guard.

Implements / pins (story §3 A1–A9):
    FR-006  pseudonymization integrity reported as a structurally distinct family.
    FR-009  pseudonymization integrity as a 5-axis family (reversal/collision/
            referential/key-separation/integrity).
    FR-010  enforce anon vs pseudo as distinct families — NO merge path/field.
    NFR-014 pseudonymization integrity: unauthorized-reversal = 0; referential
            integrity = 100%.
    NFR-015 key/state separation (Art-4(5) proxy): artifact-alone re-join = FAIL.
    AX-004  anonymization (irreversible) ≠ pseudonymization (reversible-under-key)
            — never merged into one de-id score (THE headline).
    AX-002  determinism (two runs on identical synthetic inputs → equal scores).
    AX-001  synthetic corpora only ("Jane Roe", "jane.roe@example.test"); the
            unauthorized-reversal probe uses a SYNTHETIC wrong key, never a secret.

Everything here is PURE-PYTHON over SYNTHETIC fixtures. There is NO real PII, no
network, no real key material. The two scorer families share no mutable state and
neither imports the other; the no-merge guard is an AST/structural CI gate with a
non-vacuous meta-guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pii_anon.eval_framework.metrics.deid_families as deid_families
from pii_anon.eval_framework.metrics.base import LabeledSpan
from pii_anon.eval_framework.metrics.deid_families import (
    AnonymizationScore,
    AnonymizationScorer,
    DeidFamilyScores,
    PseudonymizationIntegrityScore,
    PseudonymizationIntegrityScorer,
)

# ---------------------------------------------------------------------------
# Synthetic fixtures (AX-001) — never real PII.
# ---------------------------------------------------------------------------

_ORIGINAL = "Contact Jane Roe at jane.roe@example.test about case 42."
# Anonymized: the original surface forms are GONE (irreversible) → low risk.
_ANON_CLEAN = "Contact [PERSON] at [EMAIL] about case 42."
# A leaky anonymization: the original name survives → high re-identification risk.
_ANON_LEAKY = "Contact Jane Roe at [EMAIL] about case 42."

# A pseudonym map: same plaintext → same surrogate (referentially consistent).
_PSEUDO_MAP = {
    "Jane Roe": "PERSON_0001",
    "jane.roe@example.test": "EMAIL_0001",
}
# The authorized (synthetic) reversal key — never a real secret (AX-001).
_AUTHORIZED_KEY = "synthetic-test-key-0001"
_WRONG_KEY = "synthetic-WRONG-key-9999"

# A synthetic canary whose characters do NOT overlap the clean anonymized text
# above (CanaryExposureMetric scores partial CHARACTER overlap, so a canary that
# shares letters with "[person]/[email]/about/case" would partial-match a CLEAN
# output and spuriously read as exposed). "ZZZQVWXYJ" shares no chars with it.
_CANARY = "ZZZQVWXYJ"


def _labels(original: str) -> list[LabeledSpan]:
    """Ground-truth spans for the two PII surfaces in ``original`` (synthetic)."""
    name_start = original.index("Jane Roe")
    email_start = original.index("jane.roe@example.test")
    return [
        LabeledSpan("PERSON", name_start, name_start + len("Jane Roe")),
        LabeledSpan(
            "EMAIL_ADDRESS", email_start, email_start + len("jane.roe@example.test")
        ),
    ]


# ---------------------------------------------------------------------------
# A1 — two distinct family scores [UNIT-TEST]
# ---------------------------------------------------------------------------


def test_a1_two_distinct_family_subrecords_no_merged_field() -> None:
    """[UNIT-TEST] A1: scoring a synthetic eval result yields TWO separate
    sub-records on DeidFamilyScores (.anonymization + .pseudonymization_integrity)
    with NO merged de-id number anywhere on the record."""
    anon = AnonymizationScorer().score(
        labels=_labels(_ORIGINAL),
        original_text=_ORIGINAL,
        anonymized_text=_ANON_CLEAN,
        canary_strings=[_CANARY],
    )
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
    )
    scores = DeidFamilyScores(anonymization=anon, pseudonymization_integrity=pseudo)

    # Two separate, separately-addressable sub-records.
    assert isinstance(scores.anonymization, AnonymizationScore)
    assert isinstance(scores.pseudonymization_integrity, PseudonymizationIntegrityScore)
    assert scores.anonymization is anon
    assert scores.pseudonymization_integrity is pseudo

    # NO merged field on the record (positive separateness / AX-004 wall).
    field_names = set(vars(scores).keys())
    assert field_names == {"anonymization", "pseudonymization_integrity"}
    for banned in ("combined", "deid_score", "privacy_score", "merged", "overall"):
        assert banned not in field_names


def test_a1_no_api_returns_a_single_combined_deid_number() -> None:
    """[UNIT-TEST] A1: neither scorer nor the record exposes a callable/field that
    returns one merged anon+pseudo de-id scalar."""
    record_attrs = dir(DeidFamilyScores)
    for banned in ("combined_score", "deid_score", "merged_score", "to_single_score"):
        assert banned not in record_attrs


# ---------------------------------------------------------------------------
# A3 — unauthorized-reversal = 0 (NFR-014) [UNIT-TEST]
# ---------------------------------------------------------------------------


def test_a3_unauthorized_reversal_zero_on_correct_key_path() -> None:
    """[UNIT-TEST] A3 / NFR-014: with only the AUTHORIZED key, the
    unauthorized-reversal rate is 0.0 and the axis is satisfied."""
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[
            (_AUTHORIZED_KEY, True),
            (_AUTHORIZED_KEY, True),
        ],
    )
    assert pseudo.unauthorized_reversal_rate == 0.0
    assert pseudo.unauthorized_reversal_ok is True


def test_a3_unauthorized_reversal_has_teeth_wrong_key_succeeding_fails() -> None:
    """[UNIT-TEST] A3 / NFR-014 TEETH: a synthetic attempt that REVERSES WITHOUT
    the authorized key (wrong-key succeeds) is counted as an unauthorized reversal
    → rate > 0 → the axis FAILS. Proves the metric is not vacuous."""
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[
            (_AUTHORIZED_KEY, True),
            (_WRONG_KEY, True),  # a wrong key that nonetheless reversed → a leak
        ],
    )
    assert pseudo.unauthorized_reversal_rate > 0.0
    assert pseudo.unauthorized_reversal_ok is False


def test_a3_wrong_key_that_fails_to_reverse_is_not_unauthorized() -> None:
    """[UNIT-TEST] A3: a wrong-key attempt that correctly FAILS to reverse is NOT
    an unauthorized reversal (the system behaved correctly)."""
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[
            (_AUTHORIZED_KEY, True),
            (_WRONG_KEY, False),  # wrong key, correctly rejected
        ],
    )
    assert pseudo.unauthorized_reversal_rate == 0.0
    assert pseudo.unauthorized_reversal_ok is True


# ---------------------------------------------------------------------------
# A4 — key/state separation (NFR-015) [UNIT-TEST]
# ---------------------------------------------------------------------------


def test_a4_artifact_alone_rejoin_is_flagged_fail_when_no_external_secret() -> None:
    """[UNIT-TEST] A4 / NFR-015: given the pseudonymization artifact ALONE (no
    external secret available), the key/state-separation axis flags artifact-alone
    re-join as FAIL — reversal must require the external secret."""
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
        artifact_alone_rejoinable=True,  # the artifact alone can re-join → leak
    )
    assert pseudo.key_state_separation_ok is False


def test_a4_external_secret_required_passes_key_state_separation() -> None:
    """[UNIT-TEST] A4 / NFR-015: when re-join requires the external secret (the
    artifact alone CANNOT re-join), the key/state-separation axis passes."""
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
        artifact_alone_rejoinable=False,  # external secret required → separated
    )
    assert pseudo.key_state_separation_ok is True


# ---------------------------------------------------------------------------
# Referential integrity + collision (FR-009 axes)
# ---------------------------------------------------------------------------


def test_referential_integrity_100_when_consistent_mapping() -> None:
    """[UNIT-TEST] FR-009: a 1:1 plaintext→surrogate map (no collisions, fully
    consistent) yields referential_integrity == 1.0 and collision_rate == 0.0."""
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
    )
    assert pseudo.referential_integrity == 1.0
    assert pseudo.collision_rate == 0.0


def test_collision_rate_detects_two_plaintexts_sharing_one_surrogate() -> None:
    """[UNIT-TEST] FR-009 TEETH: two DISTINCT plaintexts mapping to the SAME
    surrogate is a collision → collision_rate > 0 (referential integrity broken)."""
    colliding = {
        "Jane Roe": "PERSON_0001",
        "John Doe": "PERSON_0001",  # collision: distinct plaintext, same surrogate
    }
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=colliding,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
    )
    assert pseudo.collision_rate > 0.0


# ---------------------------------------------------------------------------
# Anonymization family — irreversibility (aggregates Reid/Leakage/Canary)
# ---------------------------------------------------------------------------


def test_anonymization_clean_output_has_low_reidentification_risk() -> None:
    """[UNIT-TEST] A clean anonymization (original surfaces removed) → low
    re-identification risk and a high irreversibility score."""
    anon = AnonymizationScorer().score(
        labels=_labels(_ORIGINAL),
        original_text=_ORIGINAL,
        anonymized_text=_ANON_CLEAN,
        canary_strings=[_CANARY],
    )
    assert anon.reidentification_risk == 0.0
    assert anon.irreversibility_score >= 0.99


def test_anonymization_leaky_output_raises_reidentification_risk() -> None:
    """[UNIT-TEST] TEETH: a leaky anonymization (the original name survives) raises
    re-identification risk and lowers the irreversibility score."""
    anon = AnonymizationScorer().score(
        labels=_labels(_ORIGINAL),
        original_text=_ORIGINAL,
        anonymized_text=_ANON_LEAKY,
        canary_strings=[_CANARY],
    )
    assert anon.reidentification_risk > 0.0
    assert anon.irreversibility_score < 1.0


# ---------------------------------------------------------------------------
# A8 — determinism (AX-002)
# ---------------------------------------------------------------------------


def test_a8_scorers_are_deterministic_on_identical_inputs() -> None:
    """[PROPERTY-TEST] A8 / AX-002: two runs of each scorer on identical synthetic
    inputs yield equal scores."""
    anon1 = AnonymizationScorer().score(
        labels=_labels(_ORIGINAL),
        original_text=_ORIGINAL,
        anonymized_text=_ANON_CLEAN,
        canary_strings=[_CANARY],
    )
    anon2 = AnonymizationScorer().score(
        labels=_labels(_ORIGINAL),
        original_text=_ORIGINAL,
        anonymized_text=_ANON_CLEAN,
        canary_strings=[_CANARY],
    )
    assert anon1 == anon2

    pseudo1 = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
    )
    pseudo2 = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
    )
    assert pseudo1 == pseudo2


# ---------------------------------------------------------------------------
# A7 — anon≠pseudo wall preserved (AX-004 / FR-010) [AUDIT]
# ---------------------------------------------------------------------------


def test_a7_anonymization_score_carries_no_pseudonymization_field() -> None:
    """[AUDIT] A7 / AX-004: the anonymization sub-record never emits a
    pseudonymization-integrity number (and vice-versa). The two families do not
    leak each other's fields."""
    anon = AnonymizationScorer().score(
        labels=_labels(_ORIGINAL),
        original_text=_ORIGINAL,
        anonymized_text=_ANON_CLEAN,
        canary_strings=[_CANARY],
    )
    pseudo = PseudonymizationIntegrityScorer().score(
        pseudonym_map=_PSEUDO_MAP,
        authorized_key=_AUTHORIZED_KEY,
        reversal_attempts=[(_AUTHORIZED_KEY, True)],
    )
    anon_fields = set(vars(anon).keys())
    pseudo_fields = set(vars(pseudo).keys())
    # The anonymization family must not carry pseudonymization axes.
    for pseudo_axis in (
        "unauthorized_reversal_rate",
        "referential_integrity",
        "collision_rate",
        "key_state_separation_ok",
    ):
        assert pseudo_axis not in anon_fields
    # The pseudonymization family must not carry anonymization axes.
    for anon_axis in ("reidentification_risk", "leakage", "canary_exposure"):
        assert anon_axis not in pseudo_fields


# ---------------------------------------------------------------------------
# A2 / A8 — no-merge CI guard (THE headline AX-004 teeth) [SECURITY-TEST][AUDIT]
# ---------------------------------------------------------------------------

# Denylist of merged anon+pseudo de-id field / attribute / method names. A name
# that conflates the two families into ONE de-id number is forbidden by AX-004.
_MERGED_DEID_NAMES: frozenset[str] = frozenset(
    {
        "deid_score",
        "combined_deid_score",
        "merged_deid_score",
        "privacy_score",  # the historical merged field must NOT appear here
        "overall_deid_score",
        "unified_deid_score",
    }
)

_FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "pii_anon.swarm",
        "pii_anon.moe",
        "pii_anon.fusion",
        "pii_anon.policy",
    }
)


def _deid_families_path() -> Path:
    return Path(deid_families.__file__)


def _assigned_target_names(tree: ast.AST) -> set[str]:
    """Every assigned/annotated attribute name + dataclass field name in a tree."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def _scan_for_merged_names(source: str) -> set[str]:
    """Return any denylisted merged-de-id name appearing as an assigned target or
    function name in ``source`` (AST-based — cannot false-positive on prose)."""
    tree = ast.parse(source)
    return _assigned_target_names(tree) & _MERGED_DEID_NAMES


def test_a2_no_merged_deid_field_in_deid_families_module() -> None:
    """[SECURITY-TEST][AUDIT] A2 / AX-004 / FR-010: the deid_families module
    declares ZERO merged anon+pseudo de-id field/method (AST denylist scan)."""
    source = _deid_families_path().read_text(encoding="utf-8")
    offenders = _scan_for_merged_names(source)
    assert offenders == set(), f"merged de-id name(s) found in deid_families.py: {offenders}"


def test_a2_meta_guard_scan_is_non_vacuous() -> None:
    """[SECURITY-TEST][AUDIT] A2 META-GUARD: the no-merge scan is NOT vacuous — a
    SYNTHETIC source that DOES declare a merged ``deid_score`` field is flagged.
    Without this, the denylist scan could silently pass on a renamed denylist."""
    synthetic_merged = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Bad:\n"
        "    deid_score: float = 0.0\n"  # a forbidden merged field
    )
    offenders = _scan_for_merged_names(synthetic_merged)
    assert "deid_score" in offenders  # the scan WOULD catch a real violation


def test_a2_positive_separateness_both_families_addressable() -> None:
    """[AUDIT] A2 positive assertion: both families are SEPARATELY addressable as
    distinct dataclasses with distinct field sets (no single merged record)."""
    assert hasattr(deid_families, "AnonymizationScore")
    assert hasattr(deid_families, "PseudonymizationIntegrityScore")
    assert hasattr(deid_families, "DeidFamilyScores")
    # The carrier record has EXACTLY the two distinct sub-record fields.
    sample = DeidFamilyScores(
        anonymization=AnonymizationScore(
            reidentification_risk=0.0,
            leakage=0.0,
            canary_exposure=0.0,
            irreversibility_score=1.0,
        ),
        pseudonymization_integrity=PseudonymizationIntegrityScore(
            unauthorized_reversal_rate=0.0,
            referential_integrity=1.0,
            collision_rate=0.0,
            key_state_separation_ok=True,
            integrity_score=1.0,
        ),
    )
    assert set(vars(sample).keys()) == {"anonymization", "pseudonymization_integrity"}


def test_a8_deid_families_imports_nothing_from_detection_layers() -> None:
    """[PROPERTY-TEST] A8 / import-boundary: deid_families.py imports nothing from
    swarm/moe/fusion/policy (AST-based; mirrors test_rating_import_boundary)."""
    source = _deid_families_path().read_text(encoding="utf-8")
    tree = ast.parse(source)
    violations = [
        m
        for m in _imported_modules(tree)
        if any(m == root or m.startswith(root + ".") for root in _FORBIDDEN_IMPORT_ROOTS)
    ]
    assert not violations, f"deid_families.py must not import detection layers: {violations}"


def test_a7_two_families_do_not_import_each_other_in_a_merging_way() -> None:
    """[AUDIT] A7 / AX-004: the module defines both scorers WITHOUT one scorer
    importing/aggregating the other into a merged number — they are co-located but
    structurally independent (neither references the other's class in its own
    score() return)."""
    source = _deid_families_path().read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Locate the two scorer classes; assert neither instantiates the other.
    class_calls: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in (
            "AnonymizationScorer",
            "PseudonymizationIntegrityScorer",
        ):
            called: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    called.add(sub.func.id)
            class_calls[node.name] = called
    assert "PseudonymizationIntegrityScore" not in class_calls.get("AnonymizationScorer", set())
    assert "AnonymizationScore" not in class_calls.get(
        "PseudonymizationIntegrityScorer", set()
    )


# ---------------------------------------------------------------------------
# A9 — backward-compat / no public-API regression on SystemScorecard
# ---------------------------------------------------------------------------


def test_a9_scorecard_preserves_privacy_score_and_adds_two_distinct_fields() -> None:
    """[CONTRACT-TEST] A9: SystemScorecard keeps ``privacy_score`` (backward-compat
    public API) AND gains the two distinct additive family fields; all three are
    serialized in to_dict()."""
    from pii_anon.eval_framework.rating.scorecard import SystemScorecard

    sc = SystemScorecard(
        system_name="pii-anon",
        privacy_score=0.70,
        anonymization_score=0.95,
        pseudonymization_integrity_score=0.93,
        composite_score=0.80,
    )
    # The historical field is preserved (value unchanged, still addressable).
    assert sc.privacy_score == 0.70
    # The two new fields exist additively and are DISTINCT from privacy_score.
    assert sc.anonymization_score == 0.95
    assert sc.pseudonymization_integrity_score == 0.93

    d = sc.to_dict()
    assert d["privacy_score"] == 0.70  # backward-compat key preserved
    assert d["anonymization_score"] == 0.95
    assert d["pseudonymization_integrity_score"] == 0.93
    # composite_score numeric behaviour unchanged (reads whatever it is given).
    assert d["composite_score"] == 0.80


def test_a9_scorecard_defaults_preserve_existing_leaderboard_shape() -> None:
    """[CONTRACT-TEST] A9: the new fields default to 0.0 (additive — existing
    leaderboard JSON callers that never set them keep their shape), and the
    historical to_dict keys all remain present."""
    from pii_anon.eval_framework.rating.scorecard import SystemScorecard

    d = SystemScorecard(system_name="legacy").to_dict()
    for legacy_key in (
        "system_name", "available", "f1", "precision", "recall",
        "latency_p50_ms", "docs_per_hour", "privacy_score", "utility_score",
        "fairness_score", "composite_score", "elo_rating", "elo_rd",
        "samples", "evaluation_track", "license_name",
    ):
        assert legacy_key in d, f"legacy to_dict key dropped: {legacy_key}"
    assert d["anonymization_score"] == 0.0
    assert d["pseudonymization_integrity_score"] == 0.0


def test_a2_scorecard_has_no_merged_deid_field() -> None:
    """[SECURITY-TEST][AUDIT] A2 / AX-004: SystemScorecard carries the two DISTINCT
    family fields but NO single merged anon+pseudo de-id field (e.g.
    ``deid_score``/``combined_deid_score``). ``privacy_score`` is allowed to remain
    ONLY as the documented deprecated backward-compat field — it is NOT a NEW
    merged de-id field and is explicitly excluded from this scan."""
    from pii_anon.eval_framework.rating.scorecard import SystemScorecard

    field_names = set(SystemScorecard("x").__dataclass_fields__.keys())
    merged_new = {
        "deid_score",
        "combined_deid_score",
        "merged_deid_score",
        "overall_deid_score",
        "unified_deid_score",
    }
    assert field_names & merged_new == set()
    # Positive: BOTH distinct family fields are present and separately addressable.
    assert "anonymization_score" in field_names
    assert "pseudonymization_integrity_score" in field_names
