"""STANDING contract: every PATTERN_REGISTRY label must be reachable as eval truth.

The DEA_NUMBER/NPI_NUMBER incident class (SP1 Tasks 2-3, commits bd979f0 /
3f52af3): a ``PatternSpec`` emitted detections under a label the eval could
never match against ground truth, so the pattern did zero census work — its
truth spans scored FN forever while the detections were either noise-dropped
(the FP-universe rule, ``competitor_compare.py:1802-1809``) or silently
inflated FP. This file pins that the incident class can never silently recur,
and that the loader alias-map copies never drift apart or away from the
census authority.

Label flow (what "reachable as eval truth" means)
-------------------------------------------------
1. Corpus truth (raw DATA v2 annotation types) is normalized at LOAD time by
   ``pii_anon.benchmarks.datasets._EVAL_DATA_ENTITY_TYPE_MAP``.
2. At eval time BOTH the loaded truth and every detection are normalized by
   the census authority ``_normalize_entity_type`` in the PINNED
   ``pii_anon.evaluation.competitor_compare`` (imported read-only here).
3. Truth normalizing to ``_BENCHMARK_IGNORE`` is excluded from scoring.

So the scoreable truth universe is::

    { _normalize_entity_type(LOADER_MAP.get(t, t)) for t in corpus types }
        - { "_BENCHMARK_IGNORE" }

and a registry label L is reachable iff ``_normalize_entity_type(L)`` lands
in that universe. NOTE the loader threading is load-bearing: corpus
``VEHICLE_IDENTIFICATION_NUMBER`` -> loader ``VIN`` -> authority passthrough
``VIN`` is scoreable truth (baseline-n2000 carries a VIN row) even though the
authority's own entry for the raw corpus name maps it to ignore — computing
reachability from the authority map's RHS alone would falsely flag VIN.
"""

from __future__ import annotations

import pytest

from pii_anon.benchmarks.datasets import (
    _EVAL_DATA_ENTITY_TYPE_MAP as _BENCHMARKS_LOADER_MAP,
)
from pii_anon.engines.regex.patterns import PATTERN_REGISTRY
from pii_anon.eval_framework.datasets.schema import (
    _EVAL_DATA_ENTITY_TYPE_MAP as _SCHEMA_LOADER_MAP,
)

# Read-only import of the PINNED census authority (never edit that module).
from pii_anon.evaluation.competitor_compare import _normalize_entity_type

_BENCHMARK_IGNORE = "_BENCHMARK_IGNORE"

# ── The DATA v2 corpus annotation-type universe ────────────────────────────
# Provenance: pii_anon_datasets 2.2.0, packaged splits/*.jsonl.gz — census
# re-derived 2026-07-10 (sp3 v2.2.0 re-baseline) by exhaustive scan of every
# annotation's entity_type across all splits (66 types = the prior 63 + the 3
# GDPR Article-9 special-category additions; nothing removed).
# If the dataset package revs, re-derive and update this constant in the same
# change that bumps the dataset dependency.
DATA_V2_CORPUS_ENTITY_TYPES: frozenset[str] = frozenset({
    "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "SOCIAL_SECURITY_NUMBER",
    "ORGANIZATION_NAME", "STREET_ADDRESS", "DATE_OF_BIRTH", "TIMESTAMP",
    "BANK_ACCOUNT_NUMBER", "MEDICAL_RECORD_NUMBER", "CREDIT_CARD_NUMBER",
    "PASSPORT_NUMBER", "IP_ADDRESS", "EMPLOYEE_ID", "IBAN",
    "DRIVER_LICENSE_NUMBER", "BANK_ROUTING_NUMBER", "NATIONAL_ID_NUMBER",
    "INSURANCE_POLICY_NUMBER", "TAX_ID", "SWIFT_BIC_CODE", "API_KEY",
    "PASSWORD", "INVOICE_NUMBER", "PIN", "CVV", "JOB_TITLE",
    "MEDICATION_NAME", "HEALTH_CONDITION", "SALARY", "NPI_NUMBER",
    "HEALTH_INSURANCE_ID", "USERNAME", "MAC_ADDRESS", "COURT_CASE_NUMBER",
    "BAR_NUMBER", "DOCKET_NUMBER", "PROCEDURE_NAME", "LICENSE_PLATE",
    "LOCATION_NAME", "DEA_NUMBER", "CREDIT_CARD_FRAGMENT", "BIOMETRIC_ID",
    "CRYPTOCURRENCY_ADDRESS", "AUTHENTICATION_TOKEN", "DEVICE_IDENTIFIER",
    "VISA_NUMBER", "PRESCRIPTION_NUMBER", "URL", "POSTAL_CODE",
    "VEHICLE_IDENTIFICATION_NUMBER", "SOCIAL_MEDIA_HANDLE",
    "LATITUDE_LONGITUDE", "GENDER", "EDUCATION_LEVEL", "MARITAL_STATUS",
    "ETHNICITY", "POLITICAL_OPINION", "RELIGIOUS_BELIEF", "NATIONALITY",
    "AGE", "HOUSEHOLD_SIZE", "VEHICLE_MODEL",
    # sp3 v2.2.0 rev: the 3 GDPR Article-9 special-category additions
    # (taxonomy 63 -> 66). Re-derived by an exhaustive scan of every
    # annotation entity_type across all packaged 2.2.0 splits (2026-07-10).
    "GENETIC_DATA", "SEXUAL_ORIENTATION", "TRADE_UNION_MEMBERSHIP",
})

# ── Documented allowlist: registry labels allowed to be census-unreachable ─
# Every entry needs exactly one verified reason. Adding a label here is an
# explicit, reviewed decision to keep a production-valid detection that can
# never earn census credit — NOT a way to silence the test.
ALLOWED_NON_CORPUS_LABELS: dict[str, str] = {
    # No matching corpus surface exists (detections noise-dropped by the
    # FP-universe rule — competitor_compare.py:1802-1809):
    "AADHAAR": "India national ID — the corpus carries no Aadhaar annotations",
    "CANADIAN_SIN": "Canadian SIN — the corpus carries no SIN annotations",
    "UK_NI_NUMBER": "UK National Insurance — the corpus carries none",
    "JWT_TOKEN": "corpus folds JWTs into AUTHENTICATION_TOKEN, which the "
                 "authority maps to _BENCHMARK_IGNORE",
    "SWIFT_BIC": "corpus SWIFT_BIC_CODE truth maps to _BENCHMARK_IGNORE",
    "ZIP_CODE": "distinct production masking label (transforms/policies key "
                "generalize-keep-3 on it); corpus POSTAL_CODE truth maps to "
                "ADDRESS; ZIP_CODE detections are noise-dropped at eval",
    # Authority maps the label itself to _BENCHMARK_IGNORE (census-invisible
    # by design — dropped in _normalize_findings before scoring):
    "DATE_TIME": "generic datetimes; corpus TIMESTAMP truth is ignored",
    "GPS_COORDINATES": "corpus LATITUDE_LONGITUDE maps to _BENCHMARK_IGNORE "
                       "in both the loader and the authority",
    "URL_WITH_PII": "corpus labels plain URL, which the authority ignores",
    "DATE_ISO": "no corpus DATE_ISO type; authority ignores the label "
                "(historic 115-FP suppression note at patterns.py ~line 220)",
    # Corpus truth for these types EXISTS but the PINNED authority maps the
    # type to _BENCHMARK_IGNORE (its stale 'without detection support'
    # block — patterns now exist), so truth is census-excluded and the
    # detections are census-invisible. Un-ignoring them requires editing the
    # PINNED authority (out of scope here; a census-coverage opportunity):
    "AGE": "corpus AGE truth exists; authority maps AGE -> ignore",
    "API_KEY": "corpus API_KEY truth exists; authority maps it -> ignore",
    "PASSWORD": "corpus PASSWORD truth exists; authority maps it -> ignore",
    "SALARY": "corpus SALARY truth exists; authority maps it -> ignore",
    "INVOICE_NUMBER": "corpus truth exists; authority maps it -> ignore",
    "INSURANCE_POLICY_NUMBER": "corpus truth exists; authority maps it "
                               "-> ignore",
    "COURT_CASE_NUMBER": "corpus truth exists; authority maps it -> ignore "
                         "(BAR_NUMBER/DOCKET_NUMBER pass through and ARE "
                         "scoreable — only this sibling is ignored)",
    # sp2 external-coverage tranche (2026-06-12): corpus truth exists for
    # every one of these, but the PINNED authority maps each to ignore, so
    # they are census-invisible internally. They earn EXTERNAL credit via
    # the pii-anon-eval-data baselines harness (its adapter LABEL_MAP maps
    # each native label to the same-named canonical-63 type, where it
    # scores against real gold). Un-ignoring internally would require
    # editing the pinned authority — out of scope, census-coverage
    # opportunity noted.
    "AUTHENTICATION_TOKEN": "sp3: corpus AUTHENTICATION_TOKEN truth exists "
                            "(bearer/base64/JWT auth tokens); authority maps "
                            "it -> ignore. Earns external credit via the DATA "
                            "harness LABEL_MAP identity entry",
    "BIOMETRIC_ID": "corpus truth exists; authority maps it -> ignore",
    "CREDIT_CARD_FRAGMENT": "corpus truth exists; authority maps it -> ignore",
    "DEVICE_IDENTIFIER": "corpus truth exists; authority maps it -> ignore",
    "EDUCATION_LEVEL": "corpus truth exists; authority maps it -> ignore",
    "ETHNICITY": "corpus truth exists; authority maps it -> ignore",
    "GENDER": "corpus truth exists; authority maps it -> ignore",
    "HEALTH_CONDITION": "corpus truth exists; authority maps it -> ignore",
    "HEALTH_INSURANCE_ID": "corpus truth exists; authority maps it -> ignore",
    "HOUSEHOLD_SIZE": "corpus truth exists; authority maps it -> ignore",
    "JOB_TITLE": "corpus truth exists; authority maps it -> ignore",
    "MARITAL_STATUS": "corpus truth exists; authority maps it -> ignore",
    "MEDICATION_NAME": "corpus truth exists; authority maps it -> ignore",
    "NATIONALITY": "corpus truth exists; authority maps it -> ignore",
    "POLITICAL_OPINION": "corpus truth exists; authority maps it -> ignore",
    "PRESCRIPTION_NUMBER": "corpus truth exists; authority maps it -> ignore",
    "PROCEDURE_NAME": "corpus truth exists; authority maps it -> ignore",
    "RELIGIOUS_BELIEF": "corpus truth exists; authority maps it -> ignore",
    "VEHICLE_MODEL": "corpus truth exists; authority maps it -> ignore",
    # sp3 NOTE: the 3 GDPR Article-9 additions (SEXUAL_ORIENTATION /
    # TRADE_UNION_MEMBERSHIP / GENETIC_DATA) are NOT allowlisted — the 2.2.0
    # census authority scores them, so they are census-REACHABLE and the
    # reachability contract guards them directly (internal + external credit).
}


def _eval_truth_canonicals() -> frozenset[str]:
    """The scoreable truth universe: corpus -> loader map -> authority."""
    canonicals = set()
    for raw in DATA_V2_CORPUS_ENTITY_TYPES:
        after_loader = _BENCHMARKS_LOADER_MAP.get(raw, raw)
        canon = _normalize_entity_type(after_loader)
        if canon != _BENCHMARK_IGNORE:
            canonicals.add(canon)
    return frozenset(canonicals)


def test_every_registered_label_is_reachable_as_eval_truth() -> None:
    """No PatternSpec may emit a label the eval can never match (DEA-class).

    A label is reachable iff its authority-normalized form is in the
    scoreable truth universe (labels that NORMALIZE to a reachable canonical,
    e.g. IBAN -> BANK_ACCOUNT, are reachable). Unreachable labels do zero
    census work: their detections are noise-dropped or counted FP, and any
    corpus truth they should have matched scores FN on every record.
    """
    reachable = _eval_truth_canonicals()
    unreachable = {
        spec.entity_type
        for spec in PATTERN_REGISTRY
        if spec.entity_type not in ALLOWED_NON_CORPUS_LABELS
        and _normalize_entity_type(spec.entity_type) not in reachable
    }
    assert not unreachable, (
        f"DEA_NUMBER/NPI_NUMBER incident class: PATTERN_REGISTRY labels "
        f"{sorted(unreachable)} are NOT reachable as eval truth — every "
        f"detection they emit is census-invisible or a guaranteed FP, and "
        f"the truth they should match scores FN forever. Fix by relabeling "
        f"the PatternSpec to a canonical in {sorted(reachable)} (the "
        f"DEA/NPI fix pattern), or add a documented entry to "
        f"ALLOWED_NON_CORPUS_LABELS with the verified reason, or, if the "
        f"pii_anon_datasets dependency revved, re-derive "
        f"DATA_V2_CORPUS_ENTITY_TYPES per its provenance comment."
    )


def test_allowlist_entries_stay_honest() -> None:
    """Anti-rot: allowlist entries must be real registry labels that are
    STILL unreachable. If a label becomes reachable (corpus/authority
    evolution) or leaves the registry, its entry must be deleted — a stale
    allowlist would silently re-admit the DEA-class bug for that label."""
    registry_labels = {spec.entity_type for spec in PATTERN_REGISTRY}
    reachable = _eval_truth_canonicals()
    stale_not_registered = set(ALLOWED_NON_CORPUS_LABELS) - registry_labels
    assert not stale_not_registered, (
        f"ALLOWED_NON_CORPUS_LABELS entries {sorted(stale_not_registered)} "
        f"are not PATTERN_REGISTRY labels — delete the stale entries."
    )
    now_reachable = {
        label
        for label in ALLOWED_NON_CORPUS_LABELS
        if _normalize_entity_type(label) in reachable
    }
    assert not now_reachable, (
        f"ALLOWED_NON_CORPUS_LABELS entries {sorted(now_reachable)} are now "
        f"reachable as eval truth — remove them from the allowlist so the "
        f"reachability contract guards them again."
    )


# ── Map-copy consistency ────────────────────────────────────────────────────

def test_loader_map_copies_are_identical() -> None:
    """benchmarks/datasets.py and eval_framework/datasets/schema.py carry
    copies of the loader alias map that MUST stay byte-equal — the schema
    copy documents itself as a mirror of the benchmarks copy."""
    assert _BENCHMARKS_LOADER_MAP == _SCHEMA_LOADER_MAP, (
        "Loader alias-map copies drifted: "
        f"benchmarks-only={ {k: v for k, v in _BENCHMARKS_LOADER_MAP.items() if _SCHEMA_LOADER_MAP.get(k) != v} } "
        f"schema-only={ {k: v for k, v in _SCHEMA_LOADER_MAP.items() if _BENCHMARKS_LOADER_MAP.get(k) != v} }"
    )


# Loader entries whose RHS deliberately diverges from what the PINNED census
# authority would do with the same RAW key, pinned EXACTLY as
# key -> (loader_value, authority_value_for_raw_key).
#
# Why these divergences are CORRECT and must NOT be "fixed" by syncing the
# editable loader copies to the authority: the loader runs FIRST and only on
# TRUTH, so its RHS (a canonical the authority preserves — see the
# fixed-point assertion below) is what the census scores against; the
# authority's entry for the same RAW corpus name only ever fires on a
# DETECTOR that emits the raw name (a suppress-don't-FP safety net — our
# registry emits the canonicals). Syncing loader -> ignore would DELETE live
# truth from the census: baseline-n2000.json carries VIN TP=5, MAC_ADDRESS
# TP=37, MEDICAL_RECORD_NUMBER TP=85 rows that these loader entries feed,
# and corpus CREDIT_CARD_FRAGMENT spans score as CREDIT_CARD truth — a
# measurement-affecting regression, the exact opposite of this contract.
DOCUMENTED_LOADER_AUTHORITY_DIVERGENCES: dict[str, tuple[str, str]] = {
    "CREDIT_CARD_FRAGMENT": ("CREDIT_CARD", _BENCHMARK_IGNORE),
    "DEVICE_IDENTIFIER": ("MAC_ADDRESS", _BENCHMARK_IGNORE),
    "HEALTH_INSURANCE_ID": ("MEDICAL_RECORD_NUMBER", _BENCHMARK_IGNORE),
    "VEHICLE_IDENTIFICATION_NUMBER": ("VIN", _BENCHMARK_IGNORE),
}


def test_loader_map_agrees_with_census_authority() -> None:
    """Every loader entry must agree with the authority's normalization of
    the same key (``_normalize_entity_type(key) == value``), except the
    exactly-pinned documented divergences. Also: every loader RHS must be a
    fixed point of the authority, so loaded truth is already canonical and
    the authority never re-maps it."""
    stale_documented = set(DOCUMENTED_LOADER_AUTHORITY_DIVERGENCES) - set(
        _BENCHMARKS_LOADER_MAP
    )
    assert not stale_documented, (
        f"DOCUMENTED_LOADER_AUTHORITY_DIVERGENCES keys {sorted(stale_documented)} "
        f"are no longer loader-map entries — delete the stale pins."
    )
    undocumented = {}
    for key, value in _BENCHMARKS_LOADER_MAP.items():
        authority_value = _normalize_entity_type(key)
        documented = DOCUMENTED_LOADER_AUTHORITY_DIVERGENCES.get(key)
        if documented is not None:
            assert (value, authority_value) == documented, (
                f"Documented divergence for {key} changed shape: now "
                f"loader->{value}, authority->{authority_value}, "
                f"documented {documented}. Re-verify and re-pin."
            )
        elif authority_value != value:
            undocumented[key] = (value, authority_value)
    assert not undocumented, (
        f"Loader map drifted from the census authority with NO documented "
        f"reason: {undocumented} (key: (loader_value, authority_value)). "
        f"Either sync the editable loader copies to the authority, or pin "
        f"the divergence in DOCUMENTED_LOADER_AUTHORITY_DIVERGENCES with "
        f"the verified truth-flow evidence."
    )
    not_fixed_point = {
        v: _normalize_entity_type(v)
        for v in set(_BENCHMARKS_LOADER_MAP.values())
        if _normalize_entity_type(v) != v
    }
    assert not not_fixed_point, (
        f"Loader RHS values are not authority fixed points: "
        f"{not_fixed_point} — loaded truth would be silently re-mapped at "
        f"eval time."
    )


def test_corpus_census_constant_matches_packaged_dataset_version() -> None:
    """The frozen 66-type census is valid for pii_anon_datasets 2.2.0 ONLY.

    If this fails after a dataset rev: re-derive DATA_V2_CORPUS_ENTITY_TYPES
    from the new corpus per the constant's provenance comment, update the
    constant AND this pin in the same change.
    """
    pytest.importorskip("pii_anon_datasets")
    import importlib.metadata

    import pii_anon_datasets

    assert pii_anon_datasets.__version__ == "2.2.0", (
        f"DATA_V2_CORPUS_ENTITY_TYPES was frozen for pii_anon_datasets 2.2.0 "
        f"but the installed module reports {pii_anon_datasets.__version__!r}. "
        f"Re-derive DATA_V2_CORPUS_ENTITY_TYPES from the new corpus per its "
        f"provenance comment and update this pin in the same change. "
        f"(pip metadata: {importlib.metadata.version('pii-anon-datasets')!r})"
    )
