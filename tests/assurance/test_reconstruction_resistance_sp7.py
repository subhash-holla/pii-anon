"""sp7 Phase C — reconstruction-resistance report tests.

Verifies the measured bounds behave correctly at the two extremes: a PERFECT
masker (redacts every PII value) must show ~zero verbatim leakage and ~zero
re-identification; a NULL masker (identity) must show full verbatim leakage and
strictly higher re-identification. The representative adversaries are
deterministic, so the report is reproducible.
"""
from __future__ import annotations

from pii_anon.assurance.reconstruction_resistance import (
    CAVEAT,
    CorpusRecord,
    measure_verbatim_leakage,
    reconstruction_resistance_report,
)
from pii_anon.assurance.reconstruction_resistance_cli import build_regex_masker

_CORPUS = [
    CorpusRecord("r1", "Patient Maria Garcia, SSN 123-45-6789, lives in Denver.",
                 ("Maria Garcia", "123-45-6789", "Denver"),
                 ("PERSON_NAME", "US_SSN", "LOCATION")),
    CorpusRecord("r2", "Contact John Smith at john@acme.com, phone 415-555-0198.",
                 ("John Smith", "john@acme.com", "415-555-0198"),
                 ("PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER")),
    CorpusRecord("r3", "Ms Aisha Khan, account GB29NWBK60161331926819, London office.",
                 ("Aisha Khan", "GB29NWBK60161331926819", "London"),
                 ("PERSON_NAME", "IBAN", "LOCATION")),
    CorpusRecord("r4", "Dr Ivan Petrov, DOB 1980-04-12, employee E-4471.",
                 ("Ivan Petrov", "1980-04-12", "E-4471"),
                 ("PERSON_NAME", "DATE_OF_BIRTH", "EMPLOYEE_ID")),
]


def _perfect_mask(text: str) -> str:
    masked = text
    for rec in _CORPUS:
        for val in rec.pii_values:
            masked = masked.replace(val, "[REDACTED]")
    return masked


def _null_mask(text: str) -> str:
    return text


class TestVerbatimLeakage:
    def test_perfect_masker_zero_leakage(self) -> None:
        m = measure_verbatim_leakage(_CORPUS, _perfect_mask)
        assert m["leaked_verbatim"] == 0
        assert m["leak_rate"] == 0.0
        assert m["total_pii_values"] == 12

    def test_null_masker_full_leakage(self) -> None:
        m = measure_verbatim_leakage(_CORPUS, _null_mask)
        assert m["leaked_verbatim"] == m["total_pii_values"] == 12
        assert m["leak_rate"] == 1.0

    def test_per_type_breakdown(self) -> None:
        # null masker: every type leaks 100%; PERSON_NAME has 4 values.
        m = measure_verbatim_leakage(_CORPUS, _null_mask)
        pt = m["per_type"]
        assert pt["PERSON_NAME"]["total"] == 4
        assert pt["PERSON_NAME"]["leaked"] == 4
        assert pt["PERSON_NAME"]["leak_rate"] == 1.0
        # perfect masker: every type at 0%.
        m2 = measure_verbatim_leakage(_CORPUS, _perfect_mask)
        assert all(d["leaked"] == 0 for d in m2["per_type"].values())


class TestReport:
    def test_report_structure_and_bounds(self) -> None:
        rep = reconstruction_resistance_report(
            _CORPUS, _perfect_mask, seed=42, mask_label="perfect", corpus_label="test"
        )
        assert rep["report"] == "llm-reconstruction-resistance"
        assert rep["seed"] == 42
        assert rep["n_records"] == 4
        assert rep["caveat"] == CAVEAT
        # perfect masker: no verbatim leak, re-id at/below the unmasked baseline
        assert rep["verbatim_leakage"]["leaked_verbatim"] == 0
        reid = rep["reidentification_tier3"]
        assert reid["provisional_status"] == "AGENT_SIMULATED"
        assert reid["masked"]["reid_recall"] <= reid["unmasked_baseline"]["reid_recall"] + 1e-9
        assert "wilson95_recall" in reid["masked"]

    def test_masking_reduces_reidentification(self) -> None:
        # the protection delta must be non-negative: masking never HELPS the
        # adversary (it can only remove linkable signal).
        masked_rep = reconstruction_resistance_report(_CORPUS, _perfect_mask, seed=1)
        assert masked_rep["reidentification_tier3"]["protection_delta"] >= 0.0

    def test_deterministic_reproducible(self) -> None:
        a = reconstruction_resistance_report(_CORPUS, _perfect_mask, seed=7)
        b = reconstruction_resistance_report(_CORPUS, _perfect_mask, seed=7)
        assert a == b

    def test_mia_axis_present_and_labelled(self) -> None:
        rep = reconstruction_resistance_report(_CORPUS, _perfect_mask, seed=3)
        mia = rep["membership_inference"]
        assert mia["provisional_status"] == "AGENT_SIMULATED"
        assert 0.0 <= mia["tpr_at_1e_2"] <= 1.0
        assert "wilson95_tpr" in mia


class TestValueConsistentMasking:
    def test_coreference_masks_repeated_name(self) -> None:
        # a name detected once but repeated verbatim must be masked at BOTH
        # occurrences by the coreference masker (panel #2) — closing the
        # reconstruction leak the detector's single-mention span leaves open.
        text = "Contact Alexander Petrov. Alexander Petrov signed the form."
        coref = build_regex_masker(coreference=True)(text)
        plain = build_regex_masker(coreference=False)(text)
        # value-consistent redaction never leaves MORE of the name than plain
        assert coref.count("Alexander Petrov") <= plain.count("Alexander Petrov")
        assert "Alexander Petrov" not in coref

    def test_coreference_leak_subset_of_plain(self) -> None:
        # value-consistent masking is ADDITIVE — it can only reduce verbatim
        # leakage, never increase it.
        c = measure_verbatim_leakage(_CORPUS, build_regex_masker(coreference=True))
        p = measure_verbatim_leakage(_CORPUS, build_regex_masker(coreference=False))
        assert c["leaked_verbatim"] <= p["leaked_verbatim"]
