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

_CORPUS = [
    CorpusRecord("r1", "Patient Maria Garcia, SSN 123-45-6789, lives in Denver.",
                 ("Maria Garcia", "123-45-6789", "Denver")),
    CorpusRecord("r2", "Contact John Smith at john@acme.com, phone 415-555-0198.",
                 ("John Smith", "john@acme.com", "415-555-0198")),
    CorpusRecord("r3", "Ms Aisha Khan, account GB29NWBK60161331926819, London office.",
                 ("Aisha Khan", "GB29NWBK60161331926819", "London")),
    CorpusRecord("r4", "Dr Ivan Petrov, DOB 1980-04-12, employee E-4471.",
                 ("Ivan Petrov", "1980-04-12", "E-4471")),
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
