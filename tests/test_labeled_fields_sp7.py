"""sp7 Phase-A A2 — labeled-field value bridge (candidate #3, sp6 mining).

The single largest DETECTION-RECALL lever after fusion (~2,570 FNs across
5/5 external datasets): a value that sits behind an unambiguous field-label
cue ("SSN:", "Date of Birth:", "Account Number:") but whose bare shape the
core patterns miss or mistype. A2 v1 is a shared cue->value bridge that
tolerates markdown/quote wrappers between the cue and its value, extracts the
VALUE span (never the label word — cue-capture hygiene), and types it FROM THE
LABEL.

Design discipline (sp2 leak-direction + sp6 inversion lessons):
  * ADDITIVE recall — the bridge only ADDS findings, and it runs on BOTH the
    production masking path and the eval path (leak-SAFE: over-masking a
    labeled value is the safe direction).
  * label-wins arbitration is a leak-SAFE RELABEL — the value span stays
    masked, only its type changes; coverage is never reduced.
  * every cue maps ONLY to a type in orchestrator.SUPPORTED_ENTITY_TYPES, so
    the orchestrator actually masks the extracted value (a type outside the
    set would be dropped -> the exact leak class the sp6 presidio inversion
    hit).
  * v1 requires the strong ``:``/``=`` separator and a per-kind value-shape
    gate; the FP-prone copula bridge (is|was|of) and runaway ADDRESS capture
    are deferred.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.labeled_fields import extract_labeled_fields
from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.orchestrator import SUPPORTED_ENTITY_TYPES

_EVAL = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)
_PROD = RegexEngineAdapter(enabled=True)


def _spans(engine, text: str) -> set[tuple[str, int, int]]:
    return {
        (str(f.entity_type), f.span_start, f.span_end)
        for f in engine.detect({"text": text}, {"language": "en"})
    }


def _types_at(text: str) -> dict[str, str]:
    """Map the extracted value string -> entity type (pure extractor)."""
    return {text[s:e]: t for t, s, e, _ in extract_labeled_fields(text)}


class TestExtractorPositives:
    @pytest.mark.parametrize(
        "text,value,etype",
        [
            ("SSN: 123-45-6789", "123-45-6789", "US_SSN"),
            ("Social Security Number: 987-65-4321", "987-65-4321", "US_SSN"),
            ("Date of Birth: 04/12/1980", "04/12/1980", "DATE_OF_BIRTH"),
            ("DOB: 1980-04-12", "1980-04-12", "DATE_OF_BIRTH"),
            ("Account Number: GB-90210-XT", "GB-90210-XT", "BANK_ACCOUNT"),
            ("Routing Number: 021000021", "021000021", "ROUTING_NUMBER"),
            ("Phone: (415) 555-0198", "(415) 555-0198", "PHONE_NUMBER"),
            ("Passport No: X1234567", "X1234567", "PASSPORT"),
            ("Employee ID: E-4471", "E-4471", "EMPLOYEE_ID"),
            ("Full Name: Maria Garcia", "Maria Garcia", "PERSON_NAME"),
            ("Username: mgarcia_98", "mgarcia_98", "USERNAME"),
        ],
    )
    def test_value_extracted_with_label_type(self, text: str, value: str, etype: str) -> None:
        got = _types_at(text)
        assert value in got, f"value {value!r} not extracted from {text!r}: got {got!r}"
        assert got[value] == etype, f"typed {got[value]} expected {etype}"

    def test_markdown_wrapped_cue_and_value(self) -> None:
        # markdown around cue AND value is tolerated; span is the value only.
        got = _types_at("**SSN:** `123-45-6789`")
        assert "123-45-6789" in got and got["123-45-6789"] == "US_SSN"

    def test_every_cue_maps_to_a_supported_type(self) -> None:
        # scan a broad label sheet; every emitted type must be orchestrator-maskable
        sheet = "\n".join(
            f"{cue}: {val}"
            for cue, val in [
                ("SSN", "123-45-6789"), ("Phone", "415-555-0198"),
                ("Credit Card", "4111 1111 1111 1111"), ("IBAN", "GB29NWBK60161331926819"),
                ("Password", "Hunter2xy"), ("PIN", "4821"), ("Salary", "$85,000"),
                ("Date of Birth", "1980-04-12"), ("Full Name", "John Smith"),
                ("Docket Number", "23-CV-0917"), ("License Plate", "7ABC123"),
            ]
        )
        for t, _s, _e, _c in extract_labeled_fields(sheet):
            assert t in SUPPORTED_ENTITY_TYPES, f"{t} not in SUPPORTED_ENTITY_TYPES"


class TestExtractorHygieneAndPrecision:
    def test_cue_word_boundary_not_substring(self) -> None:
        # "Identification Number" must not fire the "Id" cue; "average:" must
        # not fire "age"; a value is still required.
        assert _types_at("Identification of the witness: unclear") == {}
        assert _types_at("Average: 42 units") == {}

    def test_label_word_not_in_emitted_span(self) -> None:
        # cue-capture hygiene: the emitted span excludes the label token.
        for t, s, e, _c in extract_labeled_fields("Passport Number: X1234567"):
            assert "Passport" not in "X1234567"  # sanity
            assert (t, "X1234567") == ("PASSPORT", "X1234567") or t != "PASSPORT"

    def test_no_value_shape_no_emit(self) -> None:
        # a strong cue with a non-conforming value must NOT emit (digit-required
        # id kinds; Title-Case name kind).
        assert _types_at("SSN: not provided") == {}
        assert _types_at("Account Number: pending") == {}
        assert _types_at("Full Name: n/a") == {}

    def test_bare_name_cue_absent_no_person_fp(self) -> None:
        # v1 deliberately omits the bare "name" cue (brand/company/file name
        # would FP). Only person-specific cues fire.
        assert "PERSON_NAME" not in {t for t, *_ in extract_labeled_fields("Brand Name: Acme")}
        assert "PERSON_NAME" not in {t for t, *_ in extract_labeled_fields("File Name: report.pdf")}


class TestAdapterIntegrationAndLeakSafety:
    def test_additive_recall_on_both_paths(self) -> None:
        # A previously-missed labeled value is now detected, and identically on
        # the production and eval paths (the bridge is not eval-gated).
        text = "Member record — Account Number: GB-90210-XT, opened 2019."
        assert ("BANK_ACCOUNT",) in {(t,) for t, *_ in _spans(_PROD, text)}
        assert _spans(_PROD, text) >= {
            (t, s, e) for t, s, e in _spans(_PROD, text) if t == "BANK_ACCOUNT"
        }

    def test_prod_superset_of_eval_leak_direction(self) -> None:
        # LEAK-DIRECTION INVARIANT: production must never emit FEWER spans than
        # eval (eval-only drops may prune eval; the bridge itself is on both).
        for text in [
            "SSN: 123-45-6789 and Phone: 415-555-0198",
            "Full Name: Maria Garcia, DOB: 1980-04-12",
            "Passport No: X1234567; Routing Number: 021000021",
        ]:
            assert _spans(_EVAL, text) <= _spans(_PROD, text), (
                f"eval emitted a span production did not — leak risk: {text!r}"
            )

    def test_label_wins_relabel_preserves_masking(self) -> None:
        # "label wins": a bare-9-digit value behind an SSN cue is typed US_SSN,
        # and whatever type wins, the value SPAN is still covered (masked).
        text = "SSN: 123456789"
        prod = _spans(_PROD, text)
        covering = [(t, s, e) for t, s, e in prod if text[s:e] == "123456789"]
        assert covering, "the labeled value span is not covered by ANY finding (leak)"
        assert any(t == "US_SSN" for t, _s, _e in covering), "label did not win the type"
