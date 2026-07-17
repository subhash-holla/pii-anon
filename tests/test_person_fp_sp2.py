"""sp2 dev-iteration 6 — remaining systematic PERSON_NAME FP/FN classes.

Dev-split evidence: PERSON_NAME's leftover false positives are dominantly
spans that ARE other entities' correct values (organizations, job titles,
health conditions, medications) plus document-header triples and street
names; and the iteration-1 second-token colon guard cost the dialogue form
("⟦Daniel Moore⟧: No further questions" — gold) an FN.
"""
from __future__ import annotations

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENGINE = RegexEngineAdapter(enabled=True)


_EVAL_ENGINE = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)


def _spans(text: str, entity_type: str) -> list[str]:
    return sorted(
        text[f.span_start : f.span_end]
        for f in _ENGINE.detect({"text": text}, {"language": "en"})
        if f.entity_type == entity_type
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


def _eval_spans(text: str, entity_type: str) -> list[str]:
    # The cross-type arbitration is EVAL-ONLY (production over-masks to stay
    # leak-safe; see test_sp2_remediation). These tests document the
    # benchmark-precision behaviour, so they use the eval adapter.
    return sorted(
        text[f.span_start : f.span_end]
        for f in _EVAL_ENGINE.detect({"text": text}, {"language": "en"})
        if f.entity_type == entity_type
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


class TestCrossTypeArbitration:
    def test_org_span_is_not_also_a_person(self) -> None:
        text = "Employee Robert Anderson works at Bluth Company. Thanks."
        assert "Bluth Company" in _eval_spans(text, "ORGANIZATION")
        assert "Bluth Company" not in _eval_spans(text, "PERSON_NAME")

    def test_job_title_span_is_not_also_a_person(self) -> None:
        text = "Current employer: Soylent Corp | Position: Medical Director | Annual income: $133,000"
        assert "Medical Director" in _eval_spans(text, "JOB_TITLE")
        assert "Medical Director" not in _eval_spans(text, "PERSON_NAME")

    def test_health_condition_span_is_not_also_a_person(self) -> None:
        text = "Patient presents with symptoms consistent with Acute Bronchitis."
        assert "Acute Bronchitis" in _eval_spans(text, "HEALTH_CONDITION")
        assert "Acute Bronchitis" not in _eval_spans(text, "PERSON_NAME")

    def test_medication_span_is_not_also_a_person(self) -> None:
        text = "Current medications include Albuterol Inhaler. Reports chest pain."
        assert "Albuterol Inhaler" in _eval_spans(text, "MEDICATION_NAME")
        assert "Albuterol Inhaler" not in _eval_spans(text, "PERSON_NAME")


class TestHeaderAndAddressFPs:
    def test_employer_tax_label_not_a_person(self) -> None:
        text = "Length of employment: 9 years\nEmployer Tax ID: 85-5503355"
        assert all("Employer" not in s for s in _spans(text, "PERSON_NAME"))

    def test_document_header_triple_not_a_person(self) -> None:
        text = "Wire Transfer Confirmation\nFrom: Melissa Martin (Acct: 6613168096)"
        names = _spans(text, "PERSON_NAME")
        assert "Melissa Martin" in names, names
        assert all("Confirmation" not in s for s in names), names

    def test_street_name_after_house_number_not_a_person(self) -> None:
        text = "Wages: $131,000\nAddress: 9953 Dogwood Ct, Greenville, AZ 63153"
        assert all("Dogwood" not in s for s in _spans(text, "PERSON_NAME"))


class TestDialogueSpeakerForm:
    def test_speaker_colon_name_is_detected(self) -> None:
        # "⟦Daniel Moore⟧: No further questions" — gold includes the name;
        # the generic second-token colon guard must NOT reject a non-label
        # surname before a colon (only known field-label words are guarded).
        text = "Complaint submitted to Dunder Mifflin.\n\nDaniel Moore: No further questions at this time."
        assert "Daniel Moore" in _spans(text, "PERSON_NAME")

    def test_field_label_second_token_still_guarded(self) -> None:
        # The original absorption class stays fixed: a KNOWN field-label
        # word directly before a colon is the next field, not a surname.
        text = "Name:   Donald Rodriguez  \t Email: donald.rodriguez56@protonmail.com"
        names = _spans(text, "PERSON_NAME")
        assert "Donald Rodriguez" in names, names
        assert all("Email" not in s for s in names), names
