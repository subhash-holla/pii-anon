"""sp2 dev-iteration 1 — PERSON_NAME exact-extent fixes (strict-match scoring).

Every case here is a REAL failure shape observed on the pii-anon-eval-data
dev split (sp2 iteration-0 inspection). The external assessment scores
strict ``(start, end, entity_type)`` matches, so extent bugs cost an FN
AND an FP simultaneously:

1. ``_PERSON_KEYWORD`` was case-sensitive — "Patient Ronald Jackson" never
   produced the correct-extent keyword match while ``_PERSON_FULL_NAME``
   absorbed the role word.
2. ``_PERSON_FULL_NAME``'s first-word exclusion lacked the corpus role nouns
   (Patient / Contact / Applicant / Resident / ...).
3. The optional 2nd/3rd name token absorbed the NEXT field's Title-Case
   label across tabs ("Donald Rodriguez␉Email: ..." → span ended at "Email").
4. No pattern matched the bare field-label form "Name: <Name>".
5. Nested/duplicate same-type emissions ("Melissa" inside "Melissa White";
   two patterns emitting the identical span) are pure FPs under multiset
   strict scoring.
"""
from __future__ import annotations

from collections import Counter

from pii_anon.engines.regex_adapter import RegexEngineAdapter


def _person_spans(text: str) -> list[tuple[int, int, str]]:
    engine = RegexEngineAdapter(enabled=True)
    return sorted(
        (f.span_start, f.span_end, text[f.span_start : f.span_end])
        for f in engine.detect({"text": text}, {"language": "en"})
        if f.entity_type == "PERSON_NAME"
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


class TestRoleWordExtent:
    def test_capitalized_patient_keyword_yields_name_only_extent(self) -> None:
        text = "Patient Ronald Jackson (MRN: MRN-1010506) presents today."
        spans = _person_spans(text)
        assert ("Ronald Jackson" in [s[2] for s in spans]), spans
        assert all(not s[2].startswith("Patient") for s in spans), spans

    def test_contact_prefix_not_absorbed(self) -> None:
        text = "Bluth Company. Contact Robert Anderson at robert.anderson92@fastmail.com"
        spans = _person_spans(text)
        assert "Robert Anderson" in [s[2] for s in spans], spans
        assert all(not s[2].startswith("Contact ") for s in spans), spans

    def test_applicant_prefix_not_absorbed(self) -> None:
        text = "Underwriter notes: Applicant Richard Lopez meets initial criteria."
        spans = _person_spans(text)
        assert "Richard Lopez" in [s[2] for s in spans], spans
        assert all(not s[2].startswith("Applicant") for s in spans), spans


class TestFieldLabelAbsorption:
    def test_following_label_token_not_absorbed_across_tabs(self) -> None:
        text = "Name:   Donald Rodriguez  \t Email: donald.rodriguez56@protonmail.com"
        spans = _person_spans(text)
        assert "Donald Rodriguez" in [s[2] for s in spans], spans
        assert all("Email" not in s[2] for s in spans), spans

    def test_bare_name_label_form_matches(self) -> None:
        # NB: classic placeholder names ("Jane Doe") sit on the deny list by
        # design — the fixture must use a real-shaped name.
        text = "Full Name - Marcus Webb"
        spans = _person_spans(text)
        assert "Marcus Webb" in [s[2] for s in spans], spans

    def test_username_label_does_not_trigger_name_label_pattern(self) -> None:
        # "Username: jdoe" — the labeled-name pattern must not see the "name"
        # inside "Username" (lowercase value also fails the capture shape).
        spans = _person_spans("Username: jdoe55, Status: active")
        assert spans == [], spans


class TestNestedDuplicateSuppression:
    def test_contained_same_type_span_is_dropped(self) -> None:
        text = "Record for Melissa White, Cryptocurrency Address: 0xc6f654c18ed"
        spans = _person_spans(text)
        texts = [s[2] for s in spans]
        assert "Melissa White" in texts, spans
        assert "Melissa" not in texts, spans

    def test_identical_spans_are_emitted_once(self) -> None:
        # Multiple person patterns legitimately match the same mention; the
        # multiset strict scorer counts the second identical span as an FP.
        text = "patient Maria Lopez was discharged."
        counts = Counter(_person_spans(text))
        assert counts, "expected at least one PERSON_NAME span"
        assert max(counts.values()) == 1, counts


class TestNonNameRejection:
    def test_plural_systems_job_title_not_a_person(self) -> None:
        spans = _person_spans("Job: Systems Administrator. Marital Status: Married.")
        assert all("Systems" not in s[2] for s in spans), spans
