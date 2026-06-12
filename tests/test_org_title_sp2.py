"""sp2 dev-iteration 2 — title-prefix PERSON extents + ORGANIZATION hygiene.

Observed failure shapes on the pii-anon-eval-data dev split (strict matching):

1. Title-prefix person patterns emitted "Dr. Christopher Allen" INCLUDING the
   honorific while gold is name-only — with same-type containment dedup the
   wrong container now wins, so the extent bug costs recall directly.
2. ORGANIZATION patterns matched across newlines and sentence periods
   ("Bluth Company. Contact Robert Anderson", "Nordic Analytics\\nEIN") —
   the token atom allowed '.' and separators used \\s+.
3. _ORGANIZATION_CONTEXT was fully case-insensitive, so the CAPTURE accepted
   arbitrary case, and noun keywords without a colon captured the next
   field's label ("Employer Tax ID:" → "Tax ID").
"""
from __future__ import annotations

from pii_anon.engines.regex_adapter import RegexEngineAdapter


def _spans(text: str, entity_type: str) -> list[str]:
    engine = RegexEngineAdapter(enabled=True)
    return sorted(
        text[f.span_start : f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if f.entity_type == entity_type
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


class TestTitlePrefixExtent:
    def test_doctor_title_excluded_from_span(self) -> None:
        text = "Referring Physician: Dr. Karen Anderson | NPI: 3705585218"
        names = _spans(text, "PERSON_NAME")
        assert "Karen Anderson" in names, names
        assert all(not n.startswith("Dr") for n in names), names

    def test_signed_by_doctor_name_only(self) -> None:
        text = "Electronically signed by Dr. Robert Torres"
        names = _spans(text, "PERSON_NAME")
        assert "Robert Torres" in names, names
        assert all(not n.startswith("Dr") for n in names), names

    def test_title_with_bare_surname_keeps_title(self) -> None:
        # Annotation convention is split by shape: title + bare surname keeps
        # the honorific ("⟦Ms. Davis⟧"), title + full name drops it
        # ("Dr. ⟦Karen Anderson⟧" — tested above).
        text = "I spoke with Ms. Davis and resolved the complaint."
        names = _spans(text, "PERSON_NAME")
        assert "Ms. Davis" in names, names
        assert "Davis" not in names, names


class TestOrganizationHygiene:
    def test_org_does_not_cross_sentence_period(self) -> None:
        text = (
            "Employee Robert Anderson works at Bluth Company. "
            "Contact Robert Anderson at robert.anderson92@fastmail.com"
        )
        orgs = _spans(text, "ORGANIZATION_NAME") or _spans(text, "ORGANIZATION")
        assert any(o == "Bluth Company" for o in orgs), orgs
        assert all("Contact" not in o for o in orgs), orgs

    def test_org_does_not_cross_newline(self) -> None:
        text = "Employer: Nordic Analytics\nEIN: 83-9386718"
        orgs = _spans(text, "ORGANIZATION") + _spans(text, "ORGANIZATION_NAME")
        assert any(o == "Nordic Analytics" for o in orgs), orgs
        assert all("EIN" not in o for o in orgs), orgs

    def test_noun_keyword_without_colon_does_not_capture_next_label(self) -> None:
        text = "Length of employment: 9 years\nEmployer Tax ID: 85-5503355"
        orgs = _spans(text, "ORGANIZATION") + _spans(text, "ORGANIZATION_NAME")
        assert all("Tax" not in o for o in orgs), orgs

    def test_industry_suffix_does_not_cross_blank_line(self) -> None:
        text = "John Lee — Project Manager at Atlantic Data Systems\n\nSoftware engineer with experience"
        orgs = _spans(text, "ORGANIZATION") + _spans(text, "ORGANIZATION_NAME")
        assert any(o == "Atlantic Data Systems" for o in orgs), orgs
        assert all("Software" not in o for o in orgs), orgs

    def test_camelcase_org_detected(self) -> None:
        text = "Hello, I'm Daniel Moore from InnovateLabs. My email is d@example.com."
        orgs = _spans(text, "ORGANIZATION") + _spans(text, "ORGANIZATION_NAME")
        assert "InnovateLabs" in orgs, orgs

    def test_camelcase_mc_surnames_not_org(self) -> None:
        orgs = _spans("Please call McDonald about the case.", "ORGANIZATION") + _spans(
            "Please call McDonald about the case.", "ORGANIZATION_NAME"
        )
        assert "McDonald" not in orgs, orgs
