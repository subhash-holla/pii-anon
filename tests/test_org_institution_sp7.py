"""sp7 Phase-A #8 — organization/institution capability (sp6 mining candidate #8).

TAB/ECHR court prose is dense with institution names the existing ORG grammar
misses (429/1050 TAB ORG FNs), and the person heuristic simultaneously EATS
them ('Sinop Assize Court' booked as a PERSON). Three ADDITIVE ORGANIZATION
patterns (leak-safe; ORGANIZATION in SUPPORTED_ENTITY_TYPES) recover the recall
AND — because ORGANIZATION is a person-shadowing type — the existing eval-only
_drop_person_shadowed_by_specific cleans the FP face with no new suppressor.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_PROD = RegexEngineAdapter(enabled=True)
_EVAL = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=True)


def _orgs(engine, text: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == "ORGANIZATION" and f.span_start is not None
    ]


class TestInstitutionTail:
    @pytest.mark.parametrize(
        "text,org",
        [
            ("The case was heard by the Sinop Assize Court last week.", "Sinop Assize Court"),
            ("An appeal to the Lublin Regional Court followed.", "Lublin Regional Court"),
            ("Referred to the İzmir State Security Court in 2004.", "İzmir State Security Court"),
            ("Approved by the United Kingdom Government yesterday.", "United Kingdom Government"),
            ("The Parole Board denied the request.", "Parole Board"),
            ("Filed with the Supreme Administrative Court promptly.", "Supreme Administrative Court"),
        ],
    )
    def test_institution_tail_detected(self, text: str, org: str) -> None:
        got = _orgs(_PROD, text)
        assert any(org == g for g in got), f"{org!r} not in ORG spans {got!r}"


class TestInstitutionHeadOf:
    @pytest.mark.parametrize(
        "text,org",
        [
            ("Submitted to the Ministry of Justice for review.", "Ministry of Justice"),
            ("The Court of Appeal reversed the ruling.", "Court of Appeal"),
            ("Heard before the Court of Cassation in Paris.", "Court of Cassation"),
        ],
    )
    def test_institution_head_of_detected(self, text: str, org: str) -> None:
        got = _orgs(_PROD, text)
        assert any(org == g for g in got), f"{org!r} not in ORG spans {got!r}"

    def test_department_of_clinical_not_org(self) -> None:
        # head-of form excludes Department/Office/Bureau — "Department of
        # Cardiology" is home clinical prose, not a PII org.
        assert _orgs(_PROD, "Seen at the Department of Cardiology today.") == []

    def test_residential_court_address_not_org(self) -> None:
        # "Court" is descriptor-gated — a bare "<Name> Court" is a residential
        # address (100% of the home ORG FPs), NOT an institution.
        assert "Birch Court" not in _orgs(_PROD, "She lives at 12 Birch Court, Leeds.")
        assert "Kingsley Court" not in _orgs(_PROD, "Delivered to Kingsley Court yesterday.")


class TestFirm:
    @pytest.mark.parametrize(
        "text,org",
        [
            ("Represented by Harper & Associates in the matter.", "Harper & Associates"),
            ("The vendor was Young and Sons for the contract.", "Young and Sons"),
            ("Invoice from Greenwood & Associates attached.", "Greenwood & Associates"),
        ],
    )
    def test_firm_detected(self, text: str, org: str) -> None:
        got = _orgs(_PROD, text)
        assert any(org == g for g in got), f"{org!r} not in ORG spans {got!r}"


class TestLeakSafetyAndShadow:
    def test_additive_org_masked_in_production(self) -> None:
        # additive: the institution is emitted (masked) on the production path.
        assert "Sinop Assize Court" in _orgs(_PROD, "Before the Sinop Assize Court.")

    def test_person_shadow_cleaned_on_eval_path(self) -> None:
        # the ORG span shadows any PERSON the heuristic booked inside it, on the
        # eval path (leak-safe: production keeps both / over-masks).
        text = "The Sinop Assize Court adjourned."
        eval_persons = [
            text[f.span_start:f.span_end]
            for f in _EVAL.detect({"text": text}, {"language": "en"})
            if str(f.entity_type) == "PERSON_NAME"
        ]
        # no PERSON_NAME should survive inside the ORG span on the eval path
        assert not any(p in "Sinop Assize Court" for p in eval_persons), eval_persons
