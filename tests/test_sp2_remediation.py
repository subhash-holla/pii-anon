"""sp2 adversarial-review remediation (the 9-round close findings).

Each test pins a fix for a confirmed finding from the sp2 adversarial review:
the SHOWSTOPPER production PII-leak, the 'Record shows' eval-integrity MAJOR,
and the over-capturing labeled patterns (HEALTH_CONDITION / EDUCATION_LEVEL /
ORGANIZATION CamelCase / MEDICATION) + the ISO-8601 timezone-extent bug.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter


def _spans(text: str, entity_type: str, *, eval_arbitration: bool = False) -> list[str]:
    engine = RegexEngineAdapter(enabled=True, eval_cross_type_arbitration=eval_arbitration)
    return sorted(
        text[f.span_start : f.span_end]
        for f in engine.detect({"text": text}, {"language": "en"})
        if f.entity_type == entity_type
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


# ── SHOWSTOPPER: production must never drop a maskable PERSON_NAME in favour ──
# of a non-masked specific type (that drop leaks the name through anonymization)

class TestLeakSafeProductionArbitration:
    @pytest.mark.parametrize(
        "text",
        [
            "Position: Mary Johnson",
            "Job: Karen Mitchell",
            "Occupation: David Chen reported to HR",
            "Vehicle: Laura Martinez",
            "Procedure: Daniel Foster",
        ],
    )
    def test_person_survives_in_default_production_mode(self, text: str) -> None:
        # Default adapter (production / masking path): the PERSON_NAME finding
        # MUST survive even when a non-masked specific type covers the same
        # span — over-masking is the safe direction; dropping leaks the name.
        names = _spans(text, "PERSON_NAME", eval_arbitration=False)
        assert any(name in text for name in names) and names, (
            f"PERSON_NAME dropped in production mode for {text!r}: {names}"
        )

    def test_eval_mode_still_arbitrates_real_job_title(self) -> None:
        # With the eval flag, a PERSON FP on a REAL job title is removed
        # (benchmark precision) — the legitimate use of the arbitration.
        text = "Position: Medical Director | Annual income: $133,000"
        assert "Medical Director" in _spans(text, "JOB_TITLE", eval_arbitration=True)
        assert "Medical Director" not in _spans(
            text, "PERSON_NAME", eval_arbitration=True
        )


# ── MAJOR: eval-integrity — drop the generator's 'Record shows' filler anchor ─

class TestRecordShowsAnchorRemoved:
    @pytest.mark.parametrize(
        ("text", "etype"),
        [
            ("Works for Acme Corp. Record shows American.", "NATIONALITY"),
            ("Record shows AB1234567. Record shows European.", "ETHNICITY"),
            ("Record shows Male. Record shows Liberal.", "POLITICAL_OPINION"),
            ("Account active. Record shows Buddhist.", "RELIGIOUS_BELIEF"),
        ],
    )
    def test_record_shows_filler_no_longer_detects(self, text: str, etype: str) -> None:
        assert _spans(text, etype) == [], _spans(text, etype)

    @pytest.mark.parametrize(
        ("text", "etype", "expected"),
        [
            ("Nationality: American", "NATIONALITY", "American"),
            ("Ethnicity: European", "ETHNICITY", "European"),
            ("Political Opinion: Liberal", "POLITICAL_OPINION", "Liberal"),
            ("Religious Belief: Buddhist", "RELIGIOUS_BELIEF", "Buddhist"),
            ("Gender: Male. DOB: 1984-05-09.", "GENDER", "Male"),
            ("Marital Status: Married.", "MARITAL_STATUS", "Married"),
        ],
    )
    def test_legitimate_field_label_still_detects(
        self, text: str, etype: str, expected: str
    ) -> None:
        assert expected in _spans(text, etype), _spans(text, etype)


# ── MAJOR: ISO-8601 timezone designator must survive a trailing sentence period ─

class TestIso8601TimezoneExtent:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("System access logged at 2021-10-23T09:53:00Z.", "2021-10-23T09:53:00Z"),
            ("Last login 2021-01-06T08:34:00+02:00.", "2021-01-06T08:34:00+02:00"),
            ("Event 2024-08-26T09:01:00Z occurred.", "2024-08-26T09:01:00Z"),
        ],
    )
    def test_timezone_kept_before_period(self, text: str, expected: str) -> None:
        assert expected in _spans(text, "DATE_TIME"), _spans(text, "DATE_TIME")

    def test_ip_fragment_still_not_a_datetime(self) -> None:
        assert _spans("logged in from 208.74.38.190.", "DATE_TIME") == []


# ── MAJOR: HEALTH_CONDITION lead-ins must not capture person names ────────────

class TestHealthConditionNoNames:
    @pytest.mark.parametrize(
        "text",
        [
            "Evaluation of Mark Thompson was completed on site.",
            "Consultation regarding Sarah Johnson is scheduled.",
            "Treatment for Maria Garcia begins Monday.",
            "History of Robert Lee was reviewed.",
        ],
    )
    def test_lead_in_does_not_capture_name(self, text: str) -> None:
        assert _spans(text, "HEALTH_CONDITION") == [], _spans(text, "HEALTH_CONDITION")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("presents with symptoms consistent with Acute Bronchitis.", "Acute Bronchitis"),
            ("MRN: MRN-5929414 Diagnosis: Chronic Kidney Disease", "Chronic Kidney Disease"),
            ("seen for evaluation of Type 2 Diabetes Mellitus.", "Type 2 Diabetes Mellitus"),
            ("Diagnosis: Osteoarthritis", "Osteoarthritis"),
        ],
    )
    def test_real_condition_still_detected(self, text: str, expected: str) -> None:
        assert expected in _spans(text, "HEALTH_CONDITION"), _spans(text, "HEALTH_CONDITION")


# ── MAJOR: MEDICATION must require a dose or form word (no bare names) ────────

class TestMedicationNoNames:
    def test_still_taking_name_not_a_medication(self) -> None:
        assert _spans("The patient is still taking Robert Williams", "MEDICATION_NAME") == []

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Care completed. Gabapentin 300mg as needed.", "Gabapentin 300mg"),
            ("Current medications include Albuterol Inhaler.", "Albuterol Inhaler"),
            ("Current Medications: Metformin 500mg", "Metformin 500mg"),
        ],
    )
    def test_real_medication_still_detected(self, text: str, expected: str) -> None:
        assert expected in _spans(text, "MEDICATION_NAME"), _spans(text, "MEDICATION_NAME")


# ── MAJOR: EDUCATION_LEVEL bare 'Master'/'Bachelor'/'Associate' FP ───────────

class TestEducationLevelBareWords:
    @pytest.mark.parametrize(
        "text",
        [
            "Please review the Master Service Agreement attached.",
            "She was promoted to Senior Associate last year.",
            "The Bachelor aired its finale on Monday.",
        ],
    )
    def test_bare_degree_word_not_education(self, text: str) -> None:
        assert _spans(text, "EDUCATION_LEVEL") == [], _spans(text, "EDUCATION_LEVEL")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("DOB: 1984-05-09. Education: PhD. Job: Welder.", "PhD"),
            ("Record: Bachelor's Degree in Computer Science.", "Bachelor's Degree in Computer Science"),
            ("She holds a Master's degree.", "Master's"),
            ("Education: MBA", "MBA"),
        ],
    )
    def test_real_education_still_detected(self, text: str, expected: str) -> None:
        assert expected in _spans(text, "EDUCATION_LEVEL"), _spans(text, "EDUCATION_LEVEL")


# ── MAJOR: ORGANIZATION CamelCase must be context-anchored ───────────────────

class TestOrganizationCamelCase:
    @pytest.mark.parametrize(
        "text",
        [
            "Connect to the WiFi network in the lobby.",
            "The JavaScript bundle uses PowerPoint exports via GitHub.",
            "DeAndre Williams and LaToya Brown signed the lease.",
            "Witness DiCaprio testified at the hearing.",
        ],
    )
    def test_bare_camelcase_token_not_org(self, text: str) -> None:
        assert _spans(text, "ORGANIZATION") == [], _spans(text, "ORGANIZATION")

    def test_context_anchored_camelcase_org_detected(self) -> None:
        text = "Hello, I'm Daniel Moore from InnovateLabs. My email is d@example.com."
        assert "InnovateLabs" in _spans(text, "ORGANIZATION"), _spans(text, "ORGANIZATION")


# ── MINOR: _ZW must be the three zero-width codepoints (trojan-source guard) ──

def test_zw_constant_is_three_zero_width_codepoints() -> None:
    from pii_anon.engines.regex.patterns import _ZW

    assert [ord(c) for c in _ZW] == [0x200B, 0x200C, 0x200D]
