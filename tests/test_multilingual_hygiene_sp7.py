"""sp7 panel (multilingual lens) — two measured home-benchmark hygiene fixes.

1. NATIONAL_ID duplicate on a US_SSN span: "Tax ID: 573-33-7773" emitted BOTH
   US_SSN and NATIONAL_ID on the same span — under strict one-to-one scoring
   the second is a pure FP. The multilingual home docs (zh/ko/hi/ar) use the
   English "Tax ID:" scaffold over SSN-format values, so this dominated their
   measured F2 deficit (panel: zh 0.906->0.931, ko 0.941->0.966, hi
   0.943->0.968, ar 0.940->0.965). Leak-SAFE dedup: the span stays masked as
   US_SSN (the _drop_dob_shadowed_dates class) — runs on ALL paths.

2. Honorific-as-ORGANIZATION: the _ORGANIZATION_CONTEXT verb triggers
   ("belongs to"/"works for") greedily captured the honorific ("belongs to
   Mr. Wu" -> ORGANIZATION 'Mr'). A bare honorific is never an organization;
   vetoing it at the capture head is spec-justified boundary hygiene.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_PROD = RegexEngineAdapter(enabled=True)


def _by_type(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in _PROD.detect({"text": text}, {"language": "en"}):
        if f.span_start is not None:
            out.setdefault(str(f.entity_type), []).append(text[f.span_start:f.span_end])
    return out


class TestNationalIdSsnDedup:
    def test_ssn_format_tax_id_single_finding(self) -> None:
        got = _by_type("Tax ID: 573-33-7773 registered.")
        assert got.get("US_SSN") == ["573-33-7773"]
        # the duplicate NATIONAL_ID on the same span is dropped (span stays
        # masked as US_SSN — leak-safe dedup).
        assert "573-33-7773" not in got.get("NATIONAL_ID", [])

    def test_non_ssn_national_id_survives(self) -> None:
        # a national id NOT claimed by US_SSN keeps its own finding.
        got = _by_type("National ID: X4419283K on file.")
        assert any("X4419283K" in v for v in got.get("NATIONAL_ID", [])), got

    def test_coverage_preserved(self) -> None:
        # the deduped span is still fully covered (masked) by US_SSN.
        text = "Tax ID: 573-33-7773 registered."
        cov: set[int] = set()
        for f in _PROD.detect({"text": text}, {"language": "en"}):
            if f.span_start is not None:
                cov.update(range(f.span_start, f.span_end))
        start = text.index("573-33-7773")
        assert set(range(start, start + len("573-33-7773"))) <= cov


class TestHonorificNotOrganization:
    @pytest.mark.parametrize(
        "text",
        [
            "Email hana.wu@ex.com belongs to Mr. Wu and the team.",
            "The device works for Dr. Chen exclusively.",
            "This account belongs to Mrs. Tanaka now.",
        ],
    )
    def test_bare_honorific_not_org(self, text: str) -> None:
        orgs = _by_type(text).get("ORGANIZATION", [])
        assert not any(o.rstrip(".") in ("Mr", "Mrs", "Ms", "Dr", "Prof") for o in orgs), orgs

    def test_real_org_after_verb_trigger_survives(self) -> None:
        got = _by_type("She works for Initech Solutions downtown.")
        assert any("Initech" in o for o in got.get("ORGANIZATION", [])), got
