"""Tests for the version-token exemption in scrub_label (presentation-only).

A pipeline/dataset version like "2.3.1" has >=3 digits and len>=5, so the structured-token label
scrubber used to redact it to "[redacted]" — safe, but it made every report's provenance show a
redacted version. The exemption keeps a tight DOTTED-numeric version readable while still
redacting every PII shape. The egress gate (scan_output) is unchanged and remains the real guard;
this only affects how a free-form LABEL is rendered.

Round-20 close hardening: a 4-digit-LEADING dotted token (YYYY.MM[.DD]) is ambiguous with a
year-first date / birthdate and is NOT exempted. Classic semver (1-3-digit leading) stays
readable; day/month-first dotted dates are caught by the detector pass in scrub_label.
"""

from __future__ import annotations

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.adapters import pii_anon_adapter
from pii_anon.assurance.pii_egress import scan_output, scrub_label


@pytest.mark.parametrize("version", [
    "2.3.1", "1.5.0rc1", "10.4.2", "2.3", "v2.3.1", "1.0.0-beta2",
    "3.11.4", "0.1.0", "v24.04", "999.1.2",
])
def test_real_semver_preserved(version: str) -> None:
    assert scrub_label(version) == version


@pytest.mark.parametrize("pii", [
    "123456789",            # bare 9-digit id (no dots)
    "123-45-6789",          # SSN (hyphens, not dots)
    "555-867-5309",         # phone (hyphens)
    "ACCT-96865468",        # account label
    "96865468",             # bare 8-digit account
    "4111 1111 1111 1111",  # card (spaces)
    "12.345.678.901",       # 12 digits chunked across dots -> over the 8-digit cap
    "123.456.789",          # 9 digits chunked -> over the cap
    "1.2.3.4.5.6",          # 6 segments -> not a version shape
])
def test_pii_shapes_still_redacted(pii: str) -> None:
    assert "[redacted]" in scrub_label(pii)


@pytest.mark.parametrize("date_label", [
    "1990.05.21", "2024.01.02", "2001.09.11", "1985.12.31", "2023.12.25",
    "2024.10.15",  # full-date CalVer collides with a date -> redacted
    "2024.1",      # year.month -> date-ambiguous -> redacted
    "v2024.1",     # a 'v' does not rescue a year-leading form (digit-run extractor strips it)
])
def test_year_first_dates_redacted(date_label: str) -> None:
    # the round-20 upheld leak: a year-first dotted date must NOT survive as a "version"
    assert "[redacted]" in scrub_label(date_label)


def test_day_month_first_dates_caught_by_detector() -> None:
    # the runner always passes the detector; day/month-first dotted dates are caught there.
    pii = pii_anon_adapter()
    for d in ("21.05.1990", "05.21.1990", "90.05.21", "21.05.90", "31.12.1985"):
        assert "[redacted]" in scrub_label(d, pii.detect), d


def test_decorated_2digit_year_dates_caught_by_detector_deferral() -> None:
    # round-21 leak: a v-prefix / pre-release suffix BLINDS the detector on the raw token, so the
    # exemption strips the decoration to the bare core and re-consults the detector.
    pii = pii_anon_adapter()
    for d in ("v90.05.21", "V90.05.21", "90.05.21rc1", "90.5.21beta", "v99.12.31", "99.12.31rc1"):
        assert "[redacted]" in scrub_label(d, pii.detect), d


def test_detector_deferral_does_not_over_redact_decorated_semver() -> None:
    # the deferral must keep real decorated semver readable (the bare core is detector-clean).
    pii = pii_anon_adapter()
    for v in ("1.5.0rc1", "1.0.0-beta2", "v24.04", "2.3.1", "10.4.2", "v2.3.1"):
        assert scrub_label(v, pii.detect) == v, v


def test_version_embedded_in_label_only_version_kept() -> None:
    out = scrub_label("acme-dlp 2.3.1 acct 96865468")
    assert "2.3.1" in out
    assert "96865468" not in out


def test_exemption_does_not_touch_egress_gate() -> None:
    # the egress gate must STILL flag a real structured token regardless of the label exemption.
    corpus = [("r1", "build 2.3.1 shipped; ssn 123-45-6789")]
    findings = scan_output("leak: 123-45-6789 here", corpus)
    assert any(f.kind == "containment" for f in findings)


def test_year_first_date_label_does_not_reach_artifact(tmp_path) -> None:
    # end-to-end: a year-first-date dataset name / pipeline version must not leak into report.json.
    rows = [{"id": f"r{i}", "text": f"note {i}", "labels": []} for i in range(60)]
    pipe = PipelineAdapter(name="p", detect=lambda t: [], transform=lambda t: t,
                           version="1990.05.21", deterministic=True)
    run_assurance_report(AssuranceConfig(
        records=rows, pipeline=pipe, dataset_name="2001.09.11",
        dimensions=("leakage",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=200))
    blob = (tmp_path / "report.json").read_text()
    assert "1990.05.21" not in blob
    assert "2001.09.11" not in blob
