"""Tests for the redesigned fail-closed PII-egress gate (structured-token +
detector-surface containment; no blanket character k-grams)."""

from __future__ import annotations

import pytest

from pii_anon.assurance.pii_egress import PiiEgressError, assert_safe, scan_output

_CORPUS = [
    ("r1", "contact alice@acme-corp.example for details"),
    ("r2", "ssn 123-45-6789 on file"),
    ("r3", "call 555-867-5309 today"),
]


def test_clean_output_no_findings() -> None:
    out = "Detection F1 0.91; 200 records; EMAIL recall 0.98."
    assert scan_output(out, _CORPUS) == []


def test_structured_email_caught() -> None:
    findings = scan_output("leak: alice@acme-corp.example here", _CORPUS)
    assert any(f.kind == "containment" for f in findings)


def test_structured_ssn_and_phone_caught() -> None:
    assert any(f.kind == "containment" for f in scan_output("value 123-45-6789 here", _CORPUS))
    assert any(f.kind == "containment" for f in scan_output("dial 555-867-5309 now", _CORPUS))


def test_finding_does_not_leak_pii() -> None:
    findings = scan_output("alice@acme-corp.example and 123-45-6789", _CORPUS)
    assert findings
    for f in findings:
        assert "alice@" not in f.detail and "123-45" not in f.detail


def test_common_english_no_false_positive() -> None:
    # The report's boilerplate shares long common-English runs with a NON-PII dataset.
    # The old blanket k-gram scan false-positived here; the redesign must not.
    corpus = [("r1", "please contact the support team for more information about the details")]
    out = ("Methodology: residual leakage is a lower bound. Please contact the support "
           "team for more information about the details and limitations.")
    assert scan_output(out, corpus) == []


def test_unstructured_name_caught_via_detector() -> None:
    def detector(text: str):
        idx = text.find("Bobby Tables")
        return [("PERSON", idx, idx + len("Bobby Tables"))] if idx >= 0 else []

    findings = scan_output("leaked Bobby Tables here", [("r1", "Bobby Tables enrolled")], detector=detector)
    assert any(f.kind == "detector" for f in findings)


def test_detector_false_positive_on_report_metadata_ignored() -> None:
    # detector flags the report's own date, which is NOT in the corpus -> not a leak.
    def detector(text: str):
        return [("DATE_TIME", 0, 10)]

    findings = scan_output("2026-06-17 report", [("r1", "unrelated content here")], detector=detector)
    assert not any(f.kind == "detector" for f in findings)


def test_detector_error_fails_closed() -> None:
    def detector(text: str):
        raise RuntimeError("kaboom")

    findings = scan_output("clean text", [("r1", "unrelated")], detector=detector)
    assert any(f.kind == "detector_error" for f in findings)


def test_assert_safe_raises_and_message_pii_free() -> None:
    with pytest.raises(PiiEgressError) as ei:
        assert_safe("alice@acme-corp.example", _CORPUS, artifact="report.json")
    assert "alice@" not in str(ei.value)
    assert "report.json" in str(ei.value)


def test_assert_safe_passes_on_clean() -> None:
    assert_safe("F1 0.9, 100 records", _CORPUS)
