"""Tests for pipeline adapters — capabilities, PII-scrubbed errors, redaction."""

from __future__ import annotations

from pii_anon.assurance.adapters import (
    PipelineAdapter,
    pii_anon_adapter,
    redact_spans,
)


def _good_detect(text: str):
    return [("EMAIL", 0, 5)] if len(text) >= 5 else []


def test_capabilities() -> None:
    a = PipelineAdapter(name="x", detect=_good_detect)
    assert a.capabilities().can_detect is True
    assert a.capabilities().can_transform is False


def test_safe_detect_normalizes_and_filters() -> None:
    a = PipelineAdapter(name="x", detect=lambda t: [("EMAIL", 0, 5), ("BAD", 99, 1), ("X", -1, 2)])
    spans, err = a.safe_detect("hello world")
    assert err is None
    assert spans == [("EMAIL", 0, 5)]  # malformed/out-of-range dropped


def test_safe_detect_scrubs_exception_message() -> None:
    pii = "John Smith SSN 123-45-6789"

    def boom(text: str):
        raise ValueError(f"crashed on {text}")

    a = PipelineAdapter(name="x", detect=boom)
    spans, err = a.safe_detect(pii)
    assert spans == []
    assert err == "detect raised ValueError"
    assert "John Smith" not in err and "123-45-6789" not in err


def test_safe_transform_failure_is_none_not_zero() -> None:
    def boom(text: str) -> str:
        raise RuntimeError("boom " + text)

    a = PipelineAdapter(name="x", transform=boom)
    out, err = a.safe_transform("secret@x.com")
    assert out is None  # fail-closed: leakage assessor reads NOT_ASSESSABLE, never 0
    assert err == "transform raised RuntimeError"
    assert "secret@x.com" not in err


def test_safe_transform_nonstr_rejected() -> None:
    a = PipelineAdapter(name="x", transform=lambda t: 123)  # type: ignore[arg-type,return-value]
    out, err = a.safe_transform("hi")
    assert out is None
    assert "not str" in err


def test_safe_transform_missing() -> None:
    a = PipelineAdapter(name="x")
    out, err = a.safe_transform("hi")
    assert out is None
    assert "no transform" in err


def test_redact_spans_removes_pii_offset_safe() -> None:
    text = "Email alice@example.com and bob@example.com today"
    spans = [("EMAIL", 6, 23), ("EMAIL", 28, 43)]
    out = redact_spans(text, spans)
    assert "alice@example.com" not in out
    assert "bob@example.com" not in out
    assert out.count("[REDACTED]") == 2


def test_pii_anon_adapter_detects_and_redacts_email() -> None:
    adapter = pii_anon_adapter()
    assert adapter.capabilities().can_detect and adapter.capabilities().can_transform
    text = "Please contact alice@example.com regarding the account."
    spans, err = adapter.safe_detect(text)
    assert err is None
    assert any(s[0].upper().startswith("EMAIL") for s in spans), spans
    out, terr = adapter.safe_transform(text)
    assert terr is None and out is not None
    assert "alice@example.com" not in out
