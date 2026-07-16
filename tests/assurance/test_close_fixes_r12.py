"""Regression tests for the upheld findings from the round-12 (certified-path) close."""

from __future__ import annotations


import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report

from pii_anon.assurance.runner import AssuranceRunError


# --- #1 unbounded / enormous predictor output cannot hang or OOM the harness ------
def test_infinite_generator_detect_is_bounded() -> None:
    def infinite(text):
        while True:
            yield ("X", 0, 1)

    adapter = PipelineAdapter(name="inf", detect=infinite)
    spans, err = adapter.safe_detect("hello world")  # must RETURN (not hang)
    assert spans == []
    assert err is not None and "capped" in err


def test_enormous_finite_detect_is_bounded() -> None:
    def huge(text):
        return (("X", 0, 1) for _ in range(5_000_000))

    spans, err = PipelineAdapter(name="huge", detect=huge).safe_detect("hi")
    assert spans == [] and err is not None and "capped" in err


def test_infinite_generator_end_to_end_completes(tmp_path) -> None:
    def infinite(text):
        while True:
            yield ("EMAIL_ADDRESS", 0, 1)

    rows = [{"id": f"r{i}", "text": f"rec {i} hello world here", "labels": [["EMAIL_ADDRESS", 0, 3]]}
            for i in range(20)]
    # the run COMPLETES (the unbounded detector degrades to an adapter error), never hangs
    run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="inf", detect=infinite, transform=lambda t: "[X]", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=40))
    assert (tmp_path / "report.json").is_file()


# --- #2 a lone-surrogate record: no crash, and no PII via the __context__ chain ----
def test_lone_surrogate_text_no_crash_no_context_leak(tmp_path) -> None:
    secret = "CTX_LEAK_PROOF_SECRET"
    rows = [{"id": f"r{i}", "text": f"{secret} \ud800 rec {i} content"} for i in range(25)]
    pipe = PipelineAdapter(name="a", detect=lambda t: [], transform=lambda t: t, version="1", deterministic=True)
    cfg = AssuranceConfig(records=rows, pipeline=pipe, dimensions=("detection",), outputs=("json",),
                          out_dir=str(tmp_path), seed=42, n_resamples=40)
    # surrogate-safe fingerprint -> the run SUCCEEDS (no crash)
    run_assurance_report(cfg)
    assert (tmp_path / "report.json").is_file()
    assert secret not in (tmp_path / "report.json").read_text()


def test_run_error_has_no_chained_context(tmp_path, monkeypatch) -> None:
    secret = "CTX_LEAK_PROOF_SECRET2"
    rows = [{"id": f"r{i}", "text": f"{secret} rec {i}"} for i in range(5)]
    import pii_anon.assurance.runner as runner_mod

    def boom(*a, **k):
        raise RuntimeError(f"carries {rows}")

    monkeypatch.setattr(runner_mod, "assess_detection", boom)
    cfg = AssuranceConfig(records=rows, pipeline=PipelineAdapter(name="u", detect=lambda t: [], transform=lambda t: t),
                          dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=5)
    with pytest.raises(AssuranceRunError) as ei:
        run_assurance_report(cfg)
    # the chained (possibly PII-bearing) exception is fully DETACHED, not just suppressed
    assert ei.value.__context__ is None
    assert ei.value.__cause__ is None
    assert secret not in str(ei.value)
