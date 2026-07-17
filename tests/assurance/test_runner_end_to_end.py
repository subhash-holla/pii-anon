"""End-to-end runner tests — the full pipeline incl. both gates and file output.

All data is SYNTHETIC (AX-001). The PII-egress assertions use synthetic tokens
that look like PII; the point is that they must never survive into an artifact.
"""

from __future__ import annotations

import json
import re

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.claim_strength import ClaimStrength

_EMAIL_RE = re.compile(r"[\w.]+@[\w.]+\.\w+")


def _user_detect(text: str):
    # a competent-but-imperfect detector: finds emails, misses every 5th
    hits = [(m.start(), m.end()) for m in _EMAIL_RE.finditer(text)]
    return [("EMAIL", s, e) for i, (s, e) in enumerate(hits) if i % 5 != 4]


def _user_transform(text: str) -> str:
    return _EMAIL_RE.sub("[EMAIL]", text)


def _synthetic_records(n: int = 60):
    rows = []
    for i in range(n):
        surface = f"user{i}@host.example.com"
        text = f"Record {i}: please contact {surface} for details."
        start = text.index(surface)
        rows.append({
            "id": f"r{i}",
            "text": text,
            "labels": [["EMAIL", start, start + len(surface)]],
        })
    return rows


def _config(tmp_path, **kw):
    defaults = dict(
        records=_synthetic_records(),
        dataset_name="synthetic-test",
        pipeline=PipelineAdapter(
            name="acme-dlp", detect=_user_detect, transform=_user_transform,
            version="1.0.0", deterministic=True,
        ),
        dimensions=("detection", "leakage"),
        outputs=("json", "markdown"),
        out_dir=str(tmp_path),
        seed=1,
        n_resamples=200,
    )
    defaults.update(kw)
    return AssuranceConfig(**defaults)


def test_end_to_end_writes_all_artifacts(tmp_path) -> None:
    run_assurance_report(_config(tmp_path))
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "repro-bundle" / "provenance.json").is_file()
    assert (tmp_path / "repro-bundle" / "synthetic-mirror.jsonl").is_file()
    assert (tmp_path / "repro-bundle" / "methodology.md").is_file()


def test_report_json_is_valid_and_structured(tmp_path) -> None:
    run_assurance_report(_config(tmp_path))
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert data["dataset"]["mode"] == "labeled"
    assert "detection" in data["dimensions"]
    assert "leakage" in data["dimensions"]
    assert set(data["systems"]) == {"acme-dlp", "pii-anon"}
    assert data["dataset"]["fingerprint"]


def test_uncertified_harness_demotes_to_advisory(tmp_path, monkeypatch) -> None:
    # When the harness is NOT certified, even a powered labeled number must be ADVISORY,
    # with the reason stated (the fail-closed default before certification).
    import pii_anon.assurance.provenance as prov_mod
    monkeypatch.setattr(prov_mod, "HARNESS_CLOSE_CERTIFIED", False)
    report = run_assurance_report(_config(tmp_path))
    pa = report.dimensions["detection"]["pii-anon"]
    assert pa.claim_strength is ClaimStrength.ADVISORY
    assert any("adversarial close" in r for r in pa.reasons)


def test_certified_harness_allows_measured(tmp_path) -> None:
    # With the harness CERTIFIED (the shipped default) a perfect, powered, labeled
    # detection number is MEASURED (publishable).
    rows = []
    for i in range(80):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    # a perfect user detector that exactly reproduces the gold spans
    import re as _re
    rx = _re.compile(r"\S+@\S+\.\S+")

    def perfect(text: str):
        return [("EMAIL_ADDRESS", m.start(), m.end()) for m in rx.finditer(text)]

    report = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="perfect", detect=perfect, transform=lambda t: t,
                                               version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=200))
    assert report.dimensions["detection"]["perfect"].claim_strength is ClaimStrength.MEASURED


def test_no_raw_pii_in_any_artifact(tmp_path) -> None:
    run_assurance_report(_config(tmp_path))
    for fname in ("report.json", "report.md", "repro-bundle/provenance.json"):
        content = (tmp_path / fname).read_text(encoding="utf-8")
        assert "user3@host.example.com" not in content
        assert "user42@host.example.com" not in content


def test_predictor_that_leaks_via_exception_is_contained(tmp_path) -> None:
    # A user pipeline that raises with the raw text in the message must not leak it.
    def evil_detect(text: str):
        raise ValueError(f"cannot parse: {text}")

    cfg = _config(tmp_path)
    cfg.pipeline = PipelineAdapter(name="evil", detect=evil_detect, transform=lambda t: "[X]")
    run_assurance_report(cfg)
    for fname in ("report.json", "report.md"):
        content = (tmp_path / fname).read_text(encoding="utf-8")
        assert "host.example.com" not in content


def test_html_and_summary_outputs_now_render(tmp_path) -> None:
    # Phase 2 implements the HTML executive + one-page summary renderers (was NotImplementedError).
    run_assurance_report(_config(tmp_path, outputs=("html", "summary")))
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "summary.html").is_file()
    assert (tmp_path / "report.html").read_text().startswith("<!DOCTYPE html>")


def test_run_is_deterministic(tmp_path) -> None:
    run_assurance_report(_config(tmp_path / "a"))
    run_assurance_report(_config(tmp_path / "b"))
    j1 = json.loads((tmp_path / "a" / "report.json").read_text())
    j2 = json.loads((tmp_path / "b" / "report.json").read_text())
    # drop the timestamp (the only wall-clock field)
    for j in (j1, j2):
        j["generated_at"] = None
        j["provenance"]["timestamp"] = None
    assert j1 == j2
