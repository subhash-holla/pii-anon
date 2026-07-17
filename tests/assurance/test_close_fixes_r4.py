"""Regression tests for the upheld findings from the round-4 adversarial close."""

from __future__ import annotations


from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.assessors.leakage import _is_distinctive, _surface_leaked
from pii_anon.assurance.pii_egress import scan_output
from pii_anon.assurance.spans import canonical_type
from pii_anon.assurance.stats import Counts, cohens_d_paired, paired_bootstrap, pooled_f1


def _records(n=40):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now please."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


# --- #1 name-shaped entity_type is bucketed, never reaches an artifact ------------
def test_canonical_type_buckets_unknown() -> None:
    assert canonical_type("EMAIL_ADDRESS") == "EMAIL_ADDRESS"  # canonical
    assert canonical_type("email") == "email"                  # alias
    bucketed = canonical_type("Hernandez_Maiden_Name")
    assert bucketed.startswith("other:") and "Hernandez" not in bucketed


def test_name_shaped_type_not_in_any_artifact(tmp_path) -> None:
    token = "Hernandez_Maiden_Name"
    rows = [{"id": f"r{i}", "text": f"Record {i} content here padded longer text ok",
             "labels": [[token, 0, 6]]} for i in range(40)]
    run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="u", detect=lambda t: [(token, 0, 6)],
                                               transform=lambda t: t, deterministic=True),
        dimensions=("detection",), outputs=("json", "markdown"), out_dir=str(tmp_path),
        seed=1, n_resamples=20))
    for f in ("report.json", "report.md", "repro-bundle/synthetic-mirror.jsonl"):
        content = (tmp_path / f).read_text(encoding="utf-8")
        assert "Hernandez" not in content and "hernandez" not in content


# --- #2 Cohen's d maximal gap -> None (never a misleading 0.0) --------------------
def test_cohens_d_maximal_is_none() -> None:
    assert cohens_d_paired([-1.0] * 10) is None  # constant non-zero -> maximal effect
    assert cohens_d_paired([0.0] * 10) == 0.0    # constant zero -> genuinely no effect


def test_paired_bootstrap_maximal_effect_none() -> None:
    a = [Counts(tp=0, fp=0, fn=1) for _ in range(50)]
    b = [Counts(tp=1, fp=0, fn=0) for _ in range(50)]
    res = paired_bootstrap(a, b, pooled_f1, seed=1, n_resamples=200)
    assert res.delta == -1.0
    assert res.effect_size is None  # not 0.0


def test_markdown_renders_maximal_not_zero(tmp_path) -> None:
    weak = PipelineAdapter(name="weak", detect=lambda t: [], transform=lambda t: t, deterministic=True)
    run_assurance_report(AssuranceConfig(
        records=_records(60), pipeline=weak, dimensions=("detection",), outputs=("markdown",),
        out_dir=str(tmp_path), seed=1, n_resamples=120))
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "maximal" in md  # the maximal consistent gap is labelled, not shown as 0.0000


# --- #3 markdown injection is neutralized ----------------------------------------
def test_markdown_injection_neutralized(tmp_path) -> None:
    evil = "<script>alert(1)</script>"
    user = PipelineAdapter(name=evil, detect=lambda t: [], transform=lambda t: t, deterministic=True)
    run_assurance_report(AssuranceConfig(
        records=_records(20), pipeline=user, dimensions=("detection",), outputs=("markdown",),
        out_dir=str(tmp_path), seed=1, n_resamples=20))
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "<script>" not in md  # HTML-escaped


# --- minor: leakage specificity (no common-word / short-digit false leaks) -------
def test_common_word_not_distinctive() -> None:
    assert _is_distinctive("information", 4) is False
    assert _is_distinctive("John Smith", 4) is True
    assert _is_distinctive("alice@x.com", 4) is True


def test_short_digit_run_not_normalized_leak() -> None:
    from pii_anon.assurance.assessors.leakage import _norm_text
    out1 = "code 12 3 45 here"
    out2 = "x 123 45 6789 y"
    assert _surface_leaked("12345", out1, "12345", _norm_text(out1)) is False  # 5 digits < 7
    assert _surface_leaked("123-45-6789", out2, "123456789", _norm_text(out2)) is True  # 9 digits


# --- minor: egress finding never echoes a (possibly-PII) record_id ----------------
def test_egress_finding_hashes_record_id() -> None:
    rid = "patient_jane@secret.example"
    findings = scan_output("leaked alice@acme.example data", [(rid, "alice@acme.example record")])
    assert findings
    for f in findings:
        assert "patient_jane" not in f.detail and "secret.example" not in f.detail
