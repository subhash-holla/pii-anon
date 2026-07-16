"""Regression tests for the round-18 sample-report finding: the detector-surface
containment layer (Layer 2) false-positived on bare short numeric surfaces.

A realistic dataset of "...invoice 1000..." records made the detector flag the 4-digit
invoice numbers as PII; the bare-substring containment test then matched those 4 digits
coincidentally inside sha256 digests / numeric metrics in report.json, failing the egress
gate on a report that carried NO raw text. The fix adds a distinctiveness floor + a
token-boundary containment test to Layer 2 (both the raw side and the output side), bringing
it into line with Layer 1 (which already excludes <5-char / pure-4-digit tokens). Real-leak
recall (emails, full names, long IDs) is preserved.
"""

from __future__ import annotations

import re

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.adapters import pii_anon_adapter
from pii_anon.assurance.pii_egress import (
    _is_distinctive_surface,
    _token_contained,
    scan_output,
)


# --- #1 the realistic sample that triggered the block now completes -----------------
def test_realistic_invoice_dataset_does_not_false_positive(tmp_path) -> None:
    rows = []
    for i in range(120):
        txt = f"Ticket {i}: please contact user{i}@acme.example.com about invoice {1000 + i}."
        pre = f"Ticket {i}: please contact "
        rows.append({
            "id": f"rec-{i}", "text": txt,
            "labels": [["EMAIL", len(pre), len(pre) + len(f"user{i}@acme.example.com")]],
        })

    def acme_detect(t: str):
        return [("EMAIL", m.start(), m.end()) for m in re.finditer(r"\S+@\S+", t)]

    def acme_transform(t: str) -> str:
        return re.sub(r"\S+@\S+", "[EMAIL]", t)

    pipe = PipelineAdapter(name="acme-dlp", detect=acme_detect, transform=acme_transform,
                           version="2.3", deterministic=True)
    # must NOT raise PiiEgressError on a report that carries only hashes / metrics / prose
    run_assurance_report(AssuranceConfig(
        records=rows, pipeline=pipe, dimensions=("detection", "leakage"),
        outputs=("json", "markdown"), out_dir=str(tmp_path), seed=7, n_resamples=300))
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "repro-bundle" / "synthetic-mirror.jsonl").is_file()


# --- #2 a bare short-numeric detector surface is NOT a containment finding ----------
def test_bare_four_digit_surface_not_flagged_raw_side() -> None:
    def detector(text: str):
        # flags the 4-digit invoice number, as a real detector might
        m = re.search(r"\b\d{4}\b", text)
        return [("ID", m.start(), m.end())] if m else []

    # the 4 digits appear coincidentally inside a hash-like string in the artifact
    artifact = '{"fingerprint": "9c1000fa2b", "n": 1000}'
    findings = scan_output(artifact, [("r1", "invoice 1000 paid")], detector=detector)
    assert not any(f.kind == "detector" for f in findings)


# --- #3 a real distinctive leak (full name) STILL fires (raw + output side) ---------
def test_real_name_leak_still_caught() -> None:
    pii = pii_anon_adapter()
    artifact = '{"raw_note": "the customer is Margaret Thompson from Boston"}'
    findings = scan_output(artifact, [("r1", "Contact Margaret Thompson re: claim.")],
                           detector=pii.detect)
    assert any(f.kind == "detector" for f in findings)


# --- #4 token-boundary: an embedded substring is not a match, a delimited one is ----
def test_token_boundary_excludes_embedded_includes_delimited() -> None:
    assert _token_contained("ref id 1234567.", "1234567") is True
    assert _token_contained("hash a1234567ff", "1234567") is False  # embedded in alnum run
    assert _token_contained('{"x": "John Smith"}', "John Smith") is True


# --- #5 the distinctiveness floor: shape, not length alone --------------------------
def test_distinctiveness_floor() -> None:
    for non_distinctive in ("1000", "1119", "42", "2026", "abc", "code1"):
        assert _is_distinctive_surface(non_distinctive) is False
    for distinctive in ("a@b.com", "John Smith", "123456789", "Bjarnason", "Chen-Williams!"):
        assert _is_distinctive_surface(distinctive) is True
