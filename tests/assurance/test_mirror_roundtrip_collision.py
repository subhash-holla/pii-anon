"""Round-trip on the shipped synthetic mirror must not false-positive the egress gate.

The reproducibility claim ("a synthetic mirror is shipped so the methodology is
reproducible end-to-end on non-real data") is only true if feeding the shipped
``repro-bundle/synthetic-mirror.jsonl`` back through the runner succeeds. A
regenerated mirror-of-a-mirror drew its synthetic email tokens from a tiny reserved
namespace (``user{idx}{0-999}@example.com``); two independent draws collided
~1/1000 per email (~6% over 60 records), and the egress containment scan then
flagged the colliding synthetic token verbatim. Salting the RNG with the dataset
fingerprint only RANDOMIZES that collision — seed=42 over this 60-record corpus
deterministically reproduces it. The fix is to not regenerate a mirror when the
input is already a mirror (the input itself is the non-real reproduction corpus).
"""

from __future__ import annotations

import json
import re

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.dataset import load_records
from pii_anon.assurance.synthetic_mirror import is_synthetic_mirror, synthesize_mirror

_EMAIL = re.compile(r"\S+@\S+\.\S+")


def _email_records(n=60):
    rows = []
    for i in range(n):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL_ADDRESS", st, st + len(s)]]})
    return rows


def _email_pipeline(name="acme"):
    return PipelineAdapter(
        name=name,
        detect=lambda t: [("EMAIL_ADDRESS", m.start(), m.end()) for m in _EMAIL.finditer(t)],
        transform=lambda t: _EMAIL.sub("[EMAIL]", t), version="1", deterministic=True,
    )


def test_shipped_mirror_roundtrips_at_colliding_seed(tmp_path) -> None:
    """seed=42 deterministically collides on the pre-fix path; the round-trip through
    BOTH json and markdown must pass both gates (no AssuranceRunError)."""
    seed = 42
    out_a = tmp_path / "a"
    run_assurance_report(AssuranceConfig(
        records=_email_records(60), pipeline=_email_pipeline(), dimensions=("detection", "leakage"),
        outputs=("json", "markdown"), out_dir=str(out_a), seed=seed, n_resamples=60))
    mirror = out_a / "repro-bundle" / "synthetic-mirror.jsonl"
    mirror_rows = [json.loads(line) for line in mirror.read_text().splitlines() if line.strip()]
    assert mirror_rows  # the forward run shipped a non-empty mirror

    # feed the shipped mirror back through the runner — json + markdown — must not raise
    out_b = tmp_path / "b"
    rep = run_assurance_report(AssuranceConfig(
        records=mirror_rows, pipeline=_email_pipeline(), dimensions=("detection", "leakage"),
        outputs=("json", "markdown"), out_dir=str(out_b), seed=seed, n_resamples=60))
    assert (out_b / "report.json").is_file()
    assert (out_b / "report.md").is_file()
    assert rep is not None


def test_is_synthetic_mirror_classifies_correctly() -> None:
    """The skip signal must fire for a shipped mirror and NEVER for real data (a false
    positive would skip an artifact; a false negative is the safe default — regenerate)."""
    real = load_records(_email_records(5))
    assert is_synthetic_mirror(real) is False

    mirror_ds = load_records([
        {"id": r.record_id, "text": r.text, "language": r.language,
         "labels": [[s.entity_type, s.start, s.end] for s in (r.labels or ())]}
        for r in synthesize_mirror(real, seed=3)
    ])
    assert is_synthetic_mirror(mirror_ds) is True

    # near-miss real data must NOT be misclassified: text starts with "Synthetic record"
    # but the ids are real, and ids that look mirror-ish but text does not.
    near_a = load_records([{"id": "r0", "text": "Synthetic record 0 (en). hello world here"}])
    near_b = load_records([{"id": "mirror-deadbeef-0", "text": "totally real free text here"}])
    assert is_synthetic_mirror(near_a) is False
    assert is_synthetic_mirror(near_b) is False
    # a partial mirror (one real record mixed in) is NOT treated as a mirror -> regenerate
    partial = load_records([
        {"id": "mirror-deadbeef-0", "text": "Synthetic record 0 (en). a@b.com x",
         "labels": [["EMAIL_ADDRESS", 26, 33]]},
        {"id": "r1", "text": "Record 1: contact user1@host.example.com now."},
    ])
    assert is_synthetic_mirror(partial) is False


def test_forward_run_still_ships_full_mirror_and_gates(tmp_path) -> None:
    """The fix must NOT degrade the real-data path: a forward run still ships a non-empty
    synthetic mirror, and the full egress gate still runs on every artifact."""
    out = tmp_path / "fwd"
    rep = run_assurance_report(AssuranceConfig(
        records=_email_records(60), pipeline=_email_pipeline(), dimensions=("detection",),
        outputs=("json",), out_dir=str(out), seed=42, n_resamples=60))
    mirror = (out / "repro-bundle" / "synthetic-mirror.jsonl").read_text()
    assert mirror.strip()  # non-empty: real-data feature intact
    # the real-data run carries the normal claim, NOT the round-trip note
    assert not any("input is itself a synthetic mirror" in line for line in rep.limitations)


def test_roundtrip_emits_empty_mirror_with_honest_note(tmp_path) -> None:
    """On a round-trip the regenerated mirror is intentionally empty and the report says so."""
    out_a = tmp_path / "a"
    run_assurance_report(AssuranceConfig(
        records=_email_records(40), pipeline=_email_pipeline(), dimensions=("detection",),
        outputs=("json",), out_dir=str(out_a), seed=42, n_resamples=60))
    mirror_rows = [json.loads(line) for line in
                   (out_a / "repro-bundle" / "synthetic-mirror.jsonl").read_text().splitlines() if line.strip()]
    out_b = tmp_path / "b"
    rep = run_assurance_report(AssuranceConfig(
        records=mirror_rows, pipeline=_email_pipeline(), dimensions=("detection",),
        outputs=("json",), out_dir=str(out_b), seed=42, n_resamples=60))
    assert (out_b / "repro-bundle" / "synthetic-mirror.jsonl").read_text().strip() == ""
    assert any("input is itself a synthetic mirror" in line for line in rep.limitations)


def test_roundtrip_clean_across_seeds(tmp_path) -> None:
    """Robustness sweep: the round-trip must pass both gates for every seed, not just the
    one that happens to dodge the (pre-fix) probabilistic collision."""
    for seed in (1, 42, 12345):
        out_a = tmp_path / f"a{seed}"
        run_assurance_report(AssuranceConfig(
            records=_email_records(60), pipeline=_email_pipeline(), dimensions=("detection",),
            outputs=("json",), out_dir=str(out_a), seed=seed, n_resamples=60))
        rows = [json.loads(line) for line in
                (out_a / "repro-bundle" / "synthetic-mirror.jsonl").read_text().splitlines() if line.strip()]
        out_b = tmp_path / f"b{seed}"
        # must not raise (AssuranceRunError on any egress finding)
        run_assurance_report(AssuranceConfig(
            records=rows, pipeline=_email_pipeline(), dimensions=("detection",),
            outputs=("json",), out_dir=str(out_b), seed=seed, n_resamples=60))
        assert (out_b / "report.json").is_file()
