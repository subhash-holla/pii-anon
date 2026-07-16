"""Regression tests for the upheld findings from the round-17 (certified-path) close."""

from __future__ import annotations

import re

import pytest

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.claim_strength import ClaimStrength, DimensionResult
from pii_anon.assurance.provenance import build_provenance
from pii_anon.assurance.report import AssuranceReport, NoFabricationError
from pii_anon.assurance.spans import canonical_type

_ZWSP = "​"


# --- #1 invisible-character laundering scores as a leak (PII stays readable) ------
def test_invisible_char_laundering_scored_as_leak(tmp_path) -> None:
    rows = []
    for i in range(60):
        e = f"user{i}smith@host.example.com"
        text = f"Contact {e} for record {i}."
        st = text.index(e)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL", st, st + len(e)]]})

    def launder(text: str) -> str:  # insert a ZERO WIDTH SPACE inside each email local part
        return re.sub(r"(\S+)@(\S+)",
                      lambda m: m.group(1)[:3] + _ZWSP + m.group(1)[3:] + "@" + m.group(2), text)

    report = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="leaky", detect=lambda t: [], transform=launder,
                                               version="1", deterministic=True),
        dimensions=("leakage",), outputs=("json",), out_dir=str(tmp_path), seed=42, n_resamples=300))
    # the email is fully human-readable after the invisible-char insert -> must score ~100% leak
    assert report.dimensions["leakage"]["leaky"].value == 1.0


# --- #2 a surrogate in an entity type buckets cleanly, never crashes the run ------
def test_surrogate_entity_type_buckets() -> None:
    out = canonical_type(chr(0xDCE9) + "type")
    assert out.startswith("other:")  # no UnicodeEncodeError


def test_surrogate_gold_type_end_to_end_completes(tmp_path) -> None:
    rows = [{"id": f"r{i}", "text": "x a@b.com y", "labels": [[chr(0xDCE9) + "T", 2, 9]]} for i in range(20)]
    run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="u", detect=lambda t: [], transform=lambda t: t,
                                               version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=1, n_resamples=200))
    assert (tmp_path / "report.json").is_file()


# --- OBSERVATION hardened: a delta outside [-1,1] is rejected ---------------------
def test_delta_out_of_range_rejected() -> None:
    prov = build_provenance(
        dataset_name="d", dataset_fingerprint="abc", seed=1, pii_anon_version="0",
        user_pipeline_name="u", user_pipeline_version="1", user_pipeline_deterministic=True,
        user_pipeline_output_hash="h", timestamp="t",
    )
    res = DimensionResult("detection", "u", ClaimStrength.ADVISORY, value=0.5, support=60)
    report = AssuranceReport(
        "d", "abc", "labeled", "gold", 60, ("u", "pii-anon"), "pii-anon", prov,
        {"detection": {"u": res}},
        detection_comparisons={"u": {"delta": 5.0, "ci": None, "p_value": 0.01,
                                     "significant_holm": False, "effect_size": 0.0, "n": 60,
                                     "claim_strength": "advisory"}},
    )
    with pytest.raises(NoFabricationError):
        report.to_dict()
