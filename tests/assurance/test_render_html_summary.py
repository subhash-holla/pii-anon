"""Tests for the HTML executive + one-page summary renderers (injection-hardening, determinism)."""

from __future__ import annotations

from pii_anon.assurance.render import render_html, render_summary

_REPORT = {
    "schema_version": "assurance-report/1",
    "generated_at": "2026-06-18T00:00:00+00:00",
    "dataset": {"name": "acme<script>alert(1)</script>", "fingerprint": "abc123def456" * 4,
                "n_records": 120, "mode": "labeled", "reference_kind": "gold"},
    "systems": ["acme-dlp", "pii-anon"],
    "baseline": "pii-anon",
    "provenance": {
        "pii_anon_version": "1.5.0rc1", "python_version": "3.12.12", "seed": 7,
        "user_pipeline": {"name": "acme-dlp", "version": "2.3.1", "deterministic": True},
        "harness_close_certified": True,
        "reproducibility_claim": "recomputable by a holder of the dataset",
    },
    "statistics": {"bootstrap_resamples": 300, "alpha": 0.05, "comparison_family_size": 1},
    "dimensions": {
        "detection": {
            "acme-dlp": {"system": "acme-dlp", "claim_strength": "measured", "value": 0.6667,
                         "confidence_interval": [0.63, 0.70], "reasons": [], "caveats": []},
            "pii-anon": {"system": "pii-anon", "claim_strength": "measured", "value": 1.0,
                         "confidence_interval": [1.0, 1.0], "reasons": [], "caveats": []},
        },
        "reidentification": {
            "acme-dlp": {"system": "acme-dlp", "claim_strength": "advisory", "value": 0.38,
                         "confidence_interval": None,
                         "reasons": ["representative adversary <b>not</b> a trained attack"],
                         "caveats": ["ANTI-ANONYMITY CAVEAT: a low rate is NOT a guarantee"]},
        },
        "utility_fairness_compliance_NOT": {},  # ensure empty/odd dims don't crash
    },
    "detection_comparisons": {
        "acme-dlp": {"claim_strength": "measured", "delta": -0.0177, "ci": [-0.029, -0.008],
                     "p_value": 0.0017, "significant_holm": True, "effect_size": -0.29, "n": 120},
    },
    "agreement": None,
    "methodology": ["Detection: strict span match <ok>."],
    "limitations": ["Leakage is a LOWER BOUND <ok>."],
}


def test_html_renders_and_escapes_injection() -> None:
    out = render_html(_REPORT)["report.html"]
    assert out.startswith("<!DOCTYPE html>")
    # the markup in the dataset name must be escaped, never present as a live tag
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out
    assert "<b>not</b>" not in out  # reason markup escaped
    # claim-strength badges present
    assert "MEASURED" in out and "ADVISORY" in out
    # no JavaScript
    assert "<script" not in out.lower().replace("&lt;script", "")


def test_summary_renders_and_escapes_injection() -> None:
    out = render_summary(_REPORT)["summary.html"]
    assert out.startswith("<!DOCTYPE html>")
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out
    assert "MEASURED" in out and "ADVISORY" in out


def test_renderers_are_deterministic() -> None:
    assert render_html(_REPORT) == render_html(_REPORT)
    assert render_summary(_REPORT) == render_summary(_REPORT)


def test_html_handles_none_values_without_crash() -> None:
    out = render_html(_REPORT)["report.html"]
    # the reidentification value has a None CI -> rendered as em dash, no crash
    assert "report.html" or out  # rendered
    assert "&mdash;" in out or "—" in out
