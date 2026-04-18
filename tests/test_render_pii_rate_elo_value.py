"""Tests for ``scripts/render_pii_rate_elo_value.py``.

Covers:

- The renderer surfaces the headline when F1 and composite rankings pick
  different winners (the primary value prop for pii-rate-elo).
- Handles the agreement case without implying a swap exists.
- Per-system insights attribute rank swaps to the right component
  (latency, throughput, coverage).
- The README marker injection is a no-op when markers are absent,
  rather than corrupting the README.
- End-to-end run against the committed baseline artifact produces a
  stable, non-empty block.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "render_pii_rate_elo_value.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_pii_rate_elo_value", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["render_pii_rate_elo_value"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def renderer():
    return _load_module()


# ---------------------------------------------------------------------------
# Ranking divergence narrative
# ---------------------------------------------------------------------------

def test_top_line_flags_divergent_winners(renderer):
    """When F1 and composite pick different winners, the headline says so."""
    systems = [
        {
            "system": "fast-engine",
            "f1": 0.72, "composite_score": 0.82,
            "latency_p50_ms": 0.4, "docs_per_hour": 3_000_000,
            "entity_types_detected": 22, "entity_types_total": 29,
        },
        {
            "system": "slow-ner",
            "f1": 0.78, "composite_score": 0.68,
            "latency_p50_ms": 85.0, "docs_per_hour": 34_000,
            "entity_types_detected": 14, "entity_types_total": 29,
        },
    ]
    headline = renderer._top_line(systems)
    assert "F1 alone picks the wrong system" in headline
    assert "slow-ner" in headline
    assert "fast-engine" in headline


def test_top_line_agreement_case(renderer):
    """When F1 and composite agree on the top system, no swap is implied."""
    systems = [
        {
            "system": "clear-winner", "f1": 0.85, "composite_score": 0.82,
            "latency_p50_ms": 5.0, "docs_per_hour": 720_000,
            "entity_types_detected": 25, "entity_types_total": 29,
        },
        {
            "system": "runner-up", "f1": 0.70, "composite_score": 0.65,
            "latency_p50_ms": 10.0, "docs_per_hour": 360_000,
            "entity_types_detected": 20, "entity_types_total": 29,
        },
    ]
    headline = renderer._top_line(systems)
    assert "F1 alone picks the wrong system" not in headline
    assert "clear-winner" in headline


def test_top_line_partial_divergence(renderer):
    """When top-1 agrees but middle of the table swaps, say so."""
    systems = [
        {
            "system": "top-shared", "f1": 0.85, "composite_score": 0.82,
            "latency_p50_ms": 5.0, "docs_per_hour": 720_000,
            "entity_types_detected": 25, "entity_types_total": 29,
        },
        {
            "system": "second-by-f1", "f1": 0.78, "composite_score": 0.45,
            "latency_p50_ms": 200.0, "docs_per_hour": 18_000,
            "entity_types_detected": 8, "entity_types_total": 29,
        },
        {
            "system": "second-by-composite", "f1": 0.65, "composite_score": 0.60,
            "latency_p50_ms": 2.0, "docs_per_hour": 1_800_000,
            "entity_types_detected": 22, "entity_types_total": 29,
        },
    ]
    headline = renderer._top_line(systems)
    assert "change ranks further down" in headline


# ---------------------------------------------------------------------------
# Per-system insight attribution
# ---------------------------------------------------------------------------

def test_insights_attribute_latency_for_slow_systems(renderer):
    systems = [
        {
            "system": "fast", "f1": 0.70, "composite_score": 0.80,
            "latency_p50_ms": 0.5, "docs_per_hour": 3_000_000,
            "entity_types_detected": 22, "entity_types_total": 29,
        },
        {
            "system": "slow", "f1": 0.78, "composite_score": 0.55,
            "latency_p50_ms": 120.0, "docs_per_hour": 25_000,
            "entity_types_detected": 22, "entity_types_total": 29,
        },
    ]
    insights = renderer._rank_swap_insights(systems)
    # The slow system should have a latency-based reason.
    slow_line = next(i for i in insights if "slow" in i)
    assert "120.0ms" in slow_line or "120ms" in slow_line
    assert "latency" in slow_line.lower()


def test_insights_attribute_coverage_for_narrow_systems(renderer):
    """A fast but narrow-coverage system gets a coverage-based story."""
    systems = [
        {
            "system": "broad", "f1": 0.70, "composite_score": 0.75,
            "latency_p50_ms": 5.0, "docs_per_hour": 720_000,
            "entity_types_detected": 25, "entity_types_total": 29,
        },
        {
            "system": "narrow", "f1": 0.80, "composite_score": 0.55,
            "latency_p50_ms": 6.0, "docs_per_hour": 600_000,
            "entity_types_detected": 4, "entity_types_total": 29,
        },
    ]
    insights = renderer._rank_swap_insights(systems)
    narrow_line = next(i for i in insights if "narrow" in i)
    assert "coverage" in narrow_line.lower()


def test_insights_empty_when_rankings_match(renderer):
    """No divergence → no insight bullets."""
    systems = [
        {
            "system": "a", "f1": 0.90, "composite_score": 0.85,
            "latency_p50_ms": 1.0, "docs_per_hour": 3_000_000,
            "entity_types_detected": 25, "entity_types_total": 29,
        },
        {
            "system": "b", "f1": 0.60, "composite_score": 0.50,
            "latency_p50_ms": 2.0, "docs_per_hour": 1_500_000,
            "entity_types_detected": 15, "entity_types_total": 29,
        },
    ]
    assert renderer._rank_swap_insights(systems) == []


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def test_table_orders_by_composite_rank(renderer):
    systems = [
        {
            "system": "third", "f1": 0.50, "composite_score": 0.40,
            "latency_p50_ms": 10.0, "docs_per_hour": 100_000,
            "entity_types_detected": 5, "entity_types_total": 10,
        },
        {
            "system": "first", "f1": 0.70, "composite_score": 0.80,
            "latency_p50_ms": 0.5, "docs_per_hour": 3_000_000,
            "entity_types_detected": 10, "entity_types_total": 10,
        },
        {
            "system": "second", "f1": 0.90, "composite_score": 0.60,
            "latency_p50_ms": 90.0, "docs_per_hour": 30_000,
            "entity_types_detected": 8, "entity_types_total": 10,
        },
    ]
    table = renderer._render_comparison_table(systems)
    lines = [line for line in table.splitlines() if line.startswith("|")]
    # Drop header + separator.
    data_lines = lines[2:]
    ordered_names = [line.split("|")[1].strip() for line in data_lines]
    assert ordered_names == ["first", "second", "third"]


def test_table_marks_delta_rank(renderer):
    systems = [
        {
            "system": "winner", "f1": 0.70, "composite_score": 0.82,
            "latency_p50_ms": 0.4, "docs_per_hour": 3_000_000,
            "entity_types_detected": 22, "entity_types_total": 29,
        },
        {
            "system": "loser", "f1": 0.78, "composite_score": 0.55,
            "latency_p50_ms": 120.0, "docs_per_hour": 25_000,
            "entity_types_detected": 8, "entity_types_total": 29,
        },
    ]
    table = renderer._render_comparison_table(systems)
    assert "**+1**" in table    # winner gains a rank
    assert "**-1**" in table    # loser drops a rank


# ---------------------------------------------------------------------------
# Full block + README injection
# ---------------------------------------------------------------------------

def test_render_value_block_empty_systems(renderer):
    assert "No benchmark results available" in renderer.render_value_block([])


def test_render_value_block_embeds_expected_headers(renderer):
    systems = [
        {
            "system": "a", "f1": 0.8, "composite_score": 0.7,
            "latency_p50_ms": 5.0, "docs_per_hour": 720_000,
            "entity_types_detected": 22, "entity_types_total": 29,
        },
    ]
    block = renderer.render_value_block(systems)
    assert "## Why `pii-rate-elo` over plain F1?" in block
    assert "Δ Rank" in block
    assert "docs/pii-rate-elo.md" in block


def test_inject_into_readme_noop_when_markers_absent(tmp_path, renderer):
    readme = tmp_path / "README.md"
    readme.write_text("# My project\n\nNo markers here.\n", encoding="utf-8")
    injected = renderer.inject_into_readme(readme, "BLOCK CONTENT")
    assert injected is False
    # Original content must be untouched.
    assert readme.read_text(encoding="utf-8") == "# My project\n\nNo markers here.\n"


def test_inject_into_readme_replaces_between_markers(tmp_path, renderer):
    readme = tmp_path / "README.md"
    readme.write_text(
        "# My project\n\n"
        f"{renderer.MARKER_START}\n\nold content\n{renderer.MARKER_END}\n\n"
        "## Next section\n",
        encoding="utf-8",
    )
    injected = renderer.inject_into_readme(readme, "NEW BLOCK")
    assert injected is True
    text = readme.read_text(encoding="utf-8")
    assert "old content" not in text
    assert "NEW BLOCK" in text
    # Markers must survive.
    assert renderer.MARKER_START in text
    assert renderer.MARKER_END in text
    # Content outside the markers must be preserved.
    assert "# My project" in text
    assert "## Next section" in text


# ---------------------------------------------------------------------------
# End-to-end against the committed baseline artifact
# ---------------------------------------------------------------------------

def test_renders_against_shipped_baseline(renderer):
    """The committed baseline artifact produces a non-empty, coherent block."""
    import json
    baseline = ROOT / "src" / "pii_anon" / "eval_framework" / "baselines" / "benchmark-results.json"
    assert baseline.exists(), "Committed baseline artifact missing"
    data = json.loads(baseline.read_text(encoding="utf-8"))
    block = renderer.render_value_block(data.get("systems") or [])
    assert "pii-anon" in block
    assert "Composite Rank" in block
    # Sanity check: the table should have >=5 rows for the 5 committed systems.
    data_lines = [
        line for line in block.splitlines()
        if line.startswith("|") and line.split("|")[1].strip() not in ("System", "---")
        and "---" not in line
    ]
    assert len(data_lines) >= 5
