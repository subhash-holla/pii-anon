"""Render the pii-rate-elo value-proposition section of the README.

The headline from every benchmark run: different metrics rank the same
set of systems differently.  This renderer takes the ``systems`` list
out of ``benchmark-results.json`` and produces a Markdown block that
contrasts:

* The naive F1-only ranking (what most PII benchmarks report)
* The ``pii-rate-elo`` composite ranking (what production care-abouts
  yield when accuracy and cost are normalised onto one scale)

When the two rankings diverge — which they do for real systems because
F1 ignores latency, throughput, entity-type coverage, and Tier 3
re-identification resistance — the divergence itself is the argument
for the composite metric.  The script surfaces the largest rank swaps
and attributes them to the component driving the change.

The output is spliced into ``README.md`` between
``<!-- PII_RATE_ELO_VALUE_START -->`` and
``<!-- PII_RATE_ELO_VALUE_END -->`` markers.  If no benchmark artifact
is available, the existing block is preserved untouched (so a partial
or failed benchmark run does not empty the README's value prop).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

MARKER_START = "<!-- PII_RATE_ELO_VALUE_START -->"
MARKER_END = "<!-- PII_RATE_ELO_VALUE_END -->"


# ---------------------------------------------------------------------------
# Ranking utilities
# ---------------------------------------------------------------------------

def _sorted_by(systems: list[dict[str, Any]], key: str, descending: bool = True) -> list[dict[str, Any]]:
    return sorted(systems, key=lambda s: s.get(key, 0.0) or 0.0, reverse=descending)


def _rank_map(systems: list[dict[str, Any]], key: str, descending: bool = True) -> dict[str, int]:
    return {
        s["system"]: rank + 1
        for rank, s in enumerate(_sorted_by(systems, key, descending=descending))
    }


def _format_throughput(docs_per_hour: float) -> str:
    if docs_per_hour >= 1_000_000:
        return f"{docs_per_hour / 1_000_000:.1f}M/hr"
    if docs_per_hour >= 1_000:
        return f"{docs_per_hour / 1_000:.0f}K/hr"
    return f"{docs_per_hour:.0f}/hr"


# ---------------------------------------------------------------------------
# Narrative generation — attribute each rank-swap to a component
# ---------------------------------------------------------------------------

def _rank_swap_insights(systems: list[dict[str, Any]]) -> list[str]:
    """For each system whose F1 rank differs from its composite rank,
    explain *why*: latency too high, coverage too narrow, etc.

    Returns a list of plain-prose bullet strings.
    """
    f1_ranks = _rank_map(systems, "f1")
    comp_ranks = _rank_map(systems, "composite_score")
    insights: list[str] = []

    # Sort by absolute rank swap — biggest swap first.
    ordered = sorted(
        systems,
        key=lambda s: abs(f1_ranks[s["system"]] - comp_ranks[s["system"]]),
        reverse=True,
    )

    for system in ordered:
        name = system["system"]
        f1_rank = f1_ranks[name]
        comp_rank = comp_ranks[name]
        if f1_rank == comp_rank:
            continue

        swing = f1_rank - comp_rank  # positive = composite boosts this system
        f1 = system.get("f1", 0.0)
        composite = system.get("composite_score", 0.0)
        latency = system.get("latency_p50_ms", 0.0)
        throughput = system.get("docs_per_hour", 0.0)
        detected = system.get("entity_types_detected", 0)
        total = system.get("entity_types_total", 0)
        coverage_pct = (detected / total * 100) if total else 0.0

        # Figure out what moved the needle.  A system that jumps up in
        # composite usually wins on efficiency or coverage; one that
        # drops loses there.
        if swing > 0:  # composite is kinder than F1 to this system
            if latency > 0 and latency < 5:
                reason = f"its **{latency:.2f}ms p50 latency** ({_format_throughput(throughput)})"
            elif total and coverage_pct >= 80:
                reason = f"its **{coverage_pct:.0f}% entity-type coverage**"
            else:
                reason = "its operational profile"
            insights.append(
                f"**{name}** moves **#{f1_rank} → #{comp_rank}** (gains {swing}) — "
                f"F1 {f1:.3f} is middling, but {reason} "
                f"pushes the composite to {composite:.3f}."
            )
        else:  # composite is harsher than F1 to this system
            # What kept this system from turning F1 into composite?
            if latency > 50:
                reason = (
                    f"its **{latency:.1f}ms p50 latency** — {_format_throughput(throughput)} is "
                    "three orders of magnitude below the reference throughput"
                )
            elif total and coverage_pct < 60:
                reason = (
                    f"its **{coverage_pct:.0f}% entity-type coverage** "
                    f"({detected}/{total} types) leaves audits incomplete"
                )
            else:
                reason = "its operational profile"
            insights.append(
                f"**{name}** drops **#{f1_rank} → #{comp_rank}** (loses {-swing}) — "
                f"F1 {f1:.3f} looks strong, but {reason}, "
                f"so the composite lands at {composite:.3f}."
            )

    return insights


# ---------------------------------------------------------------------------
# Top-line narrative — the "why composite beats F1 alone" banner
# ---------------------------------------------------------------------------

def _top_line(systems: list[dict[str, Any]]) -> str:
    f1_ranks = _rank_map(systems, "f1")
    comp_ranks = _rank_map(systems, "composite_score")
    n_diverged = sum(
        1 for name in f1_ranks
        if f1_ranks[name] != comp_ranks[name]
    )
    f1_winner = next(
        s for s in _sorted_by(systems, "f1") if s.get("available", True)
    )
    comp_winner = next(
        s for s in _sorted_by(systems, "composite_score") if s.get("available", True)
    )

    if f1_winner["system"] != comp_winner["system"]:
        return (
            f"**F1 alone picks the wrong system.**  By F1, "
            f"`{f1_winner['system']}` looks like the winner "
            f"({f1_winner['f1']:.3f} vs {comp_winner['f1']:.3f}). "
            f"By `pii-rate-elo` composite — which folds in latency, throughput, "
            f"entity-type coverage, and (when available) Tier 3 re-identification "
            f"resistance — `{comp_winner['system']}` leads instead "
            f"({comp_winner['composite_score']:.3f} vs {f1_winner['composite_score']:.3f}). "
            f"{n_diverged} of {len(systems)} systems swap ranks between the two views."
        )

    # F1 and composite agree on the top — still useful if they diverge
    # in the middle of the table.
    if n_diverged:
        return (
            f"**F1 and `pii-rate-elo` agree on the top system (`{comp_winner['system']}`) — "
            f"but {n_diverged} of {len(systems)} systems change ranks further down** because "
            f"F1 ignores latency, throughput, and entity-type coverage. "
            f"Production decisions hinge on the composite view, not the F1 one."
        )
    return (
        f"**`{comp_winner['system']}` leads on both F1 and composite.** "
        f"Composite={comp_winner['composite_score']:.3f}, F1={comp_winner['f1']:.3f}. "
        f"The `pii-rate-elo` composite doesn't change the ranking here — but it tells you "
        f"**by how much** the leader wins on a single production-meaningful scale."
    )


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def _render_comparison_table(systems: list[dict[str, Any]]) -> str:
    f1_ranks = _rank_map(systems, "f1")
    comp_ranks = _rank_map(systems, "composite_score")
    # Display ordered by composite rank — that's the headline ranking.
    ordered = _sorted_by(systems, "composite_score")

    header = (
        "| System | F1 | F1 Rank | Composite | Composite Rank | Δ Rank | p50 Latency | Throughput | Coverage |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows: list[str] = []
    for system in ordered:
        name = system["system"]
        f1_rank = f1_ranks[name]
        comp_rank = comp_ranks[name]
        delta = f1_rank - comp_rank  # positive = composite boosts this system
        delta_str = (
            f"**+{delta}**" if delta > 0
            else f"**{delta}**" if delta < 0
            else "—"
        )
        latency = system.get("latency_p50_ms", 0.0)
        throughput = system.get("docs_per_hour", 0.0)
        detected = system.get("entity_types_detected", 0)
        total = system.get("entity_types_total", 0)
        coverage = f"{detected}/{total}" if total else "—"
        rows.append(
            f"| {name} "
            f"| {system.get('f1', 0.0):.3f} "
            f"| #{f1_rank} "
            f"| {system.get('composite_score', 0.0):.3f} "
            f"| #{comp_rank} "
            f"| {delta_str} "
            f"| {latency:.2f} ms "
            f"| {_format_throughput(throughput)} "
            f"| {coverage} |"
        )
    return header + "\n" + "\n".join(rows)


# ---------------------------------------------------------------------------
# Full block rendering
# ---------------------------------------------------------------------------

def render_value_block(systems: list[dict[str, Any]]) -> str:
    """Render the composite-vs-simple-metrics comparison as Markdown."""
    if not systems:
        return "_No benchmark results available._"

    table = _render_comparison_table(systems)
    top_line = _top_line(systems)
    insights = _rank_swap_insights(systems)

    insights_section = ""
    if insights:
        insights_section = "\n\n### Where the rankings diverge\n\n" + "\n".join(
            f"- {ins}" for ins in insights[:5]   # cap at top 5 for readability
        )

    return (
        f"## Why `pii-rate-elo` over plain F1?\n\n"
        f"{top_line}\n\n"
        f"{table}\n"
        f"{insights_section}\n\n"
        f"**Δ Rank** = F1 rank − Composite rank.  Positive means the composite "
        f"view promotes the system (it's operationally stronger than F1 suggests); "
        f"negative means the composite view demotes it (it's paying for F1 with "
        f"latency, missing entity types, or Tier 3 leakage).  See "
        f"[docs/pii-rate-elo.md](docs/pii-rate-elo.md) for the full algorithm.\n"
    )


# ---------------------------------------------------------------------------
# README injection
# ---------------------------------------------------------------------------

def inject_into_readme(readme_path: Path, block: str) -> bool:
    """Splice *block* into the README between the PII_RATE_ELO_VALUE markers.

    Returns ``True`` when the README was updated, ``False`` if the markers
    were not found (caller should report the gap).
    """
    text = readme_path.read_text(encoding="utf-8")
    start_idx = text.find(MARKER_START)
    end_idx = text.find(MARKER_END)
    if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
        return False
    new_text = (
        text[:start_idx]
        + MARKER_START + "\n\n"
        + block + "\n"
        + MARKER_END
        + text[end_idx + len(MARKER_END):]
    )
    if new_text != text:
        readme_path.write_text(new_text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the pii-rate-elo value-prop block and inject it into the README.",
    )
    parser.add_argument(
        "--input-json",
        required=True,
        type=Path,
        help="Path to benchmark-results.json.",
    )
    parser.add_argument(
        "--readme",
        default=Path("README.md"),
        type=Path,
        help="Path to the README to update.  Default: README.md.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
        help="Optional path to also write the rendered block as a standalone .md file.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the rendered block to stdout without touching the README.",
    )
    args = parser.parse_args()

    if not args.input_json.exists():
        print(f"ERROR: benchmark artifact not found: {args.input_json}", file=sys.stderr)
        sys.exit(2)

    data = json.loads(args.input_json.read_text(encoding="utf-8"))
    systems = data.get("systems") or []
    block = render_value_block(systems)

    if args.stdout:
        print(block)
        return

    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(block, encoding="utf-8")

    if not args.readme.exists():
        print(f"WARNING: README not found at {args.readme}; skipping injection.")
        return
    ok = inject_into_readme(args.readme, block)
    if ok:
        print(f"Updated {args.readme} pii-rate-elo value block.")
    else:
        print(
            f"WARNING: markers {MARKER_START} / {MARKER_END} not found in "
            f"{args.readme}; skipping injection. Add the marker pair to enable "
            "automatic updates.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
