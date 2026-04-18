"""Leaderboard generation and export for PII de-identification benchmarks.

Provides sorted views of system scorecards with multiple output formats
(JSON, Markdown table, CSV) for documentation and publishing.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .elo import PIIRateEloEngine
from .scorecard import BenchmarkScorecard, SystemScorecard


@dataclass
class Leaderboard:
    """Sorted leaderboard of system scorecards.

    Attributes
    ----------
    benchmark_name:
        Name of the benchmark this leaderboard represents.
    systems:
        List of system scorecards, sorted by the chosen ranking criterion.
    """

    benchmark_name: str
    systems: list[SystemScorecard] = field(default_factory=list)

    def sort_by_composite(self) -> None:
        """Sort systems by composite score (descending)."""
        self.systems.sort(key=lambda s: s.composite_score, reverse=True)

    def sort_by_elo(self) -> None:
        """Sort systems by Elo rating (descending)."""
        self.systems.sort(key=lambda s: s.elo_rating, reverse=True)

    def sort_by_f1(self) -> None:
        """Sort systems by F1 score (descending)."""
        self.systems.sort(key=lambda s: s.f1, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "systems": [s.to_dict() for s in self.systems],
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Export leaderboard as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Export leaderboard as a Markdown table.

        Columns: Rank, System, Composite, F1, Precision, Recall,
        Latency (ms), Throughput (docs/hr), Elo, RD
        """
        lines: list[str] = []
        lines.append(f"# {self.benchmark_name} — Leaderboard")
        lines.append("")
        lines.append(
            "| Rank | System | Composite | F1 | Precision | Recall "
            "| Latency (ms) | Throughput (docs/hr) | Elo | RD |"
        )
        lines.append(
            "|------|--------|-----------|-----|-----------|--------"
            "|--------------|----------------------|-----|-----|"
        )

        for idx, sc in enumerate(self.systems, start=1):
            if not sc.available:
                lines.append(
                    f"| {idx} | {sc.system_name} | — | — | — | — "
                    f"| — | — | — | — |"
                )
                continue

            lines.append(
                f"| {idx} "
                f"| {sc.system_name} "
                f"| {sc.composite_score:.4f} "
                f"| {sc.f1:.4f} "
                f"| {sc.precision:.4f} "
                f"| {sc.recall:.4f} "
                f"| {sc.latency_p50_ms:.3f} "
                f"| {sc.docs_per_hour:,.0f} "
                f"| {sc.elo_rating:.0f} "
                f"| {sc.elo_rd:.0f} |"
            )

        lines.append("")
        return "\n".join(lines)

    @classmethod
    def from_benchmark_scorecard(
        cls,
        bench: BenchmarkScorecard,
        *,
        engine: PIIRateEloEngine | None = None,
        sort_by: str = "elo",
    ) -> "Leaderboard":
        """Build a ranked leaderboard from a :class:`BenchmarkScorecard`.

        If *engine* already has ratings (e.g. because the caller ran a
        tournament beforehand), they are preserved; otherwise a fresh
        round-robin is run on the scorecards' composite scores.

        Parameters
        ----------
        bench:
            Container of per-system scorecards.
        engine:
            Optional Elo engine carrying pre-computed ratings.
        sort_by:
            ``"elo"`` (default), ``"composite"``, or ``"f1"``.
        """
        # Copy scorecards before writing Elo ratings onto them.  The
        # source ``BenchmarkScorecard`` may be handed in multiple times
        # (e.g. to build two leaderboards with different sort keys) so
        # mutating its stored objects would leak Elo state across calls
        # and make a second tournament start from previously-updated
        # ratings rather than the untrained 1500/350 defaults.
        systems = [dataclasses.replace(sc) for sc in bench.system_scorecards.values()]

        if engine is None:
            engine = PIIRateEloEngine()
            engine.run_round_robin(
                {sc.system_name: sc.composite_score for sc in systems}
            )

        for sc in systems:
            rating = engine.get_rating(sc.system_name)
            if rating is not None:
                sc.elo_rating = rating.rating
                sc.elo_rd = rating.rd

        board = cls(benchmark_name=bench.benchmark_name, systems=systems)
        sort_fn = {
            "elo": board.sort_by_elo,
            "composite": board.sort_by_composite,
            "f1": board.sort_by_f1,
        }.get(sort_by)
        if sort_fn is None:
            raise ValueError(
                f"sort_by must be 'elo', 'composite', or 'f1' (got {sort_by!r})"
            )
        sort_fn()
        return board

    def to_csv(self) -> str:
        """Export leaderboard as CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "rank", "system", "composite_score", "f1", "precision",
            "recall", "latency_p50_ms", "docs_per_hour", "elo_rating", "elo_rd",
        ])
        for idx, sc in enumerate(self.systems, start=1):
            writer.writerow([
                idx,
                sc.system_name,
                round(sc.composite_score, 6),
                round(sc.f1, 6),
                round(sc.precision, 6),
                round(sc.recall, 6),
                round(sc.latency_p50_ms, 3),
                round(sc.docs_per_hour, 2),
                round(sc.elo_rating, 2),
                round(sc.elo_rd, 2),
            ])
        return output.getvalue()


class LeaderboardExporter:
    """Export leaderboard to multiple file formats.

    Usage::

        exporter = LeaderboardExporter()
        paths = exporter.export(leaderboard, output_dir, formats=["json", "md", "csv"])
    """

    @staticmethod
    def export(
        leaderboard: Leaderboard,
        output_dir: str | Path,
        *,
        formats: list[str] | None = None,
        filename_prefix: str = "leaderboard",
    ) -> dict[str, Path]:
        """Export leaderboard to files in *output_dir*.

        Parameters
        ----------
        leaderboard:
            The leaderboard to export.
        output_dir:
            Directory to write files to.
        formats:
            List of output formats.  Supported: "json", "md", "csv".
            Default: all three.
        filename_prefix:
            Prefix for output filenames.

        Returns
        -------
        dict[str, Path]
            Mapping of format → output file path.
        """
        supported = {"json", "md", "csv"}
        chosen = set(formats or ["json", "md", "csv"]) & supported
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        result: dict[str, Path] = {}

        if "json" in chosen:
            path = out_dir / f"{filename_prefix}.json"
            path.write_text(leaderboard.to_json(), encoding="utf-8")
            result["json"] = path

        if "md" in chosen:
            path = out_dir / f"{filename_prefix}.md"
            path.write_text(leaderboard.to_markdown(), encoding="utf-8")
            result["md"] = path

        if "csv" in chosen:
            path = out_dir / f"{filename_prefix}.csv"
            path.write_text(leaderboard.to_csv(), encoding="utf-8")
            result["csv"] = path

        return result
