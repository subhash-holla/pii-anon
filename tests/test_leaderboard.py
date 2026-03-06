"""Tests for scorecard and leaderboard modules.

Tests cover scorecard serialization, leaderboard sorting, and multi-format
export (JSON, Markdown, CSV).
"""

from __future__ import annotations

import json
import os
import tempfile

from pii_anon.eval_framework.rating.leaderboard import (
    Leaderboard,
    LeaderboardExporter,
)
from pii_anon.eval_framework.rating.scorecard import (
    BenchmarkScorecard,
    SystemScorecard,
)


# ---------------------------------------------------------------------------
# SystemScorecard
# ---------------------------------------------------------------------------

class TestSystemScorecard:
    def test_default_values(self):
        sc = SystemScorecard(system_name="test")
        assert sc.system_name == "test"
        assert sc.available is True
        assert sc.composite_score == 0.0
        assert sc.elo_rating == 1500.0

    def test_to_dict(self):
        sc = SystemScorecard(
            system_name="pii-anon",
            f1=0.85,
            precision=0.9,
            recall=0.8,
            latency_p50_ms=0.01,
            docs_per_hour=168_000_000.0,
            composite_score=0.87,
            elo_rating=1650.0,
            elo_rd=120.0,
        )
        d = sc.to_dict()
        assert d["system_name"] == "pii-anon"
        assert d["f1"] == 0.85
        assert d["composite_score"] == 0.87
        assert d["elo_rating"] == 1650.0

    def test_to_dict_rounding(self):
        sc = SystemScorecard(
            system_name="test",
            f1=0.8512345678,
        )
        d = sc.to_dict()
        assert d["f1"] == 0.851235  # rounded to 6 decimals


# ---------------------------------------------------------------------------
# BenchmarkScorecard
# ---------------------------------------------------------------------------

class TestBenchmarkScorecard:
    def test_add_system(self):
        bs = BenchmarkScorecard(benchmark_name="test-bench")
        sc = SystemScorecard(system_name="sys-a")
        bs.add_system(sc)
        assert bs.get_system("sys-a") is sc

    def test_get_system_missing_returns_none(self):
        bs = BenchmarkScorecard(benchmark_name="test-bench")
        assert bs.get_system("nonexistent") is None

    def test_to_dict(self):
        bs = BenchmarkScorecard(benchmark_name="test-bench", dataset_name="test-ds")
        bs.add_system(SystemScorecard(system_name="a", f1=0.9))
        bs.add_system(SystemScorecard(system_name="b", f1=0.7))
        d = bs.to_dict()
        assert d["benchmark_name"] == "test-bench"
        assert "a" in d["systems"]
        assert "b" in d["systems"]

    def test_replace_system(self):
        bs = BenchmarkScorecard(benchmark_name="test-bench")
        bs.add_system(SystemScorecard(system_name="a", f1=0.5))
        bs.add_system(SystemScorecard(system_name="a", f1=0.9))
        assert bs.get_system("a").f1 == 0.9


# ---------------------------------------------------------------------------
# Leaderboard sorting
# ---------------------------------------------------------------------------

def _make_leaderboard() -> Leaderboard:
    return Leaderboard(
        benchmark_name="test",
        systems=[
            SystemScorecard(system_name="low", composite_score=0.3, elo_rating=1400, f1=0.4),
            SystemScorecard(system_name="mid", composite_score=0.6, elo_rating=1500, f1=0.6),
            SystemScorecard(system_name="high", composite_score=0.9, elo_rating=1600, f1=0.9),
        ],
    )


class TestLeaderboardSorting:
    def test_sort_by_composite(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        assert lb.systems[0].system_name == "high"
        assert lb.systems[-1].system_name == "low"

    def test_sort_by_elo(self):
        lb = _make_leaderboard()
        lb.sort_by_elo()
        assert lb.systems[0].system_name == "high"
        assert lb.systems[-1].system_name == "low"

    def test_sort_by_f1(self):
        lb = _make_leaderboard()
        lb.sort_by_f1()
        assert lb.systems[0].system_name == "high"
        assert lb.systems[-1].system_name == "low"


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------

class TestLeaderboardExportJSON:
    def test_to_json_valid(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        j = lb.to_json()
        data = json.loads(j)
        assert data["benchmark_name"] == "test"
        assert len(data["systems"]) == 3

    def test_to_dict(self):
        lb = _make_leaderboard()
        d = lb.to_dict()
        assert "benchmark_name" in d
        assert "systems" in d


class TestLeaderboardExportMarkdown:
    def test_markdown_header(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        md = lb.to_markdown()
        assert "# test — Leaderboard" in md

    def test_markdown_table_header(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        md = lb.to_markdown()
        assert "| Rank |" in md
        assert "| System |" in md or "System" in md

    def test_markdown_contains_all_systems(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        md = lb.to_markdown()
        assert "high" in md
        assert "mid" in md
        assert "low" in md

    def test_markdown_unavailable_system(self):
        lb = Leaderboard(
            benchmark_name="test",
            systems=[
                SystemScorecard(system_name="available", available=True, composite_score=0.5),
                SystemScorecard(system_name="unavailable", available=False),
            ],
        )
        md = lb.to_markdown()
        assert "unavailable" in md
        assert "—" in md  # em-dash for unavailable values


class TestLeaderboardExportCSV:
    def test_csv_header_row(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        csv_str = lb.to_csv()
        lines = csv_str.strip().split("\n")
        assert "rank" in lines[0]
        assert "system" in lines[0]
        assert "composite_score" in lines[0]

    def test_csv_data_rows(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        csv_str = lb.to_csv()
        lines = csv_str.strip().split("\n")
        assert len(lines) == 4  # header + 3 systems

    def test_csv_rank_order(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        csv_str = lb.to_csv()
        lines = csv_str.strip().split("\n")
        assert "high" in lines[1]


# ---------------------------------------------------------------------------
# LeaderboardExporter (file export)
# ---------------------------------------------------------------------------

class TestLeaderboardExporter:
    def test_export_all_formats(self):
        lb = _make_leaderboard()
        lb.sort_by_composite()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = LeaderboardExporter.export(lb, tmpdir)
            assert "json" in paths
            assert "md" in paths
            assert "csv" in paths
            for path in paths.values():
                assert path.exists()

    def test_export_single_format(self):
        lb = _make_leaderboard()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = LeaderboardExporter.export(lb, tmpdir, formats=["json"])
            assert "json" in paths
            assert "md" not in paths

    def test_export_custom_prefix(self):
        lb = _make_leaderboard()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = LeaderboardExporter.export(
                lb, tmpdir, filename_prefix="my_bench",
            )
            for path in paths.values():
                assert "my_bench" in path.name

    def test_export_creates_directory(self):
        lb = _make_leaderboard()
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "nested", "dir")
            paths = LeaderboardExporter.export(lb, nested)
            assert os.path.isdir(nested)
            for path in paths.values():
                assert path.exists()
