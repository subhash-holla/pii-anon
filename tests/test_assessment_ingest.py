"""Assessment-artifact ingestion + per-entity Elo tournament (sp2).

Fixture artifacts follow the ``pii-anon-baseline-results/v1`` shape produced
by the pii-anon-eval-data ``pii-anon baselines`` harness. Validation tests
exercise the no-fabrication discipline: NaN / bool / out-of-unit / blank-name
values must be refused loudly, never defaulted.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pii_anon.eval_framework.rating.assessment_ingest import (
    load_assessment,
    render_assessment_report,
    run_assessment_tournament,
)

_SCHEMA = "pii-anon-baseline-results/v1"


def _entity_row(f2: float, tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {"f2": f2, "counts": {"tp": tp, "fp": fp, "fn": fn, "partial": 0}}


def _detector(
    *,
    precision: float,
    recall: float,
    f1: float,
    f2: float,
    f2_macro: float,
    reachable: int,
    by_entity: dict[str, dict[str, Any]],
    status: str = "scored",
) -> dict[str, Any]:
    return {
        "status": status,
        "model_id": "test",
        "deterministic": True,
        "n_records": 10,
        "n_gold": sum(r["counts"]["tp"] + r["counts"]["fn"] for r in by_entity.values()),
        "n_pred": sum(r["counts"]["tp"] + r["counts"]["fp"] for r in by_entity.values()),
        "coverage": {"reachable": reachable, "of_total": 63},
        "micro": {"precision": precision, "recall": recall, "f1": f1, "f2": f2},
        "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": f2_macro},
        "by_entity_type": by_entity,
    }


def _write_artifact(tmp_path: Path, detectors: dict[str, Any]) -> Path:
    artifact = {
        "schema": _SCHEMA,
        "harness_version": "1.0.0",
        "matching_policy": "strict-v1",
        "dataset": {
            "n_records": 10,
            "n_gold": 100,
            "split": "dev",
            "languages": ["en"],
            "name": "pii-anon-eval-data",
            "version": "2.0.0",
        },
        "ranking": [],
        "detectors": detectors,
    }
    path = tmp_path / "baseline_results.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _three_player_detectors() -> dict[str, Any]:
    # alpha dominates everywhere; beta middles; gamma is weak and blind to
    # PHONE_NUMBER (f2 0.0 with gold present — capability is part of the contest).
    return {
        "alpha": _detector(
            precision=0.9, recall=0.9, f1=0.9, f2=0.9, f2_macro=0.8, reachable=40,
            by_entity={
                "EMAIL_ADDRESS": _entity_row(0.95, 19, 1, 1),
                "PERSON_NAME": _entity_row(0.90, 18, 2, 2),
                "PHONE_NUMBER": _entity_row(0.85, 17, 3, 3),
            },
        ),
        "beta": _detector(
            precision=0.7, recall=0.7, f1=0.7, f2=0.7, f2_macro=0.5, reachable=25,
            by_entity={
                "EMAIL_ADDRESS": _entity_row(0.75, 15, 5, 5),
                "PERSON_NAME": _entity_row(0.70, 14, 6, 6),
                "PHONE_NUMBER": _entity_row(0.65, 13, 7, 7),
            },
        ),
        "gamma": _detector(
            precision=0.4, recall=0.3, f1=0.34, f2=0.31, f2_macro=0.2, reachable=10,
            by_entity={
                "EMAIL_ADDRESS": _entity_row(0.40, 8, 12, 12),
                "PERSON_NAME": _entity_row(0.20, 4, 16, 16),
                "PHONE_NUMBER": _entity_row(0.0, 0, 0, 20),
            },
        ),
    }


class TestLoadAssessment:
    def test_happy_path_parses_players_and_gold_supported_entities(self, tmp_path: Path) -> None:
        report = load_assessment(_write_artifact(tmp_path, _three_player_detectors()))
        assert [p.name for p in report.players] == ["alpha", "beta", "gamma"]
        alpha = report.players[0]
        assert alpha.f2 == 0.9
        assert alpha.coverage_reachable == 40
        assert alpha.coverage_total == 63
        assert set(alpha.per_entity_f2) == {"EMAIL_ADDRESS", "PERSON_NAME", "PHONE_NUMBER"}
        # gamma's PHONE_NUMBER has gold (fn=20) so its 0.0 row is KEPT — a
        # blind spot is a real score, not a missing axis.
        gamma = report.players[2]
        assert gamma.per_entity_f2["PHONE_NUMBER"] == 0.0

    def test_zero_gold_entity_rows_are_not_match_fields(self, tmp_path: Path) -> None:
        detectors = _three_player_detectors()
        for det in detectors.values():
            det["by_entity_type"]["BAR_NUMBER"] = _entity_row(0.0, 0, 0, 0)
        report = load_assessment(_write_artifact(tmp_path, detectors))
        for player in report.players:
            assert "BAR_NUMBER" not in player.per_entity_f2

    def test_unscored_player_is_excluded_and_disclosed(self, tmp_path: Path) -> None:
        detectors = _three_player_detectors()
        detectors["delta"] = _detector(
            precision=0.5, recall=0.5, f1=0.5, f2=0.5, f2_macro=0.4, reachable=5,
            by_entity={"EMAIL_ADDRESS": _entity_row(0.5, 10, 10, 10)},
            status="unavailable",
        )
        report = load_assessment(_write_artifact(tmp_path, detectors))
        assert [p.name for p in report.players] == ["alpha", "beta", "gamma"]
        assert report.excluded == (("delta", "unavailable"),)

    def test_wrong_schema_is_refused(self, tmp_path: Path) -> None:
        path = _write_artifact(tmp_path, _three_player_detectors())
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["schema"] = "something-else/v9"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        with pytest.raises(ValueError, match="schema"):
            load_assessment(path)

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            # NaN survives json round-trips (Python's json emits/accepts NaN) —
            # exactly the hostile-artifact case the validators exist for.
            (lambda d: d["alpha"]["micro"].__setitem__("f2", float("nan")), "alpha"),
            (lambda d: d["alpha"]["micro"].__setitem__("recall", True), "alpha"),
            (lambda d: d["beta"]["micro"].__setitem__("f2", 1.5), "beta"),
            (lambda d: d["beta"]["coverage"].__setitem__("reachable", True), "beta"),
            (
                lambda d: d["gamma"]["by_entity_type"]["EMAIL_ADDRESS"].__setitem__(
                    "f2", float("inf")
                ),
                "gamma",
            ),
        ],
    )
    def test_malformed_scored_player_is_refused_loudly(
        self, tmp_path: Path, mutate: Any, match: str
    ) -> None:
        detectors = _three_player_detectors()
        mutate(detectors)
        path = _write_artifact(tmp_path, detectors)
        with pytest.raises(ValueError, match=match):
            load_assessment(path)

    def test_blank_player_name_is_refused(self, tmp_path: Path) -> None:
        detectors = _three_player_detectors()
        detectors["   "] = detectors.pop("gamma")
        path = _write_artifact(tmp_path, detectors)
        with pytest.raises(ValueError, match="blank"):
            load_assessment(path)


class TestAssessmentTournament:
    def test_dominant_player_ranks_first(self, tmp_path: Path) -> None:
        report = load_assessment(_write_artifact(tmp_path, _three_player_detectors()))
        result = run_assessment_tournament(report)
        rankings = result["tournament"]["rankings"]
        assert [r["name"] for r in rankings] == ["alpha", "beta", "gamma"]
        assert rankings[0]["rating"] > 1500.0 > rankings[2]["rating"]

    def test_deterministic_across_runs(self, tmp_path: Path) -> None:
        path = _write_artifact(tmp_path, _three_player_detectors())
        result_a = run_assessment_tournament(load_assessment(path))
        result_b = run_assessment_tournament(load_assessment(path))
        assert result_a == result_b

    def test_exact_rating_anchors(self, tmp_path: Path) -> None:
        # Exact reference anchors (program lesson: pin representative metrics
        # at exact values, never bounds). 3 players x 3 match fields, sorted
        # field-major order, default engine parameters.
        report = load_assessment(_write_artifact(tmp_path, _three_player_detectors()))
        rankings = run_assessment_tournament(report)["tournament"]["rankings"]
        by_name = {r["name"]: r for r in rankings}
        assert by_name["alpha"]["rating"] == pytest.approx(1561.44, abs=0.01)
        assert by_name["beta"]["rating"] == pytest.approx(1505.27, abs=0.01)
        assert by_name["gamma"]["rating"] == pytest.approx(1430.82, abs=0.01)
        assert by_name["alpha"]["num_matches"] == 6

    def test_match_fields_are_gold_supported_union(self, tmp_path: Path) -> None:
        report = load_assessment(_write_artifact(tmp_path, _three_player_detectors()))
        result = run_assessment_tournament(report)
        assert result["match_fields"] == ["EMAIL_ADDRESS", "PERSON_NAME", "PHONE_NUMBER"]

    def test_fewer_than_two_players_refused(self, tmp_path: Path) -> None:
        detectors = {"alpha": _three_player_detectors()["alpha"]}
        report = load_assessment(_write_artifact(tmp_path, detectors))
        with pytest.raises(ValueError, match="at least 2"):
            run_assessment_tournament(report)


class TestRenderAssessmentReport:
    def test_report_contains_leaderboard_significance_and_axis_disclosure(
        self, tmp_path: Path
    ) -> None:
        report = load_assessment(_write_artifact(tmp_path, _three_player_detectors()))
        result = run_assessment_tournament(report)
        text = render_assessment_report(result)
        assert "| 1 | alpha |" in text
        assert "Pairwise significance" in text
        assert "Axes evaluated" in text
        # Axis honesty: latency / Tier-3 explicitly disclosed as absent.
        assert "Latency / throughput: NOT carried by this artifact" in text
        assert "Tier-3 re-identification resistance: NOT carried by this artifact" in text

    def test_report_lists_strengths_and_blind_spots(self, tmp_path: Path) -> None:
        report = load_assessment(_write_artifact(tmp_path, _three_player_detectors()))
        result = run_assessment_tournament(report)
        text = render_assessment_report(result)
        # gamma's PHONE_NUMBER blind spot (f2 0.0 with gold present) must surface.
        assert "gamma" in text
        assert "PHONE_NUMBER" in text


class TestRateEloAssessmentCLI:
    def test_cli_renders_markdown_and_writes_artifacts(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from pii_anon.cli import create_app

        artifact = _write_artifact(tmp_path, _three_player_detectors())
        out_dir = tmp_path / "ratings"
        runner = CliRunner()
        result = runner.invoke(
            create_app(),
            [
                "rate-elo-assessment",
                "--assessment-results",
                str(artifact),
                "--artifact-dir",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "| 1 | alpha |" in result.output
        assert (out_dir / "leaderboard.md").exists()
        tournament = json.loads((out_dir / "tournament.json").read_text(encoding="utf-8"))
        assert tournament["schema"] == "pii-rate-elo-assessment-tournament/v1"

    def test_cli_refuses_malformed_artifact(self, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from pii_anon.cli import create_app

        detectors = _three_player_detectors()
        detectors["alpha"]["micro"]["f2"] = float("nan")
        artifact = _write_artifact(tmp_path, detectors)
        runner = CliRunner()
        result = runner.invoke(
            create_app(), ["rate-elo-assessment", "--assessment-results", str(artifact)]
        )
        assert result.exit_code != 0
        assert "alpha" in result.output

    def test_directory_path_is_clean_error_not_traceback(self, tmp_path: Path) -> None:
        # IsADirectoryError is an OSError SIBLING of FileNotFoundError — the
        # S7-02 close-10 DoV class. Must surface as a clean error, no traceback.
        with pytest.raises(ValueError, match="not a readable file"):
            load_assessment(tmp_path)

    def test_deeply_nested_json_is_clean_error_not_recursionerror(
        self, tmp_path: Path
    ) -> None:
        # RecursionError is NOT a JSONDecodeError — the S7-02 CLI DoV class.
        # Version note: on Python <= 3.13 json.loads blows the recursion guard
        # ("could not parse"); Python 3.14's parser handles this nesting, so
        # the artifact PARSES and fails schema validation instead. The pinned
        # property is version-independent: a clean typed ValueError, never a
        # RecursionError leak.
        path = tmp_path / "deep.json"
        path.write_text("[" * 60000 + "]" * 60000, encoding="utf-8")
        with pytest.raises(ValueError):
            load_assessment(path)
