"""Assessment-artifact ingestion for pii-rate-elo (sp2).

Ingests a merged ``pii-anon-baseline-results/v1`` leaderboard artifact (the
output of the pii-anon-eval-data ``pii-anon baselines`` harness) and rates
every scored player through the existing :class:`PIIRateEloEngine` via
**per-entity-type F2 matches**: each entity type with gold support is a match
field, every player pair plays it once, and a type a player cannot reach
scores 0.0 there — capability is part of the contest, exactly as the
assessment's coverage column discloses. With N players and E gold-supported
types that is ``E x N(N-1)/2`` matches, enough volume to shrink rating
deviations into the statistically meaningful range.

No-fabrication discipline (the program's standing rule): every value read
from the artifact that can influence a rating is validated — finite, in
[0, 1] where a rate; non-bool (``bool`` is an ``int`` subclass); non-blank
where a name. A malformed SCORED player is a hard :class:`ValueError` naming
the player and field; an UNSCORED player (``status != "scored"``) is excluded
and disclosed, never silently defaulted.

Axis honesty: this artifact carries detection quality + coverage ONLY. The
rendered report's axis-disclosure block says so explicitly; latency /
throughput / Tier-3 axes are never invented for players that lack them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from .elo import PIIRateEloEngine

__all__ = [
    "AssessmentPlayer",
    "AssessmentReport",
    "load_assessment",
    "render_assessment_report",
    "run_assessment_tournament",
]

_EXPECTED_SCHEMA = "pii-anon-baseline-results/v1"


# ---------------------------------------------------------------------------
# Validators (local twins of the SDO-gate discipline — the gate module is
# locked and must not become an import dependency of the rating package)
# ---------------------------------------------------------------------------

def _fail(ctx: str, reason: str) -> ValueError:
    return ValueError(f"assessment artifact: {ctx}: {reason}")


def _require_str(value: object, *, ctx: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(ctx, f"expected a non-blank string, got {value!r}")
    return value


def _require_int(value: object, *, ctx: str) -> int:
    # bool is an int subclass — a forged True would coerce to 1; refuse it.
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(ctx, f"expected an int (non-bool), got {value!r}")
    return value


def _require_unit(value: object, *, ctx: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(ctx, f"expected a number, got {value!r}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0 or out > 1.0:
        raise _fail(ctx, f"expected a finite value in [0, 1], got {value!r}")
    return out


def _require_mapping(value: object, *, ctx: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(ctx, f"expected an object, got {type(value).__name__}")
    return value


# ---------------------------------------------------------------------------
# Parsed shapes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AssessmentPlayer:
    """One scored detector row from the assessment artifact (validated)."""

    name: str
    precision: float
    recall: float
    f1: float
    f2: float
    f2_macro: float
    coverage_reachable: int
    coverage_total: int
    n_gold: int
    n_pred: int
    #: micro F2 per entity type, ONLY for types with gold support (tp+fn > 0).
    per_entity_f2: dict[str, float]


@dataclass(frozen=True, slots=True)
class AssessmentReport:
    """A validated assessment artifact: scored players + disclosed exclusions."""

    players: tuple[AssessmentPlayer, ...]
    #: (name, status) for every detector present but not scored — disclosed,
    #: never silently dropped.
    excluded: tuple[tuple[str, str], ...]
    dataset: dict[str, Any]
    matching_policy: str
    source_path: str


def _parse_player(name: str, raw: dict[str, Any]) -> AssessmentPlayer:
    micro = _require_mapping(raw.get("micro"), ctx=f"detectors[{name}].micro")
    macro = _require_mapping(raw.get("macro"), ctx=f"detectors[{name}].macro")
    coverage = _require_mapping(raw.get("coverage"), ctx=f"detectors[{name}].coverage")
    by_entity = _require_mapping(
        raw.get("by_entity_type"), ctx=f"detectors[{name}].by_entity_type"
    )

    per_entity_f2: dict[str, float] = {}
    for etype_raw, row_raw in by_entity.items():
        etype = _require_str(etype_raw, ctx=f"detectors[{name}].by_entity_type key")
        row = _require_mapping(row_raw, ctx=f"detectors[{name}].by_entity_type[{etype}]")
        counts = _require_mapping(
            row.get("counts"), ctx=f"detectors[{name}].by_entity_type[{etype}].counts"
        )
        tp = _require_int(counts.get("tp"), ctx=f"detectors[{name}].{etype}.counts.tp")
        fn = _require_int(counts.get("fn"), ctx=f"detectors[{name}].{etype}.counts.fn")
        if tp < 0 or fn < 0:
            raise _fail(f"detectors[{name}].{etype}.counts", "negative count")
        if tp + fn > 0:  # gold-supported type: its F2 is a real match score
            per_entity_f2[etype] = _require_unit(
                row.get("f2"), ctx=f"detectors[{name}].by_entity_type[{etype}].f2"
            )

    reachable = _require_int(
        coverage.get("reachable"), ctx=f"detectors[{name}].coverage.reachable"
    )
    of_total = _require_int(
        coverage.get("of_total"), ctx=f"detectors[{name}].coverage.of_total"
    )
    if of_total <= 0 or reachable < 0 or reachable > of_total:
        raise _fail(f"detectors[{name}].coverage", f"invalid {reachable}/{of_total}")

    return AssessmentPlayer(
        name=name,
        precision=_require_unit(micro.get("precision"), ctx=f"detectors[{name}].micro.precision"),
        recall=_require_unit(micro.get("recall"), ctx=f"detectors[{name}].micro.recall"),
        f1=_require_unit(micro.get("f1"), ctx=f"detectors[{name}].micro.f1"),
        f2=_require_unit(micro.get("f2"), ctx=f"detectors[{name}].micro.f2"),
        f2_macro=_require_unit(macro.get("f2"), ctx=f"detectors[{name}].macro.f2"),
        coverage_reachable=reachable,
        coverage_total=of_total,
        n_gold=_require_int(raw.get("n_gold"), ctx=f"detectors[{name}].n_gold"),
        n_pred=_require_int(raw.get("n_pred"), ctx=f"detectors[{name}].n_pred"),
        per_entity_f2=per_entity_f2,
    )


def load_assessment(path: str | Path) -> AssessmentReport:
    """Load + validate a merged assessment artifact (fail-loud on malformation)."""
    source = Path(path)
    artifact = _require_mapping(
        json.loads(source.read_text(encoding="utf-8")), ctx="top level"
    )

    schema = artifact.get("schema")
    if schema != _EXPECTED_SCHEMA:
        raise _fail("schema", f"expected {_EXPECTED_SCHEMA!r}, got {schema!r}")

    detectors = _require_mapping(artifact.get("detectors"), ctx="detectors")
    players: list[AssessmentPlayer] = []
    excluded: list[tuple[str, str]] = []
    for name_raw in sorted(detectors, key=str):
        name = _require_str(name_raw, ctx="detector name")
        if name != name_raw:
            raise _fail("detector name", f"blank or padded name {name_raw!r}")
        raw = _require_mapping(detectors[name_raw], ctx=f"detectors[{name}]")
        status = _require_str(raw.get("status"), ctx=f"detectors[{name}].status")
        if status != "scored":
            excluded.append((name, status))
            continue
        players.append(_parse_player(name, raw))

    dataset = _require_mapping(artifact.get("dataset"), ctx="dataset")
    matching_policy = _require_str(
        artifact.get("matching_policy"), ctx="matching_policy"
    )
    return AssessmentReport(
        players=tuple(players),
        excluded=tuple(excluded),
        dataset=dict(dataset),
        matching_policy=matching_policy,
        source_path=str(source),
    )


# ---------------------------------------------------------------------------
# The tournament
# ---------------------------------------------------------------------------

def run_assessment_tournament(
    report: AssessmentReport,
    *,
    engine: PIIRateEloEngine | None = None,
) -> dict[str, Any]:
    """Rate every player via per-entity-type F2 matches; return the full result.

    Deterministic: match order is field-major (sorted entity types), then
    sorted player pairs within each field. Elo updates are path-dependent, so
    this fixed order makes repeated runs byte-identical.
    """
    if len(report.players) < 2:
        raise ValueError(
            f"assessment tournament needs at least 2 scored players, "
            f"got {len(report.players)}"
        )
    rating_engine = engine if engine is not None else PIIRateEloEngine()

    fields: set[str] = set()
    for player in report.players:
        fields.update(player.per_entity_f2)
    match_fields = sorted(fields)

    by_name = {p.name: p for p in report.players}
    names = sorted(by_name)
    for field in match_fields:
        for name_i, name_j in combinations(names, 2):
            rating_engine.update_from_match(
                name_i,
                name_j,
                by_name[name_i].per_entity_f2.get(field, 0.0),
                by_name[name_j].per_entity_f2.get(field, 0.0),
            )

    players_block: dict[str, Any] = {}
    for name in names:
        player = by_name[name]
        supported = sorted(
            player.per_entity_f2.items(), key=lambda kv: (-kv[1], kv[0])
        )
        players_block[name] = {
            "precision": player.precision,
            "recall": player.recall,
            "f1": player.f1,
            "f2": player.f2,
            "f2_macro": player.f2_macro,
            "coverage": f"{player.coverage_reachable}/{player.coverage_total}",
            "n_gold": player.n_gold,
            "n_pred": player.n_pred,
            "strongest_entities": [
                {"entity_type": etype, "f2": f2} for etype, f2 in supported[:3]
            ],
            "weakest_entities": [
                {"entity_type": etype, "f2": f2}
                for etype, f2 in sorted(supported, key=lambda kv: (kv[1], kv[0]))[:3]
            ],
        }

    return {
        "schema": "pii-rate-elo-assessment-tournament/v1",
        "source": report.source_path,
        "dataset": report.dataset,
        "matching_policy": report.matching_policy,
        "match_fields": match_fields,
        "matches_per_pair": len(match_fields),
        "excluded": [
            {"name": name, "status": status} for name, status in report.excluded
        ],
        "players": players_block,
        "tournament": rating_engine.tournament_summary(),
    }


# ---------------------------------------------------------------------------
# The report renderer
# ---------------------------------------------------------------------------

def _significance_cell(
    pairwise: dict[str, Any], name_a: str, name_b: str
) -> str:
    for key in (f"{name_a}_vs_{name_b}", f"{name_b}_vs_{name_a}"):
        entry = pairwise.get(key)
        if isinstance(entry, dict):
            return "Y" if entry.get("significant") else "~"
    return "?"


def render_assessment_report(result: dict[str, Any]) -> str:
    """Render the tournament result as a markdown report (deterministic)."""
    rankings = result["tournament"]["rankings"]
    pairwise = result["tournament"]["pairwise_significance"]
    players = result["players"]
    dataset = result.get("dataset", {})

    lines: list[str] = []
    lines.append("# pii-rate-elo — assessment tournament")
    lines.append("")
    languages = dataset.get("languages", dataset.get("language", "?"))
    lines.append(
        f"Source: `{result.get('source', '?')}` · matching policy "
        f"`{result.get('matching_policy', '?')}` · split "
        f"`{dataset.get('split', '?')}` · languages `{languages}` · "
        f"records {dataset.get('n_records', '?')} · gold spans {dataset.get('n_gold', '?')}"
    )
    lines.append("")
    n_fields = result.get("matches_per_pair", 0)
    lines.append(
        f"Matches: {n_fields} gold-supported entity types x "
        f"{len(rankings)} players (all pairs) = "
        f"{result['tournament'].get('total_matches', '?')} matches."
    )
    lines.append("")
    lines.append("## Leaderboard")
    lines.append("")
    lines.append(
        "| # | System | Elo | ±RD | 95% CI | F2 (micro) | F2 (macro) | "
        "Precision | Recall | Coverage |"
    )
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for idx, row in enumerate(rankings, start=1):
        stats = players[row["name"]]
        lines.append(
            f"| {idx} | {row['name']} | {row['rating']:.2f} | {row['rd']:.2f} "
            f"| [{row['ci_lower']:.0f}, {row['ci_upper']:.0f}] "
            f"| {stats['f2']:.3f} | {stats['f2_macro']:.3f} "
            f"| {stats['precision']:.3f} | {stats['recall']:.3f} "
            f"| {stats['coverage']} |"
        )
    lines.append("")

    lines.append("## Pairwise significance")
    lines.append("")
    lines.append(
        "`Y` = rating gap exceeds 2·sqrt(RD_i² + RD_j²) (statistically "
        "distinguishable); `~` = within noise."
    )
    lines.append("")
    ordered = [row["name"] for row in rankings]
    lines.append("| vs | " + " | ".join(ordered) + " |")
    lines.append("|---|" + "---|" * len(ordered))
    for name_a in ordered:
        cells = []
        for name_b in ordered:
            cells.append(
                "—" if name_a == name_b else _significance_cell(pairwise, name_a, name_b)
            )
        lines.append(f"| {name_a} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-system strengths and blind spots")
    lines.append("")
    lines.append("| System | Strongest entity types (F2) | Weakest entity types (F2) |")
    lines.append("|---|---|---|")
    for name in ordered:
        stats = players[name]
        strong = ", ".join(
            f"{e['entity_type']} ({e['f2']:.2f})" for e in stats["strongest_entities"]
        )
        weak = ", ".join(
            f"{e['entity_type']} ({e['f2']:.2f})" for e in stats["weakest_entities"]
        )
        lines.append(f"| {name} | {strong} | {weak} |")
    lines.append("")

    excluded = result.get("excluded", [])
    if excluded:
        lines.append("## Excluded (present but not scored)")
        lines.append("")
        for entry in excluded:
            lines.append(f"- {entry['name']}: status `{entry['status']}`")
        lines.append("")

    lines.append("## Axes evaluated")
    lines.append("")
    lines.append(
        "- Detection quality (precision / recall / F1 / F2, micro + macro, "
        "per-entity): from this artifact, all systems."
    )
    lines.append(
        "- Entity-type coverage (label-map projection ceiling): from this "
        "artifact, all systems."
    )
    lines.append(
        "- Latency / throughput: NOT carried by this artifact — no rating "
        "credit or penalty assigned. (First-party systems carry measured "
        "latency in the internal benchmark artifact.)"
    )
    lines.append(
        "- Tier-3 re-identification resistance: NOT carried by this artifact "
        "— no rating credit or penalty assigned."
    )
    lines.append("")
    return "\n".join(lines)
