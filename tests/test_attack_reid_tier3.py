"""S5-02 — Tier-3 representative re-identification adversary + RRS power (RED).

Pins FR-011 (the representative Tier-3 adversary — QIC+BSL fused, de-circularized
via margin-commit), FR-012 (the circularity/contamination controls), and NFR-012
(Wilson CIs + the 2-rung ≥385/≥897 paired-persona power ladder). The adversary
conforms to the S5-01 ``ReidAttack`` Protocol, runs under the S5-04 sandbox,
reuses the S5-01 scorer + the non-strippable NFR-016 caveat, and is auto-scanned
by the standing attacks import-boundary CI guard.

All personas/targets are SYNTHETIC (AX-001). The adversary + Wilson math are pure
+ deterministic (AX-002): no random/uuid/time/secrets.
"""

from __future__ import annotations

import json

import pytest

from pii_anon.eval_framework.attacks.reid import (
    ANTI_ANONYMITY_CAVEAT,
    BaselineDeterministicReidAttack,
    ReidAttack,
    ReidPersona,
    ReidTarget,
    score_reid_attack,
)
from pii_anon.eval_framework.attacks.reid_tier3 import (
    RRS_RUNG_REID_HIGH,
    RRS_RUNG_REID_LOW,
    ReidPowerCell,
    ReidPowerLadder,
    RepresentativeTier3ReidAttack,
    assess_rrs_power,
    tier3_reid_attack_runner,
    wilson_interval,
)

# --------------------------------------------------------------------------- #
# Synthetic fixtures (AX-001) — a record where the correct link surfaces ONLY
# through the combined QIC + BSL signal (the baseline mis-links the decoy).
# --------------------------------------------------------------------------- #
# Target P1: signals {nurse, portland, 1985}; the decoy P2 shares a HIGHER raw
# quasi-identifier overlap (nurse+portland) so plain weighted-Jaccard picks P2,
# but P1's auxiliary_knowledge (city=portland, year=1985) lifts the fused score.
_TARGET = ReidTarget(
    target_id="P1",  # gold convention: target_id == true persona_id
    anonymized_text="[REDACTED] works as a nurse.",
    observed_signals=("nurse", "portland", "1985"),
)
_CORRECT = ReidPersona(
    persona_id="P1",
    quasi_identifiers=("nurse",),
    auxiliary_knowledge={"city": "portland", "birth_year": "1985"},
)
_DECOY = ReidPersona(
    persona_id="P2",
    quasi_identifiers=("nurse", "portland"),
    auxiliary_knowledge={},
)


def test_a1_representative_conforms_to_reid_attack_protocol() -> None:
    """[CONTRACT-TEST] A1: the representative adversary satisfies the S5-01
    ReidAttack Protocol (runtime-checkable) + exposes adversary_id + deterministic."""
    adv = RepresentativeTier3ReidAttack()
    assert isinstance(adv, ReidAttack)
    assert isinstance(adv.adversary_id, str) and adv.adversary_id
    assert adv.deterministic is True


def test_a2_representative_stronger_than_baseline_via_qic_bsl() -> None:
    """[UNIT-TEST] A2: on a record linkable only via the combined QIC+BSL signal,
    the representative adversary links the CORRECT persona where the baseline
    mis-links the higher-raw-overlap decoy — reid_recall_rep > reid_recall_base."""
    targets = [_TARGET]
    candidates = [_CORRECT, _DECOY]

    base = BaselineDeterministicReidAttack()
    rep = RepresentativeTier3ReidAttack()
    base_guesses = base.attack(targets, candidates, candidate_set_size=2)
    rep_guesses = rep.attack(targets, candidates, candidate_set_size=2)

    base_metrics = score_reid_attack(
        base_guesses, targets, adversary_id=base.adversary_id,
        deterministic=base.deterministic, candidate_set_size=2,
    )
    rep_metrics = score_reid_attack(
        rep_guesses, targets, adversary_id=rep.adversary_id,
        deterministic=rep.deterministic, candidate_set_size=2,
    )
    # The representative links P1 (correct); the baseline links P2 (the decoy).
    assert rep_guesses[0].guessed_persona_id == "P1"
    assert base_guesses[0].guessed_persona_id == "P2"
    assert rep_metrics.reid_recall > base_metrics.reid_recall


def test_a3_margin_commit_abstains_on_ambiguous_link() -> None:
    """[UNIT-TEST] A3 / FR-012: two candidates within the commit margin ⇒ the
    adversary ABSTAINS rather than guess (precision-preserving de-circularization)."""
    target = ReidTarget(
        target_id="X", anonymized_text="…", observed_signals=("alpha", "beta")
    )
    # Two personas with near-identical fused score (both share exactly one token).
    twin_a = ReidPersona(persona_id="A", quasi_identifiers=("alpha",), auxiliary_knowledge={})
    twin_b = ReidPersona(persona_id="B", quasi_identifiers=("beta",), auxiliary_knowledge={})
    guesses = RepresentativeTier3ReidAttack().attack(
        [target], [twin_a, twin_b], candidate_set_size=2
    )
    assert guesses[0].guessed_persona_id is None  # abstain on the tie


def test_a4_commits_on_clear_margin() -> None:
    """[UNIT-TEST] A4: a top candidate clearing the margin is committed."""
    target = ReidTarget(
        target_id="A", anonymized_text="…",
        observed_signals=("alpha", "beta", "gamma"),
    )
    strong = ReidPersona(
        persona_id="A", quasi_identifiers=("alpha", "beta", "gamma"),
        auxiliary_knowledge={},
    )
    weak = ReidPersona(persona_id="B", quasi_identifiers=("delta",), auxiliary_knowledge={})
    guesses = RepresentativeTier3ReidAttack().attack(
        [target], [strong, weak], candidate_set_size=2
    )
    assert guesses[0].guessed_persona_id == "A"


def test_a5_ranking_is_deterministic_and_order_invariant() -> None:
    """[PROPERTY-TEST] A5 / AX-002: two runs + a candidate permutation yield
    identical guesses (a total order); ties resolve persona_id ASC."""
    adv = RepresentativeTier3ReidAttack()
    g1 = adv.attack([_TARGET], [_CORRECT, _DECOY], candidate_set_size=2)
    g2 = adv.attack([_TARGET], [_DECOY, _CORRECT], candidate_set_size=2)  # permuted
    assert g1 == g2


@pytest.mark.parametrize(
    ("successes", "n"),
    [(0, 0), (0, 10), (10, 10), (5, 10), (1, 385), (384, 385), (897, 897)],
)
def test_a6_wilson_interval_is_bounded_and_brackets_rate(
    successes: int, n: int
) -> None:
    """[UNIT-TEST] A6: the Wilson interval is within [0,1] and brackets the point
    estimate; n=0 ⇒ (0.0, 0.0)."""
    low, high = wilson_interval(successes, n)
    assert 0.0 <= low <= high <= 1.0
    if n == 0:
        assert (low, high) == (0.0, 0.0)
    else:
        rate = successes / n
        assert low <= rate <= high


def test_a6_wilson_interval_known_symmetry() -> None:
    """[UNIT-TEST] A6: the Wilson interval is symmetric under successes↔failures
    (the interval for k/n mirrors the interval for (n-k)/n about 0.5)."""
    low_k, high_k = wilson_interval(3, 10)
    low_c, high_c = wilson_interval(7, 10)
    assert low_k == pytest.approx(1.0 - high_c, abs=1e-9)
    assert high_k == pytest.approx(1.0 - low_c, abs=1e-9)


def test_a7_two_rung_ladder_constants_are_pinned() -> None:
    """[CONTRACT-TEST] A7 / NFR-012: the 2-rung RRS power ladder literals."""
    assert RRS_RUNG_REID_LOW == 385
    assert RRS_RUNG_REID_HIGH == 897


@pytest.mark.parametrize(
    ("n", "rung", "expected"),
    [
        (384, "REID_LOW", False),
        (385, "REID_LOW", True),
        (896, "REID_HIGH", False),
        (897, "REID_HIGH", True),
    ],
)
def test_a8_powered_reflects_the_rung_threshold(
    n: int, rung: str, expected: bool
) -> None:
    """[UNIT-TEST] A8: a cell is powered iff n_paired_personas ≥ the rung min."""
    cell = ReidPowerCell.build(
        cell_id="cell-1", n_paired_personas=n, successes=min(n, 1), rung=rung
    )
    assert cell.powered is expected
    assert cell.min_required == (RRS_RUNG_REID_LOW if rung == "REID_LOW" else RRS_RUNG_REID_HIGH)


def test_a9_power_report_carries_non_strippable_caveat() -> None:
    """[SECURITY-TEST] A9 / NFR-016: every power cell carries the anti-anonymity
    caveat; a blank caveat is refused; as_outcome() always carries it."""
    cell = ReidPowerCell.build(
        cell_id="c", n_paired_personas=385, successes=40, rung="REID_LOW"
    )
    assert cell.caveat == ANTI_ANONYMITY_CAVEAT
    assert cell.as_outcome()["caveat"] == ANTI_ANONYMITY_CAVEAT
    with pytest.raises(ValueError):
        ReidPowerCell(
            cell_id="c", n_paired_personas=385, successes=40,
            success_rate=40 / 385, ci_low=0.0, ci_high=1.0,
            rung="REID_LOW", min_required=385, powered=True, caveat="   ",
        )


def test_a9_assess_rrs_power_aggregates_all_powered() -> None:
    """[UNIT-TEST] A9: assess_rrs_power yields per-cell verdicts + the aggregate
    all_powered (False iff any cell is under-powered)."""
    cells = [
        ReidPowerCell.build(cell_id="c1", n_paired_personas=385, successes=40, rung="REID_LOW"),
        ReidPowerCell.build(cell_id="c2", n_paired_personas=900, successes=90, rung="REID_HIGH"),
    ]
    ladder = assess_rrs_power(cells)
    assert isinstance(ladder, ReidPowerLadder)
    assert ladder.all_powered is True
    under = assess_rrs_power(
        [ReidPowerCell.build(cell_id="c3", n_paired_personas=100, successes=10, rung="REID_LOW")]
    )
    assert under.all_powered is False
    assert under.as_outcome()["caveat"] == ANTI_ANONYMITY_CAVEAT


def _tier3_spec(**overrides: object) -> dict[str, object]:
    """A synthetic AttackSpec selecting the tier3 representative runner."""
    base: dict[str, object] = {
        "attack_id": "synthetic-tier3-reid-001",
        "attack_kind": "reid",
        "runner": "reid_tier3_representative",
        "params": {},
        "budget": {"cpu_seconds": 5.0, "address_space_bytes": 512 * 1024 * 1024, "wall_seconds": 5.0},
        "allowed_paths": [],
        "network": "deny",
    }
    base.update(overrides)
    return base


def _tier3_inputs() -> dict[str, object]:
    """Synthetic scalar JSON-string args for the tier3 runner (AX-001)."""
    targets = [
        {"target_id": "P1", "anonymized_text": "x", "observed_signals": ["nurse", "portland", "1985"]}
    ]
    candidates = [
        {"persona_id": "P1", "quasi_identifiers": ["nurse"],
         "auxiliary_knowledge": {"city": "portland", "birth_year": "1985"}},
        {"persona_id": "P2", "quasi_identifiers": ["nurse", "portland"]},
    ]
    return {
        "targets_json": json.dumps(targets),
        "candidates_json": json.dumps(candidates),
        "candidate_set_size": 2,
    }


def test_a10_runner_runs_under_sandbox_returns_mapping() -> None:
    """[SECURITY-TEST][INTEGRATION-TEST] A10: the tier3 runner runs under the
    S5-04 sandbox and returns a Mapping carrying the metric fields + caveat."""
    from collections.abc import Mapping

    from pii_anon.eval_framework.attacks import AttackResult, run_attack_under_sandbox

    result = run_attack_under_sandbox(_tier3_spec(), inputs=_tier3_inputs())
    assert isinstance(result, AttackResult)
    assert result.attack_id == "synthetic-tier3-reid-001"
    assert result.terminated_for_budget is False
    assert isinstance(result.outcome, Mapping)
    assert result.outcome["caveat"] == ANTI_ANONYMITY_CAVEAT
    assert "reid_recall" in result.outcome


def test_a11_registry_merge_is_additive() -> None:
    """[CONTRACT-TEST] A11: both the tier3 runner AND the S5-01 baseline resolve
    in the sandbox default registry — the merge clobbers nothing."""
    from pii_anon.eval_framework.attacks import DEFAULT_ATTACK_REGISTRY

    assert "reid_tier3_representative" in DEFAULT_ATTACK_REGISTRY
    assert "reid_baseline" in DEFAULT_ATTACK_REGISTRY  # S5-01 survives


def test_a12_reid_tier3_is_import_boundary_scanned() -> None:
    """[AUDIT] A12: the new module imports nothing forbidden and is among the
    scanned attacks files (auto-covered by the *.py glob; assert it explicitly)."""
    import ast
    from pathlib import Path

    import pii_anon.eval_framework.attacks as attacks_pkg

    pkg_dir = Path(attacks_pkg.__file__).parent
    names = {p.name for p in pkg_dir.glob("*.py")}
    assert "reid_tier3.py" in names
    forbidden = {"swarm", "moe", "fusion", "policy"}
    tree = ast.parse((pkg_dir / "reid_tier3.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")
            assert not (len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")
                assert not (len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden), alias.name


def test_a13_decircularization_follows_signal_not_gold_link() -> None:
    """[SECURITY-TEST] A13 / FR-012: the adversary cannot consult the gold link —
    given a target whose gold persona (target_id) CONTRADICTS the signal, it
    follows the SIGNAL (guessing the signal-matching persona, scored as wrong)."""
    # Gold says this target IS "GOLD", but every signal points at "SIGNAL".
    target = ReidTarget(
        target_id="GOLD", anonymized_text="…",
        observed_signals=("zeta", "eta", "theta"),
    )
    gold_persona = ReidPersona(persona_id="GOLD", quasi_identifiers=("unrelated",), auxiliary_knowledge={})
    signal_persona = ReidPersona(
        persona_id="SIGNAL", quasi_identifiers=("zeta", "eta", "theta"), auxiliary_knowledge={}
    )
    guesses = RepresentativeTier3ReidAttack().attack(
        [target], [gold_persona, signal_persona], candidate_set_size=2
    )
    # It follows the signal (SIGNAL), proving it never read the gold link.
    assert guesses[0].guessed_persona_id == "SIGNAL"
    metrics = score_reid_attack(
        guesses, [target], adversary_id="x", deterministic=True, candidate_set_size=2
    )
    assert metrics.correct == 0  # honest: following signal ≠ cheating to the gold


def test_a14_sandboxed_result_is_replay_equal() -> None:
    """[PROPERTY-TEST] A14 / AX-002: two sandboxed runs produce equal outcomes
    (wall-clock excluded)."""
    from pii_anon.eval_framework.attacks import run_attack_under_sandbox

    a = run_attack_under_sandbox(_tier3_spec(), inputs=_tier3_inputs())
    b = run_attack_under_sandbox(_tier3_spec(), inputs=_tier3_inputs())
    assert a.outcome == b.outcome
