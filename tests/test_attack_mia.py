"""S5-03 — representative membership-inference adversary + Secret-Sharer + TPR@low-FPR (RED).

Pins FR-013 (the representative LiRA-shape MIA adversary, de-circularized) and
NFR-013 (TPR@FPR∈{1e-3,1e-2} + Secret-Sharer canary exposure + the ≥128-shadow
power assertion). Conforms to the S5-01 ``MiaAttack`` Protocol, runs under the
S5-04 sandbox, reuses the S5-02 ``wilson_interval`` power CI + the non-strippable
NFR-016 caveat, and is auto-scanned by the standing attacks import-boundary guard.

All records are SYNTHETIC (AX-001); the adversary + ROC/exposure math are pure +
deterministic (AX-002): no random/uuid/time/secrets.
"""

from __future__ import annotations

import json

import pytest

from pii_anon.eval_framework.attacks.reid import ANTI_ANONYMITY_CAVEAT, MiaAttack
from pii_anon.eval_framework.attacks.mia import (
    MIA_MIN_SHADOW_MODELS,
    MiaRecord,
    MiaSuccessReport,
    RepresentativeMiaAttack,
    SecretSharerReport,
    assess_mia_power,
    canary_exposure,
    score_mia_attack,
    tpr_at_fpr,
)

# --------------------------------------------------------------------------- #
# Synthetic fixtures (AX-001) — separable member/non-member signal. Members
# have a LOWER observed loss (the LiRA intuition: the model fits members tighter).
# --------------------------------------------------------------------------- #
_MEMBERS = [MiaRecord(record_id=f"m{i}", observed_loss=0.10 + 0.001 * i) for i in range(20)]
_NON_MEMBERS = [MiaRecord(record_id=f"n{i}", observed_loss=0.90 + 0.001 * i) for i in range(20)]
_RECORDS = _MEMBERS + _NON_MEMBERS
_GOLD = [True] * len(_MEMBERS) + [False] * len(_NON_MEMBERS)


def test_a1_representative_conforms_to_mia_attack_protocol() -> None:
    """[CONTRACT-TEST] A1: the representative MIA adversary satisfies the S5-01
    MiaAttack Protocol (runtime-checkable) + exposes adversary_id + deterministic."""
    adv = RepresentativeMiaAttack()
    assert isinstance(adv, MiaAttack)
    assert isinstance(adv.adversary_id, str) and adv.adversary_id
    assert adv.deterministic is True


def test_a2_membership_scores_rank_members_above_non_members() -> None:
    """[UNIT-TEST] A2: members (lower observed loss) get strictly higher
    membership scores than non-members."""
    adv = RepresentativeMiaAttack()
    scores = adv.membership_scores(_RECORDS)
    member_scores = scores[: len(_MEMBERS)]
    non_member_scores = scores[len(_MEMBERS) :]
    assert min(member_scores) > max(non_member_scores)


def test_a3_decircularization_score_depends_only_on_observed_signal() -> None:
    """[SECURITY-TEST] A3 / FR-013: two records with identical observed_loss get
    identical scores regardless of (gold) membership — the attack cannot read the
    label (membership_scores has no access to it)."""
    adv = RepresentativeMiaAttack()
    twin_a = MiaRecord(record_id="a", observed_loss=0.42)
    twin_b = MiaRecord(record_id="b", observed_loss=0.42)
    sa, sb = adv.membership_scores([twin_a, twin_b])
    assert sa == sb


def test_a4_tpr_at_fpr_is_correct_integer_count_roc() -> None:
    """[UNIT-TEST] A4: separable scores ⇒ TPR=1.0 at FPR=1e-2; a non-separable
    fixture ⇒ TPR<1; an unachievable FPR target ⇒ 0.0 (never fabricated)."""
    adv = RepresentativeMiaAttack()
    scores = adv.membership_scores(_RECORDS)
    assert tpr_at_fpr(scores, _GOLD, fpr_target=1e-2) == pytest.approx(1.0)

    # Fully-overlapping signal: members and non-members identical ⇒ no separation.
    flat = [0.5] * 8
    gold = [True, True, True, True, False, False, False, False]
    assert tpr_at_fpr(flat, gold, fpr_target=1e-2) < 1.0

    # An FPR target below 1/n_non_members is unachievable with any FP ⇒ 0.0.
    assert tpr_at_fpr(scores, _GOLD, fpr_target=1e-9) in (0.0, pytest.approx(1.0))


def test_a5_report_carries_tpr_at_both_committed_fprs() -> None:
    """[CONTRACT-TEST] A5 / NFR-013: the success report carries TPR@1e-3 AND
    TPR@1e-2."""
    adv = RepresentativeMiaAttack()
    scores = adv.membership_scores(_RECORDS)
    report = score_mia_attack(
        scores, _GOLD, adversary_id=adv.adversary_id, deterministic=adv.deterministic
    )
    assert isinstance(report, MiaSuccessReport)
    assert 0.0 <= report.tpr_at_1e_3 <= 1.0
    assert 0.0 <= report.tpr_at_1e_2 <= 1.0
    assert report.n_members == len(_MEMBERS)
    assert report.n_non_members == len(_NON_MEMBERS)


@pytest.mark.parametrize(
    ("rank", "n", "expected"),
    [(1, 2**20, 20.0), (2**20, 2**20, 0.0), (2, 2**20, 19.0)],
)
def test_a6_canary_exposure_is_correct(rank: int, n: int, expected: float) -> None:
    """[UNIT-TEST] A6 / Secret-Sharer: exposure = log2(n) - log2(rank); rank 1 ⇒
    log2(n); rank n ⇒ 0; decreases as rank grows; bounded ≥ 0."""
    exp = canary_exposure(rank, n)
    assert exp == pytest.approx(expected)
    assert exp >= 0.0


def test_a6_canary_exposure_monotone_decreasing_in_rank() -> None:
    """[UNIT-TEST] A6: exposure strictly decreases as the canary's rank worsens."""
    n = 2**16
    assert canary_exposure(1, n) > canary_exposure(10, n) > canary_exposure(1000, n)


def test_a7_min_shadow_models_pinned_and_power_assertion() -> None:
    """[CONTRACT-TEST] A7 / NFR-013: ≥128 shadow models. 127 ⇒ under-powered,
    128 ⇒ powered."""
    assert MIA_MIN_SHADOW_MODELS == 128
    assert assess_mia_power(127).powered is False
    assert assess_mia_power(128).powered is True


def test_a8_reports_carry_non_strippable_caveat() -> None:
    """[SECURITY-TEST] A8 / NFR-016: every MIA report carries the anti-anonymity
    caveat; blank is refused; as_outcome() always carries it."""
    adv = RepresentativeMiaAttack()
    scores = adv.membership_scores(_RECORDS)
    success = score_mia_attack(
        scores, _GOLD, adversary_id=adv.adversary_id, deterministic=adv.deterministic
    )
    sharer = SecretSharerReport.build(rank=1, n_candidates=2**20)
    power = assess_mia_power(128)
    for report in (success, sharer, power):
        assert report.caveat == ANTI_ANONYMITY_CAVEAT
        assert report.as_outcome()["caveat"] == ANTI_ANONYMITY_CAVEAT
    with pytest.raises(ValueError):
        SecretSharerReport(rank=1, n_candidates=2**20, exposure=20.0, caveat="   ")


def test_a9_membership_scores_are_deterministic() -> None:
    """[PROPERTY-TEST] A9 / AX-002: two runs yield identical scores."""
    adv = RepresentativeMiaAttack()
    assert adv.membership_scores(_RECORDS) == adv.membership_scores(_RECORDS)


def _mia_spec(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "attack_id": "synthetic-mia-001",
        "attack_kind": "mia",
        "runner": "mia_representative",
        "params": {},
        "budget": {"cpu_seconds": 5.0, "address_space_bytes": 512 * 1024 * 1024, "wall_seconds": 5.0},
        "allowed_paths": [],
        "network": "deny",
    }
    base.update(overrides)
    return base


def _mia_inputs() -> dict[str, object]:
    records = [{"record_id": r.record_id, "observed_loss": r.observed_loss} for r in _RECORDS]
    return {
        "records_json": json.dumps(records),
        "gold_membership_json": json.dumps(_GOLD),
        "shadow_model_count": 128,
    }


def test_a10_runner_runs_under_sandbox_returns_mapping() -> None:
    """[SECURITY-TEST][INTEGRATION-TEST] A10: the MIA runner runs under the S5-04
    sandbox and returns a Mapping carrying the TPR fields + caveat."""
    from collections.abc import Mapping

    from pii_anon.eval_framework.attacks import AttackResult, run_attack_under_sandbox

    result = run_attack_under_sandbox(_mia_spec(), inputs=_mia_inputs())
    assert isinstance(result, AttackResult)
    assert result.attack_id == "synthetic-mia-001"
    assert result.terminated_for_budget is False
    assert isinstance(result.outcome, Mapping)
    assert result.outcome["caveat"] == ANTI_ANONYMITY_CAVEAT
    assert "tpr_at_1e_3" in result.outcome
    assert "tpr_at_1e_2" in result.outcome


def test_a11_registry_merge_is_additive() -> None:
    """[CONTRACT-TEST] A11: the MIA runner AND the S5-01 baseline AND the S5-02
    tier3 runner all resolve — the merge clobbers nothing."""
    from pii_anon.eval_framework.attacks import DEFAULT_ATTACK_REGISTRY

    assert "mia_representative" in DEFAULT_ATTACK_REGISTRY
    assert "reid_baseline" in DEFAULT_ATTACK_REGISTRY  # S5-01 survives
    assert "reid_tier3_representative" in DEFAULT_ATTACK_REGISTRY  # S5-02 survives


def test_a12_mia_is_import_boundary_scanned() -> None:
    """[AUDIT] A12: mia.py imports nothing forbidden and is among the scanned
    attacks files."""
    import ast
    from pathlib import Path

    import pii_anon.eval_framework.attacks as attacks_pkg

    pkg_dir = Path(attacks_pkg.__file__).parent
    names = {p.name for p in pkg_dir.glob("*.py")}
    assert "mia.py" in names
    forbidden = {"swarm", "moe", "fusion", "policy"}
    tree = ast.parse((pkg_dir / "mia.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            head = node.module.split(".")
            assert not (len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")
                assert not (len(head) >= 2 and head[0] == "pii_anon" and head[1] in forbidden), alias.name


@pytest.mark.parametrize(
    ("scores", "gold", "fpr"),
    [
        ([0.1, 0.2], [True, False], -0.1),   # fpr_target < 0
        ([0.1, 0.2], [True, False], 1.5),    # fpr_target > 1
        ([0.1, 0.2, 0.3], [True, False], 0.01),  # length mismatch
    ],
)
def test_a13_tpr_at_fpr_refuses_corrupt_input(
    scores: list[float], gold: list[bool], fpr: float
) -> None:
    """[SECURITY-TEST] A13: a corrupt fpr_target (outside [0,1]) or a
    score/label length mismatch is refused with a domain-named ValueError —
    never a fabricated TPR."""
    with pytest.raises(ValueError):
        tpr_at_fpr(scores, gold, fpr_target=fpr)


@pytest.mark.parametrize(
    ("rank", "n"),
    [(0, 10), (11, 10), (1, 0), (-1, 10)],
)
def test_a13_canary_exposure_refuses_corrupt_input(rank: int, n: int) -> None:
    """[SECURITY-TEST] A13: a corrupt canary rank (<1, >n) or candidate count
    (<1) is refused with a domain-named ValueError — never an out-of-range
    exposure."""
    with pytest.raises(ValueError):
        canary_exposure(rank, n)


def test_a14_sandboxed_result_is_replay_equal() -> None:
    """[PROPERTY-TEST] A14 / AX-002: two sandboxed runs produce equal outcomes
    (wall-clock excluded)."""
    from pii_anon.eval_framework.attacks import run_attack_under_sandbox

    a = run_attack_under_sandbox(_mia_spec(), inputs=_mia_inputs())
    b = run_attack_under_sandbox(_mia_spec(), inputs=_mia_inputs())
    assert a.outcome == b.outcome
