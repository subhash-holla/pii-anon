"""S5-01 — ``ReidAttack`` protocol + deterministic baseline re-identification body.

DC-09 ``attacks/`` package — this story adds the re-identification attack
protocol (FR-011/FR-013 foundation) + a deterministic baseline adversary body
that runs ONLY inside the S5-04 sandbox, plus the non-strippable anti-anonymity
caveat (NFR-016 — a MUST). It shape-mirrors the DATA sibling's
``Adversary``/``Persona``/``Target``/``Guess`` contract so the Pass-2 swap to the
real offline adversary is structural.

Acceptance (story §3):
- A1  — ``ReidAttack`` Protocol is ``@runtime_checkable``.                [CONTRACT]
- A2  — baseline links an exact surviving-quasi-identifier match.          [UNIT]
- A3  — baseline abstains (``guessed_persona_id=None``) on no signal.      [UNIT]
- A4  — deterministic total-order ranking (replay + permutation equal).  [PROPERTY]
- A5  — success metrics are integer-count-based.                          [UNIT]
- A6  — non-strippable caveat (NFR-016): cannot be cleared; in outcome.   [SECURITY]
- A7  — runner runs under ``run_attack_under_sandbox``; outcome a Mapping.[SECURITY]
- A8  — ``reid_attack_runner(...)`` returns a ``Mapping[str, Any]``.       [CONTRACT]
- A9  — two sandboxed runs produce equal ``AttackResult`` (wall-clock excl).[PROPERTY]
- A10 — attacks package imports nothing forbidden  (see
        tests/test_attacks_import_boundary.py).                            [AUDIT]
- A11 — no dangerous call signatures in attacks/  (see
        tests/test_attack_sandbox.py + the extra reid scan here).          [AUDIT]

AX-001: every persona/target here is SYNTHETIC — no real PII, no real key, no
real network target. AX-002: the baseline ranking is a total order; no
``random``/``uuid``/``time``/``secrets`` are imported by the body (an AST guard
pins that). Dangerous patterns are referenced via the reworded forms
"unsafe-deserialization" / "subprocess" / "shell-out" only.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from pii_anon.eval_framework.attacks import (
    DEFAULT_ATTACK_REGISTRY,
    AttackResult,
    run_attack_under_sandbox,
)
from pii_anon.eval_framework.attacks.reid import (
    ANTI_ANONYMITY_CAVEAT,
    REID_ATTACK_REGISTRY,
    BaselineDeterministicReidAttack,
    MiaAttack,
    ReidAttack,
    ReidGuess,
    ReidPersona,
    ReidSuccessMetrics,
    ReidTarget,
    reid_attack_runner,
    score_reid_attack,
)

_REID_RUNNER = "pii_anon.eval_framework.attacks.reid.reid_attack_runner"


# --------------------------------------------------------------------------- #
# Synthetic fixtures (AX-001) — no real PII; tiny closed-world.               #
# --------------------------------------------------------------------------- #
def _persona(pid: str, qis: tuple[str, ...], source: str = "") -> ReidPersona:
    return ReidPersona(
        persona_id=pid,
        quasi_identifiers=qis,
        auxiliary_knowledge={"note": f"synthetic-{pid}"},
        source_text=source or f"synthetic source for {pid}",
    )


def _target(tid: str, anon: str, signals: tuple[str, ...]) -> ReidTarget:
    return ReidTarget(target_id=tid, anonymized_text=anon, observed_signals=signals)


def _three_personas() -> list[ReidPersona]:
    return [
        _persona("p_alpha", ("cardiology", "boston", "marathon")),
        _persona("p_beta", ("oncology", "denver", "cycling")),
        _persona("p_gamma", ("neurology", "austin", "swimming")),
    ]


# --------------------------------------------------------------------------- #
# A1 — ReidAttack Protocol is runtime-checkable.                              #
# --------------------------------------------------------------------------- #
def test_fr011_a1_reid_attack_protocol_is_runtime_checkable() -> None:
    """[CONTRACT-TEST] A1 — isinstance(baseline, ReidAttack) is True and the
    Protocol is @runtime_checkable (the FR-011/013 protocol foundation)."""
    baseline = BaselineDeterministicReidAttack()
    assert isinstance(baseline, ReidAttack)
    # @runtime_checkable Protocols expose this marker.
    assert getattr(ReidAttack, "_is_runtime_protocol", False) is True


def test_fr011_a1_baseline_exposes_required_protocol_attributes() -> None:
    """A1 — the baseline carries the version-pinned adversary_id + deterministic
    flag the Protocol requires."""
    baseline = BaselineDeterministicReidAttack()
    assert isinstance(baseline.adversary_id, str) and baseline.adversary_id
    assert baseline.deterministic is True
    assert callable(baseline.attack)


def test_fr013_a1_mia_attack_protocol_seam_is_runtime_checkable() -> None:
    """[CONTRACT-TEST] A1 — the MiaAttack Protocol seam (consumed by S5-03) is
    declared here so the package boundary + the protocol family land together;
    it too is @runtime_checkable. The baseline reid adversary does NOT implement
    membership_scores, so it is NOT a MiaAttack."""
    assert getattr(MiaAttack, "_is_runtime_protocol", False) is True
    assert not isinstance(BaselineDeterministicReidAttack(), MiaAttack)


# --------------------------------------------------------------------------- #
# A2 — baseline links an exact surviving-quasi-identifier match.              #
# --------------------------------------------------------------------------- #
def test_fr011_a2_baseline_links_exact_quasi_identifier_match() -> None:
    """[UNIT-TEST] A2 — a target whose surviving QI signals uniquely match one
    persona is guessed as that persona (highest weighted-Jaccard)."""
    personas = _three_personas()
    # surviving signals uniquely overlap p_beta's quasi-identifiers.
    target = _target("t1", "redacted ... oncology ... denver", ("oncology", "denver"))
    guesses = BaselineDeterministicReidAttack().attack([target], personas, candidate_set_size=3)
    assert len(guesses) == 1
    g = guesses[0]
    assert isinstance(g, ReidGuess)
    assert g.target_id == "t1"
    assert g.guessed_persona_id == "p_beta"
    assert g.score > 0.0


def test_fr011_a2_baseline_prefers_higher_overlap_persona() -> None:
    """A2 — among several candidates, the persona with the higher weighted-Jaccard
    overlap wins (more shared surviving signals)."""
    personas = _three_personas()
    # two overlapping tokens with alpha, one with beta -> alpha wins.
    target = _target("t2", "...", ("cardiology", "boston", "denver"))
    [g] = BaselineDeterministicReidAttack().attack([target], personas, candidate_set_size=3)
    assert g.guessed_persona_id == "p_alpha"


# --------------------------------------------------------------------------- #
# A3 — baseline abstains on no signal.                                        #
# --------------------------------------------------------------------------- #
def test_fr011_a3_baseline_abstains_on_no_surviving_signal() -> None:
    """[UNIT-TEST] A3 — a fully-anonymized target (no surviving QI/observed
    signal overlapping any candidate) yields guessed_persona_id=None (abstain;
    excluded from the precision denominator)."""
    personas = _three_personas()
    target = _target("t_clean", "[REDACTED] [REDACTED] [REDACTED]", ())
    [g] = BaselineDeterministicReidAttack().attack([target], personas, candidate_set_size=3)
    assert g.target_id == "t_clean"
    assert g.guessed_persona_id is None
    assert g.score == 0.0


def test_fr011_a3_baseline_abstains_when_signal_matches_no_candidate() -> None:
    """A3 — observed signals that overlap NO candidate persona also abstain
    (zero similarity to every candidate)."""
    personas = _three_personas()
    target = _target("t_novel", "...", ("astrophysics", "reykjavik"))
    [g] = BaselineDeterministicReidAttack().attack([target], personas, candidate_set_size=3)
    assert g.guessed_persona_id is None


# --------------------------------------------------------------------------- #
# A4 — deterministic total-order ranking (replay + permutation).             #
# --------------------------------------------------------------------------- #
def test_ax002_a4_ranking_is_deterministic_across_runs() -> None:
    """[PROPERTY-TEST] A4 — two runs over identical inputs yield identical
    guesses (AX-002 determinism)."""
    personas = _three_personas()
    targets = [
        _target("t1", "...", ("oncology", "denver")),
        _target("t2", "...", ("cardiology", "boston")),
    ]
    a = BaselineDeterministicReidAttack()
    r1 = a.attack(targets, personas, candidate_set_size=3)
    r2 = a.attack(targets, personas, candidate_set_size=3)
    assert r1 == r2


def test_ax002_a4_candidate_permutation_yields_identical_guesses() -> None:
    """[PROPERTY-TEST] A4 — permuting the candidate order does NOT change the
    guesses: the ranking is a TOTAL order (similarity desc, persona_id asc), so
    candidate input order is irrelevant."""
    personas = _three_personas()
    target = _target("t1", "...", ("oncology", "denver"))
    forward = BaselineDeterministicReidAttack().attack([target], personas, candidate_set_size=3)
    reversed_ = BaselineDeterministicReidAttack().attack(
        [target], list(reversed(personas)), candidate_set_size=3
    )
    assert forward == reversed_


def test_ax002_a4_ties_resolve_by_persona_id_ascending() -> None:
    """[PROPERTY-TEST] A4 — when two candidates have equal similarity, the tie is
    broken by persona_id ascending (a deterministic total order). Two personas
    constructed with identical quasi-identifiers but different ids: the
    lexicographically-smaller id wins, regardless of input order."""
    p_low = _persona("p_aaa", ("oncology", "denver"))
    p_high = _persona("p_zzz", ("oncology", "denver"))
    target = _target("t1", "...", ("oncology", "denver"))
    [g1] = BaselineDeterministicReidAttack().attack([target], [p_high, p_low], candidate_set_size=2)
    [g2] = BaselineDeterministicReidAttack().attack([target], [p_low, p_high], candidate_set_size=2)
    assert g1.guessed_persona_id == "p_aaa"
    assert g2.guessed_persona_id == "p_aaa"


# --------------------------------------------------------------------------- #
# A5 — success metrics are integer-count-based.                              #
# --------------------------------------------------------------------------- #
def test_nfr016_a5_score_reid_attack_uses_integer_counts() -> None:
    """[UNIT-TEST] A5 — score_reid_attack returns reid_recall=correct/n_targets,
    reid_precision=correct/n_guesses, reid_success_rate=recall*precision, with
    correct/n_targets/n_guesses integers."""
    personas = _three_personas()
    targets = [
        _target("p_alpha", "...", ("cardiology", "boston")),  # true link == p_alpha
        _target("p_beta", "...", ("oncology", "denver")),  # true link == p_beta
        _target("p_gamma", "...", ()),  # abstain (no signal)
    ]
    adv = BaselineDeterministicReidAttack()
    guesses = adv.attack(targets, personas, candidate_set_size=3)
    metrics = score_reid_attack(
        guesses,
        targets,
        adversary_id=adv.adversary_id,
        deterministic=adv.deterministic,
        candidate_set_size=3,
    )
    assert isinstance(metrics, ReidSuccessMetrics)
    assert isinstance(metrics.correct, int)
    assert isinstance(metrics.n_targets, int)
    assert isinstance(metrics.n_guesses, int)
    assert metrics.n_targets == 3
    # two committed correct guesses, one abstain (excluded from precision denom).
    assert metrics.correct == 2
    assert metrics.n_guesses == 2
    assert metrics.reid_recall == pytest.approx(2 / 3)
    assert metrics.reid_precision == pytest.approx(2 / 2)
    assert metrics.reid_success_rate == pytest.approx((2 / 3) * (2 / 2))


def test_nfr016_a5_zero_guesses_does_not_divide_by_zero() -> None:
    """A5 — all-abstain: n_guesses=0 must not raise; precision degrades to 0.0
    (no committed guess), recall 0.0."""
    personas = _three_personas()
    targets = [_target("p_alpha", "[REDACTED]", ()), _target("p_beta", "[REDACTED]", ())]
    adv = BaselineDeterministicReidAttack()
    guesses = adv.attack(targets, personas, candidate_set_size=3)
    metrics = score_reid_attack(
        guesses, targets, adversary_id=adv.adversary_id, deterministic=True, candidate_set_size=3
    )
    assert metrics.correct == 0
    assert metrics.n_guesses == 0
    assert metrics.reid_recall == 0.0
    assert metrics.reid_precision == 0.0
    assert metrics.reid_success_rate == 0.0


# --------------------------------------------------------------------------- #
# A6 — non-strippable caveat (NFR-016).                                       #
# --------------------------------------------------------------------------- #
def test_nfr016_a6_metrics_carry_anti_anonymity_caveat_by_default() -> None:
    """[CONTRACT-TEST] A6 — ReidSuccessMetrics defaults caveat to the canonical
    ANTI_ANONYMITY_CAVEAT (NFR-016: 100% of exported privacy artifacts carry the
    anti-anonymity caveat)."""
    m = ReidSuccessMetrics(
        reid_recall=0.5,
        reid_precision=0.5,
        reid_success_rate=0.25,
        correct=1,
        n_targets=2,
        n_guesses=2,
        candidate_set_size=2,
        adversary_id="adv@v1",
        deterministic=True,
    )
    assert m.caveat == ANTI_ANONYMITY_CAVEAT
    assert ANTI_ANONYMITY_CAVEAT  # non-empty canonical text


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_nfr016_a6_constructing_with_blank_caveat_is_refused(blank: str) -> None:
    """[SECURITY-TEST][CONTRACT-TEST] A6 — the caveat CANNOT be cleared: building
    ReidSuccessMetrics with an empty/whitespace caveat raises (__post_init__
    re-asserts the non-strippable caveat; NFR-016)."""
    with pytest.raises((ValueError, TypeError)):
        ReidSuccessMetrics(
            reid_recall=0.0,
            reid_precision=0.0,
            reid_success_rate=0.0,
            correct=0,
            n_targets=1,
            n_guesses=0,
            candidate_set_size=1,
            adversary_id="adv@v1",
            deterministic=True,
            caveat=blank,
        )


def test_nfr016_a6_caveat_cannot_be_stripped_post_construction() -> None:
    """A6 — the metrics object is frozen, so the caveat field cannot be mutated
    away after construction (object.__setattr__-style strip is blocked by frozen
    dataclass semantics)."""
    m = ReidSuccessMetrics(
        reid_recall=0.0,
        reid_precision=0.0,
        reid_success_rate=0.0,
        correct=0,
        n_targets=1,
        n_guesses=0,
        candidate_set_size=1,
        adversary_id="adv@v1",
        deterministic=True,
    )
    with pytest.raises((AttributeError, TypeError)):
        m.caveat = ""  # type: ignore[misc]


def test_nfr016_a6_as_outcome_always_carries_caveat() -> None:
    """[CONTRACT-TEST] A6 — as_outcome() is the sandbox runner's return Mapping
    and ALWAYS carries the non-strippable caveat + the metric fields."""
    m = ReidSuccessMetrics(
        reid_recall=0.5,
        reid_precision=0.5,
        reid_success_rate=0.25,
        correct=1,
        n_targets=2,
        n_guesses=2,
        candidate_set_size=2,
        adversary_id="adv@v1",
        deterministic=True,
    )
    outcome = m.as_outcome()
    assert isinstance(outcome, Mapping)
    assert outcome["caveat"] == ANTI_ANONYMITY_CAVEAT
    for field_name in (
        "reid_recall",
        "reid_precision",
        "reid_success_rate",
        "correct",
        "n_targets",
        "n_guesses",
        "candidate_set_size",
        "adversary_id",
        "deterministic",
    ):
        assert field_name in outcome


# --------------------------------------------------------------------------- #
# A7 — runner runs under the sandbox; A8 — runner outcome is a Mapping.       #
# --------------------------------------------------------------------------- #
def _runner_inputs() -> dict[str, Any]:
    """Synthetic scalar JSON-string args for the allow-listed runner (AX-001)."""
    targets = [
        {"target_id": "p_alpha", "anonymized_text": "...", "observed_signals": ["cardiology", "boston"]},
        {"target_id": "p_beta", "anonymized_text": "[REDACTED]", "observed_signals": []},
    ]
    candidates = [
        {"persona_id": "p_alpha", "quasi_identifiers": ["cardiology", "boston", "marathon"]},
        {"persona_id": "p_beta", "quasi_identifiers": ["oncology", "denver", "cycling"]},
    ]
    return {
        "targets_json": json.dumps(targets),
        "candidates_json": json.dumps(candidates),
        "candidate_set_size": 2,
    }


def _reid_spec(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "attack_id": "synthetic-reid-001",
        "attack_kind": "reid",
        "runner": _REID_RUNNER,
        "params": {},
        "budget": {"cpu_seconds": 5.0, "address_space_bytes": 512 * 1024 * 1024, "wall_seconds": 5.0},
        "allowed_paths": [],
        "network": "deny",
    }
    base.update(overrides)
    return base


def test_nfr016_a8_runner_returns_mapping_outcome() -> None:
    """[CONTRACT-TEST] A8 — reid_attack_runner(...) returns a Mapping[str, Any]
    (the sandbox contract; a non-Mapping return would be a SandboxViolation). The
    outcome carries the metric fields + the non-strippable caveat."""
    outcome = reid_attack_runner(**_runner_inputs())
    assert isinstance(outcome, Mapping)
    assert outcome["caveat"] == ANTI_ANONYMITY_CAVEAT
    assert outcome["n_targets"] == 2
    assert outcome["candidate_set_size"] == 2
    # p_alpha re-identified via surviving QI; p_beta abstains (no signal).
    assert outcome["correct"] == 1
    assert outcome["n_guesses"] == 1


def test_fr011_a7_runner_runs_under_sandbox_returns_mapping() -> None:
    """[SECURITY-TEST][INTEGRATION-TEST] A7 — the reid_baseline runner runs via
    run_attack_under_sandbox (scalar JSON args, attack_kind='reid') and returns an
    AttackResult whose outcome is a Mapping with the metric fields. Every attack
    body runs under the sandbox (the headline security invariant)."""
    result = run_attack_under_sandbox(_reid_spec(), inputs=_runner_inputs())
    assert isinstance(result, AttackResult)
    assert result.attack_id == "synthetic-reid-001"
    assert result.terminated_for_budget is False
    assert isinstance(result.outcome, Mapping)
    assert result.outcome["caveat"] == ANTI_ANONYMITY_CAVEAT
    assert result.outcome["n_targets"] == 2
    # the capability pre-check + egress-denied posture were asserted in the audit.
    assert any("allow-listed" in line.lower() or "capability" in line.lower() for line in result.audit)


def test_fr011_a7_reid_runner_is_registered_in_sandbox_default_registry() -> None:
    """A7 — the reid runner is additively merged into the sandbox default
    allow-list registry (a plain dict lookup is the only way a spec selects it)."""
    assert _REID_RUNNER in DEFAULT_ATTACK_REGISTRY
    assert callable(DEFAULT_ATTACK_REGISTRY[_REID_RUNNER])
    assert "reid_baseline" in REID_ATTACK_REGISTRY
    assert REID_ATTACK_REGISTRY["reid_baseline"] is reid_attack_runner
    # the recon runner from S5-04 is still present (additive, not replacing).
    assert "pii_anon.harness.attack.reconstruction_attack_score" in DEFAULT_ATTACK_REGISTRY


# --------------------------------------------------------------------------- #
# A9 — deterministic AttackResult equality (wall-clock excluded).            #
# --------------------------------------------------------------------------- #
def test_ax002_a9_two_sandboxed_runs_yield_equal_attack_result() -> None:
    """[PROPERTY-TEST] A9 — two sandboxed reid runs on the same spec+inputs yield
    equal AttackResult (AX-002; the sandbox AttackResult equality excludes
    wall-clock timing)."""
    spec = _reid_spec()
    inputs = _runner_inputs()
    r1 = run_attack_under_sandbox(spec, inputs=inputs)
    r2 = run_attack_under_sandbox(spec, inputs=inputs)
    assert r1 == r2
    assert r1.outcome == r2.outcome


# --------------------------------------------------------------------------- #
# A11 — no dangerous call signatures + no nondeterminism imports in reid.py.  #
# (the package-wide guard lives in tests/test_attack_sandbox.py; here we add a #
#  reid-scoped pin that the new body imports no random/uuid/time/secrets.)     #
# --------------------------------------------------------------------------- #
def _reid_source_tree() -> ast.AST:
    import pii_anon.eval_framework.attacks.reid as reid_mod

    path = Path(reid_mod.__file__)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


# nondeterminism-source module heads forbidden in a deterministic attack body
# (AX-002). Assembled plainly — these are not blocked substrings.
_NONDETERMINISM_MODULES: frozenset[str] = frozenset({"random", "uuid", "time", "secrets"})


def _imported_module_heads(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.append(node.module.split(".")[0])
    return out


def test_ax002_a11_reid_body_imports_no_nondeterminism_sources() -> None:
    """[AUDIT][SECURITY-TEST] A11 — reid.py imports NONE of random/uuid/time/
    secrets: the baseline ranking is a pure total order, so replay is
    byte-identical (AX-002). This is a reid-scoped complement to the package-wide
    no-unsafe-call-signature guard in tests/test_attack_sandbox.py."""
    heads = set(_imported_module_heads(_reid_source_tree()))
    leaked = heads & _NONDETERMINISM_MODULES
    assert not leaked, f"reid.py imports nondeterminism source(s): {sorted(leaked)}"


# Forbidden dynamic-eval builtins / unsafe-deserialization module heads, built
# from parts so the literal blocked substrings never appear contiguously here
# (token-reword rule; mirrors tests/test_attack_sandbox.py).
_FORBIDDEN_CALL_FUNC_NAMES: frozenset[str] = frozenset({"ev" + "al", "ex" + "ec", "compile", "__import__"})
_FORBIDDEN_MODULE_NAMES: frozenset[str] = frozenset({"p" + "ickle", "sub" + "process", "marshal", "shelve"})
_FORBIDDEN_OS_ATTR_NAMES: frozenset[str] = frozenset({"sy" + "stem", "po" + "pen", "ex" + "ecv"})


def test_ax002_a11_reid_body_has_no_unsafe_call_signatures() -> None:
    """[AUDIT][SECURITY-TEST] A11 — reid.py contains ZERO unsafe-deserialization
    imports, ZERO subprocess/shell-out attribute calls, ZERO dynamic-eval builtin
    calls. AST-based (node names), so a docstring mention cannot false-positive.
    The attack body is pure-stdlib over scalar inputs."""
    tree = _reid_source_tree()
    violations: list[str] = []
    for head in _imported_module_heads(tree):
        if head in _FORBIDDEN_MODULE_NAMES:
            violations.append(f"imports forbidden module head {head!r}")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_FUNC_NAMES:
            violations.append(f"calls forbidden builtin {func.id!r}")
        if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_OS_ATTR_NAMES:
            violations.append(f"calls forbidden attr .{func.attr}")
    assert not violations, "reid.py has unsafe call signatures: " + "; ".join(violations)


# --------------------------------------------------------------------------- #
# REFACTOR — additive edge tests (the shared JSON decoder's defensive branches #
# + helper behaviour). No production behaviour changes; coverage hardening.    #
# --------------------------------------------------------------------------- #
def test_runner_rejects_non_array_targets_json() -> None:
    """A malformed targets_json whose top level is not a JSON array is refused
    loudly (the shared object-array decoder; no unsafe-deserialization)."""
    with pytest.raises(ValueError, match="target must decode to a JSON array"):
        reid_attack_runner(targets_json=json.dumps({"not": "an array"}), candidates_json="[]", candidate_set_size=1)


def test_runner_rejects_non_array_candidates_json() -> None:
    with pytest.raises(ValueError, match="candidate must decode to a JSON array"):
        reid_attack_runner(
            targets_json="[]", candidates_json=json.dumps("scalar-not-array"), candidate_set_size=1
        )


def test_runner_rejects_non_object_target_element() -> None:
    """Every array element must be a JSON object; a scalar element is refused."""
    with pytest.raises(ValueError, match="target element must be a JSON object"):
        reid_attack_runner(targets_json=json.dumps(["scalar"]), candidates_json="[]", candidate_set_size=1)


def test_runner_rejects_non_array_observed_signals_field() -> None:
    """A target.observed_signals that is not a JSON array (e.g. a scalar) is
    refused — the field is a token list, never a scalar/object."""
    bad = [{"target_id": "t1", "observed_signals": "oncology"}]
    with pytest.raises(ValueError, match="target.observed_signals must be a JSON array"):
        reid_attack_runner(targets_json=json.dumps(bad), candidates_json="[]", candidate_set_size=1)


def test_runner_rejects_non_array_quasi_identifiers_field() -> None:
    bad = [{"persona_id": "p1", "quasi_identifiers": {"k": "v"}}]
    with pytest.raises(ValueError, match="candidate.quasi_identifiers must be a JSON array"):
        reid_attack_runner(targets_json="[]", candidates_json=json.dumps(bad), candidate_set_size=1)


def test_runner_empty_inputs_yield_zeroed_metrics_with_caveat() -> None:
    """Empty targets+candidates: zero-target metrics, no division-by-zero, and the
    non-strippable caveat still present (NFR-016 holds on the empty outcome)."""
    outcome = reid_attack_runner(targets_json="[]", candidates_json="[]", candidate_set_size=0)
    assert outcome["n_targets"] == 0
    assert outcome["n_guesses"] == 0
    assert outcome["reid_recall"] == 0.0
    assert outcome["reid_success_rate"] == 0.0
    assert outcome["caveat"] == ANTI_ANONYMITY_CAVEAT


def test_baseline_normalises_token_case_and_whitespace() -> None:
    """The weighted-Jaccard tokenisation is case/whitespace-insensitive, so a
    surviving signal matches a quasi-identifier regardless of case/padding."""
    personas = [_persona("p_alpha", ("Cardiology", " Boston "))]
    target = _target("t1", "...", ("  CARDIOLOGY ", "boston"))
    [g] = BaselineDeterministicReidAttack().attack([target], personas, candidate_set_size=1)
    assert g.guessed_persona_id == "p_alpha"
    assert g.score == pytest.approx(1.0)  # identical normalised token sets


def test_baseline_custom_adversary_id_is_pinned() -> None:
    """The adversary_id is version-pinnable (so results cite vs adversary@version)
    and flows through score_reid_attack into the outcome."""
    adv = BaselineDeterministicReidAttack(adversary_id="reid-custom@v9")
    assert adv.adversary_id == "reid-custom@v9"
    metrics = score_reid_attack(
        [], [], adversary_id=adv.adversary_id, deterministic=adv.deterministic, candidate_set_size=0
    )
    assert metrics.adversary_id == "reid-custom@v9"
    assert metrics.as_outcome()["adversary_id"] == "reid-custom@v9"
