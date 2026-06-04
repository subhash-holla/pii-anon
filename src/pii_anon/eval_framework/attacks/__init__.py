"""DC-09 ``attacks/`` package — the adversarial-attack home (S5-04 substrate).

This package is the DC-09 home the D6 design assigns to CODE for the real
Tier-3 LLM re-identification adversary (FR-011) and the full-power
membership-inference family (LiRA@128 + Secret-Sharer, FR-013). S5-04 seeds it
with the **isolation substrate** every real attack body must run inside:

- a safe structured spec loader (no unsafe-deserialization) — :mod:`.spec`
- an in-process capability + resource sandbox (capability pre-check, SOFT
  rlimits, an in-process network-egress interceptor, a wall-clock watchdog —
  NEVER a subprocess) — :mod:`.sandbox`

The attack bodies themselves are successor stories. The package imports only
the stdlib + needed ``eval_framework``/``harness`` siblings; it imports nothing
from ``swarm`` / ``moe`` / ``fusion`` / ``policy`` (an AST boundary test pins
this). It contains ZERO unsafe-deserialization, ZERO subprocess / shell-out,
and ZERO arbitrary dynamic-eval calls (an AST/source guard pins this).
"""

from __future__ import annotations

from pii_anon.eval_framework.attacks.sandbox import (
    DEFAULT_ATTACK_REGISTRY,
    AttackResult,
    AttackSandbox,
    SandboxBudgetExceeded,
    SandboxPolicy,
    run_attack_under_sandbox,
)
from pii_anon.eval_framework.attacks.spec import (
    DEFAULT_AS_BYTES,
    DEFAULT_CPU_SECONDS,
    DEFAULT_WALL_SECONDS,
    AttackKind,
    AttackSpec,
    NetworkPosture,
    ResourceBudget,
    SandboxViolation,
    load_attack_spec,
)

# S5-01 — the re-identification attack protocol + the deterministic baseline body
# (DC-09; FR-011/FR-013 foundation; NFR-016 non-strippable caveat). Importing it
# here keeps the package surface in one place and lets the import-boundary +
# dangerous-call-signature AST guards scan reid.py alongside the substrate.
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

# Additively merge the re-identification runners into the sandbox default
# allow-list registry. ``DEFAULT_ATTACK_REGISTRY`` (defined in ``sandbox``) is the
# single plain-dict allow-list ``run_attack_under_sandbox`` resolves against; we
# extend it in place WITHOUT replacing it, so the S5-04 reconstruction runner
# stays registered and the merge is the ONLY way the new bodies join the
# allow-list (no new dynamic-import path is introduced). The registry remains a
# plain in-code dict — the only way a spec selects code.
for _name, _runner in REID_ATTACK_REGISTRY.items():
    DEFAULT_ATTACK_REGISTRY[_name] = _runner  # type: ignore[index]  # plain dict (Final[Mapping] is type-level)
del _name, _runner

__all__ = [
    # spec surface
    "AttackSpec",
    "AttackKind",
    "NetworkPosture",
    "ResourceBudget",
    "SandboxViolation",
    "load_attack_spec",
    # sandbox surface
    "AttackSandbox",
    "SandboxPolicy",
    "SandboxBudgetExceeded",
    "AttackResult",
    "run_attack_under_sandbox",
    "DEFAULT_ATTACK_REGISTRY",
    "DEFAULT_CPU_SECONDS",
    "DEFAULT_AS_BYTES",
    "DEFAULT_WALL_SECONDS",
    # reid surface (S5-01)
    "ReidPersona",
    "ReidTarget",
    "ReidGuess",
    "ReidSuccessMetrics",
    "ReidAttack",
    "MiaAttack",
    "BaselineDeterministicReidAttack",
    "score_reid_attack",
    "reid_attack_runner",
    "REID_ATTACK_REGISTRY",
    "ANTI_ANONYMITY_CAVEAT",
]
