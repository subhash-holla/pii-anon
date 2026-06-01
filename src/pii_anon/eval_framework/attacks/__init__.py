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
]
