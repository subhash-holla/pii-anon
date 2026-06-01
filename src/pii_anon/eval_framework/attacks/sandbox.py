"""S5-04 — in-process capability + resource isolation for the attack harness.

DC-09 ``attacks/`` package — Security MUST #3 (D6 design): the adversarial
re-id (FR-011) / MIA (FR-013) runner must execute under in-process capability
+ resource isolation with a fail-loud policy. This module is the single
execution chokepoint:

- ``SandboxPolicy`` — the allow-list (``allowed_paths``), the network posture
  (egress denied), the resource budget; exposes the sanctioned access shims
  ``assert_path_allowed`` / ``assert_no_network`` / ``assert_no_unsafe_capability``.
- ``AttackSandbox.run(spec, attack_callable, *, inputs)`` — (1) a fail-loud
  capability pre-check BEFORE the body; (2) in-process resource limits (stdlib
  ``resource`` rlimits, SOFT, captured-and-restored in a finally, where the
  platform supports them; degrades to the watchdog + a recorded
  ``platform_limit_unavailable`` audit note otherwise); (3) an in-process
  network-egress interceptor (refuses any outbound socket connect); (4) a
  stdlib timer/thread-based wall-clock watchdog — NEVER a subprocess.
- ``run_attack_under_sandbox`` — load -> resolve runner from the allow-list ->
  ``AttackSandbox.run`` -> structured ``AttackResult``.
- ``DEFAULT_ATTACK_REGISTRY`` — the in-code allow-list (a plain dict mapping a
  dotted name to a callable). The synthetic ``reconstruction_attack_score`` is
  registered as the one allow-listed runner for the end-to-end proof.

This module uses ONLY the stdlib (``resource``, ``socket``, ``threading``).
There is NO unsafe-deserialization, NO subprocess / shell-out, and NO arbitrary
dynamic-eval anywhere in this package — an AST/source guard pins that (the
patterns are referenced in prose via the reworded forms only).
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Final

from pii_anon.eval_framework.attacks.spec import (
    AttackSpec,
    ResourceBudget,
    SandboxViolation,
    _is_within_allowed,
    load_attack_spec,
)
from pii_anon.harness.attack import reconstruction_attack_score

# ``resource`` is POSIX-only. Import defensively so the package still imports on
# a platform without it; the guard then degrades to the watchdog + audit note.
try:  # pragma: no cover - import availability is platform-specific
    import resource as _resource
except ImportError:  # pragma: no cover
    _resource = None  # type: ignore[assignment]


AttackCallable = Callable[..., Mapping[str, Any]]


class SandboxBudgetExceeded(SandboxViolation):
    """An attack exceeded its CPU / memory / wall-time budget and was terminated.

    Subclass of :class:`SandboxViolation`. ``terminated_for_budget`` is always
    True so a caller can distinguish a budget kill from a capability refusal.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.terminated_for_budget: bool = True


@dataclass(frozen=True)
class AttackResult:
    """The structured outcome of one sandboxed attack run.

    ``outcome`` is the wrapped runner's return value. ``audit`` records the
    capability checks that passed + any platform-degrade notes. Determinism
    (AX-002): equality is over (attack_id, outcome, terminated_for_budget,
    audit) only — there is no wall-clock timing field in the equality at all,
    so two runs on the same spec+inputs compare equal.
    """

    attack_id: str
    outcome: Mapping[str, Any]
    terminated_for_budget: bool
    audit: tuple[str, ...]


@dataclass(frozen=True)
class SandboxPolicy:
    """Capability policy for one attack run — default-deny.

    Constructor defaults are least-privilege (AX-006): empty ``allowed_paths``,
    ``network="deny"``, the project-default budget. :meth:`from_spec` derives a
    policy from a validated :class:`AttackSpec`.
    """

    allowed_paths: frozenset[str] = field(default_factory=frozenset)
    network: str = "deny"
    budget: ResourceBudget = field(default_factory=ResourceBudget)

    @classmethod
    def from_spec(cls, spec: AttackSpec) -> SandboxPolicy:
        return cls(allowed_paths=spec.allowed_paths, network=spec.network, budget=spec.budget)

    # -- sanctioned access shims ------------------------------------------- #
    def assert_path_allowed(self, path: str) -> str:
        """Return ``path`` iff it is one of, or strictly under, an allowed
        directory; otherwise raise ``SandboxViolation`` (path-not-allow-listed).

        Delegates to the canonical ``spec._is_within_allowed`` (path-component
        containment, not a naive ``startswith``) so a sibling that merely shares
        a string prefix is NOT allowed, and the load-time + run-time path checks
        cannot drift.
        """
        if _is_within_allowed(path, self.allowed_paths):
            return path
        raise SandboxViolation(f"path {path!r} is not in the allow-list {sorted(self.allowed_paths)!r}")

    def assert_no_network_value(self, value: str) -> None:
        """Raise unless ``value == 'deny'`` (network-egress-denied posture)."""
        if value != "deny":
            raise SandboxViolation(f"network egress is denied; spec requested network={value!r}")

    def assert_no_network(self) -> None:
        self.assert_no_network_value(self.network)

    def assert_no_unsafe_capability(self, spec: AttackSpec, registry: Mapping[str, AttackCallable]) -> None:
        """Fail-loud capability pre-check run BEFORE any attack body.

        Validates: the runner is allow-listed; ``network=="deny"``; every
        declared ``*_path`` param is within ``allowed_paths``; params are
        scalars only (already enforced at load, re-checked defensively).
        """
        if spec.runner not in registry:
            raise SandboxViolation(
                f"runner {spec.runner!r} is not allow-listed; refusing before execution "
                f"(no subprocess / shell-out / external-program target)"
            )
        self.assert_no_network_value(spec.network)
        for key, value in spec.params.items():
            if not isinstance(value, (str, int, float, bool)):
                raise SandboxViolation(f"params[{key!r}] is not a scalar (possible unsafe-deserialization smuggling)")
            if key.endswith("_path") and isinstance(value, str):
                self.assert_path_allowed(value)


# --------------------------------------------------------------------------- #
# In-process network-egress interceptor (installed only for a run's duration). #
# --------------------------------------------------------------------------- #
class _EgressGuard:
    """Monkeypatch the socket connect entrypoints to refuse outbound egress.

    Active only inside the ``with`` block; the original methods are restored in
    ``__exit__`` (also on exception). Real kernel-level egress blocking
    (namespaces) is Pass-2; this is the in-process AGENT_SIMULATED guard.
    """

    def __init__(self) -> None:
        self._orig_connect: Any = None
        self._orig_connect_ex: Any = None

    def __enter__(self) -> _EgressGuard:
        self._orig_connect = socket.socket.connect
        self._orig_connect_ex = socket.socket.connect_ex

        def _denied_connect(_self: Any, *_a: Any, **_k: Any) -> Any:
            raise SandboxViolation("network egress denied: outbound socket connect refused by sandbox")

        def _denied_connect_ex(_self: Any, *_a: Any, **_k: Any) -> Any:
            raise SandboxViolation("network egress denied: outbound socket connect_ex refused by sandbox")

        socket.socket.connect = _denied_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = _denied_connect_ex  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: Any) -> None:
        socket.socket.connect = self._orig_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = self._orig_connect_ex  # type: ignore[method-assign]


# --------------------------------------------------------------------------- #
# In-process rlimit application (SOFT only; captured-and-restored).            #
# --------------------------------------------------------------------------- #
def _apply_soft_rlimits(budget: ResourceBudget, audit: list[str]) -> list[tuple[int, tuple[int, int]]]:
    """Apply SOFT rlimits for AS + CPU where supported; return saved limits.

    Records an ``rlimit ... applied`` audit line per applied limit, or a
    ``platform_limit_unavailable`` note where a limit is unavailable (never a
    silent no-op). The caller MUST restore the returned saved limits in a
    finally.
    """
    saved: list[tuple[int, tuple[int, int]]] = []
    if _resource is None:
        audit.append("platform_limit_unavailable: resource module not importable; relying on wall-clock watchdog")
        return saved

    # RLIMIT_CPU is measured against CUMULATIVE process CPU since start, not per
    # call. Setting the bare budget as the soft limit in-process would send
    # SIGXCPU as soon as the host process (e.g. the test runner) had already
    # consumed more than ``cpu_seconds`` — destabilizing it. We therefore set
    # the CPU soft limit RELATIVE to CPU already consumed (consumed + budget),
    # so it caps THIS attack's additional CPU without killing the host process.
    # The wall-clock watchdog remains the reliable primary terminator.
    consumed_cpu_seconds = 0.0
    try:
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        consumed_cpu_seconds = float(usage.ru_utime + usage.ru_stime)
    except (ValueError, OSError):  # pragma: no cover - getrusage failure is rare
        consumed_cpu_seconds = 0.0

    cpu_soft = int(consumed_cpu_seconds + budget.cpu_seconds) + 1  # +1s headroom, ceil

    for name, which, soft_value in (
        ("RLIMIT_AS", getattr(_resource, "RLIMIT_AS", None), budget.address_space_bytes),
        ("RLIMIT_CPU", getattr(_resource, "RLIMIT_CPU", None), cpu_soft),
    ):
        if which is None:
            audit.append(f"platform_limit_unavailable: {name} not present; relying on wall-clock watchdog")
            continue
        try:
            cur_soft, cur_hard = _resource.getrlimit(which)
            saved.append((which, (cur_soft, cur_hard)))
            # SOFT limit only: never raise the hard ceiling; clamp to the hard.
            new_soft = soft_value
            if cur_hard != _resource.RLIM_INFINITY:
                new_soft = min(soft_value, cur_hard)
            _resource.setrlimit(which, (new_soft, cur_hard))
            # Determinism (AX-002 / A8): the audit line is part of AttackResult
            # equality, so it must NOT embed the run-varying consumed-CPU value
            # or the consumed-relative soft number. For RLIMIT_CPU we record the
            # stable BUDGET, noting the relative-to-consumed application; for
            # RLIMIT_AS the soft value is the budget (already deterministic).
            if name == "RLIMIT_CPU":
                audit.append(
                    f"rlimit {name} applied: budget={budget.cpu_seconds}s "
                    f"(soft set relative to CPU already consumed)"
                )
            else:
                audit.append(f"rlimit {name} applied: soft={new_soft}")
        except (ValueError, OSError) as exc:
            audit.append(f"platform_limit_unavailable: {name} setrlimit refused ({exc}); relying on wall-clock watchdog")
    return saved


def _restore_rlimits(saved: list[tuple[int, tuple[int, int]]]) -> None:
    if _resource is None:
        return
    for which, limits in saved:
        try:
            _resource.setrlimit(which, limits)
        except (ValueError, OSError):  # pragma: no cover - restore is best-effort
            pass


class AttackSandbox:
    """The single execution chokepoint for an adversarial attack body."""

    def __init__(self, policy: SandboxPolicy, *, registry: Mapping[str, AttackCallable] | None = None) -> None:
        self.policy = policy
        self.registry: Mapping[str, AttackCallable] = (
            registry if registry is not None else DEFAULT_ATTACK_REGISTRY
        )

    def run(self, spec: AttackSpec, attack_callable: AttackCallable, *, inputs: Mapping[str, Any]) -> AttackResult:
        """Execute ``attack_callable(**inputs)`` under capability + resource
        isolation. Fail-loud: any out-of-sandbox capability raises
        ``SandboxViolation`` BEFORE the body; an over-budget run raises
        ``SandboxBudgetExceeded`` (``terminated_for_budget=True``)."""
        audit: list[str] = []

        # (1) Capability pre-check — BEFORE the body is ever invoked.
        self.policy.assert_no_unsafe_capability(spec, self.registry)
        audit.append(f"capability pre-check passed: runner {spec.runner!r} is allow-listed")
        audit.append(f"network posture asserted: network={spec.network!r} (egress denied)")
        audit.append(f"path allow-list: {sorted(self.policy.allowed_paths)!r}")

        # (2) Resource limits (SOFT rlimits where supported) + (4) watchdog.
        saved = _apply_soft_rlimits(self.policy.budget, audit)
        try:
            # (3) Network-egress interceptor active for the body's duration.
            with _EgressGuard():
                outcome = self._run_with_wall_watchdog(attack_callable, inputs, self.policy.budget.wall_seconds)
        finally:
            _restore_rlimits(saved)

        audit.append("attack completed within budget")
        return AttackResult(
            attack_id=spec.attack_id,
            outcome=dict(outcome),
            terminated_for_budget=False,
            audit=tuple(audit),
        )

    @staticmethod
    def _run_with_wall_watchdog(
        attack_callable: AttackCallable, inputs: Mapping[str, Any], wall_seconds: float
    ) -> Mapping[str, Any]:
        """Run the body in a worker thread; join with a wall-clock deadline.

        If the worker is still alive after ``wall_seconds`` the run is over
        budget -> ``SandboxBudgetExceeded``. An exception raised inside the body
        (e.g. the egress guard's ``SandboxViolation``) is captured and re-raised
        in the calling thread. The watchdog is a stdlib thread join — NEVER a
        subprocess. (AGENT_SIMULATED: real cgroup wall-clock kill is Pass-2; the
        over-budget worker is abandoned as a daemon and finishes on its own.)
        """
        box: dict[str, Any] = {}

        def _target() -> None:
            try:
                box["outcome"] = attack_callable(**inputs)
            except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
                box["error"] = exc

        worker = threading.Thread(target=_target, name="attack-sandbox-worker", daemon=True)
        start = time.monotonic()
        worker.start()
        worker.join(timeout=wall_seconds)
        if worker.is_alive():
            elapsed = time.monotonic() - start
            raise SandboxBudgetExceeded(
                f"attack exceeded wall_seconds budget ({wall_seconds}s; ran >= {elapsed:.3f}s) — terminated"
            )
        if "error" in box:
            raise box["error"]
        outcome = box.get("outcome")
        if not isinstance(outcome, Mapping):
            raise SandboxViolation(
                f"attack runner must return a mapping outcome (got {type(outcome).__name__})"
            )
        return outcome


def run_attack_under_sandbox(
    spec_source: str | Mapping[str, Any],
    *,
    inputs: Mapping[str, Any],
    registry: Mapping[str, AttackCallable] | None = None,
) -> AttackResult:
    """Public entry: load -> resolve the allow-listed runner -> run sandboxed.

    Loads ``spec_source`` via the safe structured loader (no
    unsafe-deserialization), resolves ``spec.runner`` from the allow-list
    ``registry`` (a plain dict lookup — never a free dynamic import), and runs
    it under :class:`AttackSandbox`. Returns a structured :class:`AttackResult`.
    """
    reg: Mapping[str, AttackCallable] = registry if registry is not None else DEFAULT_ATTACK_REGISTRY
    spec = load_attack_spec(spec_source, registry=reg)
    attack_callable = reg[spec.runner]  # safe: load_attack_spec proved membership
    sandbox = AttackSandbox(SandboxPolicy.from_spec(spec), registry=reg)
    return sandbox.run(spec, attack_callable, inputs=inputs)


# --------------------------------------------------------------------------- #
# The in-code allow-list registry (a plain dict — the ONLY way a spec selects  #
# code). The synthetic ``reconstruction_attack_score`` is the one allow-listed #
# runner for the end-to-end proof (A5). It is wrapped, NOT rewritten.          #
# --------------------------------------------------------------------------- #
DEFAULT_ATTACK_REGISTRY: Final[Mapping[str, AttackCallable]] = {
    "pii_anon.harness.attack.reconstruction_attack_score": reconstruction_attack_score,
}
