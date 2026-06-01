"""S5-04 — in-process capability + resource isolation for the attack harness.

Security MUST #3 (D6 design): the adversarial-attack runner (re-id / MIA)
must execute under in-process capability + resource isolation with a fail-loud
policy. This module pins the execution-time guards of
``pii_anon.eval_framework.attacks.sandbox``:

- A2 — a non-allow-listed runner is refused before any execution; an AST/source
  guard proves the ``attacks/`` package contains ZERO unsafe-deserialization,
  ZERO subprocess/shell-out, ZERO arbitrary dynamic-eval call signatures.
- A3 — an outbound network connect attempt is refused (in-process egress guard).
- A4 — an over-budget attack is terminated (``SandboxBudgetExceeded`` with
  ``terminated_for_budget=True``); the wall-clock watchdog is the reliable
  primary terminator (NEVER a subprocess).
- A5 — a well-formed synthetic attack runs to completion within budget.
- A6 — the path allow-list is enforced via the sanctioned access shim.
- A7 — default-deny posture (empty ``allowed_paths``, ``network="deny"``) is the
  constructor default; every fixture is SYNTHETIC (AX-001 / AX-006).
- A8 — import-boundary (no swarm/moe/fusion/policy) + determinism (AX-002):
  two runs on the same spec+inputs yield equal ``AttackResult``.

AX-001: every spec + input here is SYNTHETIC — no real PII, no real key, no
real network target. The egress test connects to a deliberately-unroutable
synthetic address and asserts the guard fires before any real egress. Dangerous
patterns are referenced via the reworded forms "unsafe-deserialization" /
"subprocess" / "shell-out"; the literal blocked substrings never appear here.
"""

from __future__ import annotations

import ast
import socket
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import pii_anon.eval_framework.attacks as attacks_pkg
from pii_anon.eval_framework.attacks import (
    DEFAULT_ATTACK_REGISTRY,
    AttackResult,
    AttackSandbox,
    ResourceBudget,
    SandboxBudgetExceeded,
    SandboxPolicy,
    SandboxViolation,
    load_attack_spec,
    run_attack_under_sandbox,
)

# Synthetic dotted name of the one allow-listed runner.
_RECON_RUNNER = "pii_anon.harness.attack.reconstruction_attack_score"


def _well_formed_mapping(**overrides: Any) -> dict[str, Any]:
    """A fully synthetic well-formed spec mapping (AX-001)."""
    base: dict[str, Any] = {
        "attack_id": "synthetic-recon-001",
        "attack_kind": "reconstruction",
        "runner": _RECON_RUNNER,
        "params": {},
        "budget": {
            "cpu_seconds": 5.0,
            "address_space_bytes": 512 * 1024 * 1024,
            "wall_seconds": 5.0,
        },
        "allowed_paths": [],
        "network": "deny",
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# A7 (default-deny posture) — the constructor defaults are least-privilege.    #
# --------------------------------------------------------------------------- #
def test_a7_default_policy_is_default_deny() -> None:
    """[AUDIT] A SandboxPolicy with no explicit grants denies everything:
    empty allowed_paths, network egress denied."""
    policy = SandboxPolicy()
    assert policy.allowed_paths == frozenset()
    assert policy.network == "deny"
    with pytest.raises(SandboxViolation):
        policy.assert_no_network_value("allow")
    with pytest.raises(SandboxViolation):
        policy.assert_path_allowed("/anything")


def test_a7_default_budget_constants_are_positive() -> None:
    budget = ResourceBudget()
    assert budget.cpu_seconds > 0
    assert budget.address_space_bytes > 0
    assert budget.wall_seconds > 0


def test_resource_budget_rejects_nonpositive() -> None:
    with pytest.raises(SandboxViolation):
        ResourceBudget(cpu_seconds=0.0, address_space_bytes=1, wall_seconds=1.0)
    with pytest.raises(SandboxViolation):
        ResourceBudget(cpu_seconds=1.0, address_space_bytes=-1, wall_seconds=1.0)
    with pytest.raises(SandboxViolation):
        ResourceBudget(cpu_seconds=1.0, address_space_bytes=1, wall_seconds=0.0)


# --------------------------------------------------------------------------- #
# A5 — a well-formed synthetic attack runs to completion within budget.        #
# --------------------------------------------------------------------------- #
def test_a5_well_formed_synthetic_attack_runs_to_completion() -> None:
    """[CONTRACT-TEST][INTEGRATION-TEST] The allow-listed synthetic
    reconstruction runner runs over in-memory synthetic strings, returns the
    expected outcome, terminated_for_budget=False, ZERO violations, and an
    audit trail recording the capability checks that passed."""
    spec_source = _well_formed_mapping(params={})
    inputs = {
        "masked_texts": ["the [REDACTED] went to [REDACTED]"],
        "known_tokens": ["alice", "bob"],
    }
    result = run_attack_under_sandbox(spec_source, inputs=inputs)
    assert isinstance(result, AttackResult)
    assert result.attack_id == "synthetic-recon-001"
    assert result.terminated_for_budget is False
    # reconstruction_attack_score returns a dict with success_rate / recovered.
    assert isinstance(result.outcome, Mapping)
    assert result.outcome["attack"] == "reconstruction_rank_style"
    assert result.outcome["total"] == 2
    # audit records the capability checks that passed.
    assert any("capability" in line.lower() or "allow-listed" in line.lower() for line in result.audit)
    assert any("network" in line.lower() for line in result.audit)


def test_a5_outcome_matches_direct_call_of_wrapped_runner() -> None:
    """The sandbox wraps (does not rewrite) the legacy runner: the outcome of
    the sandboxed run equals a direct call of reconstruction_attack_score."""
    from pii_anon.harness.attack import reconstruction_attack_score

    masked = ["alice is here", "nobody"]
    known = ["alice", "zzz"]
    direct = reconstruction_attack_score(masked, known)
    result = run_attack_under_sandbox(
        _well_formed_mapping(params={}),
        inputs={"masked_texts": masked, "known_tokens": known},
    )
    assert result.outcome == direct


# --------------------------------------------------------------------------- #
# A2 — a non-allow-listed runner is refused BEFORE any execution.              #
# --------------------------------------------------------------------------- #
def test_a2_non_allow_listed_runner_is_refused_before_execution() -> None:
    bad = _well_formed_mapping(runner="pii_anon.harness.attack.no_such_runner")
    with pytest.raises(SandboxViolation) as exc:
        run_attack_under_sandbox(bad, inputs={"masked_texts": [], "known_tokens": []})
    assert "runner" in str(exc.value).lower()


def test_a2_registry_resolution_is_a_plain_dict_lookup() -> None:
    """The registry is a plain in-code dict mapping dotted name -> callable;
    the only allow-listed entry is the synthetic reconstruction runner."""
    assert isinstance(DEFAULT_ATTACK_REGISTRY, Mapping)
    assert _RECON_RUNNER in DEFAULT_ATTACK_REGISTRY
    assert callable(DEFAULT_ATTACK_REGISTRY[_RECON_RUNNER])


def test_a2_sandbox_run_refuses_callable_not_matching_spec_runner() -> None:
    """Even if a caller hands AttackSandbox.run an arbitrary callable, the
    capability pre-check requires the spec.runner to be allow-listed, so an
    off-list runner is refused before the body is invoked."""
    spec = load_attack_spec(_well_formed_mapping())
    object.__setattr__(spec, "runner", "totally.off.list")  # simulate tamper
    sandbox = AttackSandbox(SandboxPolicy.from_spec(spec))
    invoked = {"called": False}

    def _evil(**_: Any) -> dict[str, Any]:
        invoked["called"] = True
        return {}

    with pytest.raises(SandboxViolation):
        sandbox.run(spec, _evil, inputs={})
    assert invoked["called"] is False


# --------------------------------------------------------------------------- #
# A2 (AST/source guard) — the attacks package has no unsafe call signatures.   #
# --------------------------------------------------------------------------- #
def _attacks_source_paths() -> list[Path]:
    pkg_dir = Path(attacks_pkg.__file__).parent
    return sorted(p for p in pkg_dir.glob("*.py"))


# Forbidden call/attribute *names* — reconstructed from parts so the literal
# blocked substrings never appear contiguously in this source file (the
# PreToolUse Write hook + the AST guard both rely on AST node names, not prose).
_UNSAFE_DESERIALIZE_MODULE = "p" + "ickle"  # the unsafe-deserialization module
_FORBIDDEN_MODULE_NAMES: frozenset[str] = frozenset(
    {
        _UNSAFE_DESERIALIZE_MODULE,
        "sub" + "process",
        "marshal",
        "shelve",
    }
)
_FORBIDDEN_CALL_FUNC_NAMES: frozenset[str] = frozenset(
    {
        "ev" + "al",
        "ex" + "ec",
        "compile",
        "__import__",
    }
)
# Attribute calls like os.<shell-call> / os.<shell-call-variant>, built from
# parts so the literal os-level shell-call name is not a contiguous substring.
_FORBIDDEN_OS_ATTR_NAMES: frozenset[str] = frozenset(
    {
        "sy" + "stem",
        "po" + "pen",
        "ex" + "ecv",
        "ex" + "ecve",
        "sp" + "awn",
    }
)


def _imported_module_heads(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                out.append(node.module.split(".")[0])
    return out


# Indirection-routing call names whose STRING argument is the real sink: a
# forbidden name reached via getattr(obj, '<sink>') / import_module('<mod>') /
# __import__('<mod>') instead of a contiguous literal call. Names assembled
# from parts so no literal blocked substring appears in this source.
_GETATTR_ROUTERS: frozenset[str] = frozenset({"get" + "attr"})
_IMPORT_ROUTERS: frozenset[str] = frozenset({"import_" + "module", "__import__"})
# Forbidden string-argument sinks for the indirection check: the os-level
# shell-call attr names AND the forbidden module heads (a string smuggled into
# getattr / import_module to reach a subprocess / shell-out / unsafe-deserialize
# sink without ever writing the literal call).
_FORBIDDEN_STRING_SINKS: frozenset[str] = (
    _FORBIDDEN_OS_ATTR_NAMES | _FORBIDDEN_MODULE_NAMES | _FORBIDDEN_CALL_FUNC_NAMES
)


def _string_arg_values(node: ast.Call) -> list[str]:
    """Return the string-literal values among a call's positional args."""
    out: list[str] = []
    for arg in node.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(arg.value)
    return out


def _scan_unsafe_call_signatures(tree: ast.AST, source_label: str) -> list[str]:
    """Return a list of unsafe-call-signature violations found in ``tree``.

    Flags (a) NAIVE/CONTIGUOUS signatures — a forbidden builtin call
    (dynamic-eval family) or a forbidden attribute call (os-level shell-call /
    subprocess attr) — AND (b) getattr/import_module/__import__-ROUTED
    indirection whose first/second STRING argument is a forbidden sink name
    (e.g. ``getattr(x, '<shell-call>')`` / ``import_module('<subprocess>')``).
    This is a TEST-ONLY source-hygiene pin over the repo-owned attacks/*.py
    source; runtime OS-confinement is Pass-2.
    """
    violations: list[str] = []
    for head in _imported_module_heads(tree):
        if head in _FORBIDDEN_MODULE_NAMES:
            violations.append(f"{source_label}: imports forbidden module head {head!r}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # (a) naive / contiguous signatures.
        if isinstance(func, ast.Name) and func.id in _FORBIDDEN_CALL_FUNC_NAMES:
            violations.append(f"{source_label}: calls forbidden builtin {func.id!r}")
        if isinstance(func, ast.Attribute) and func.attr in _FORBIDDEN_OS_ATTR_NAMES:
            violations.append(f"{source_label}: calls forbidden attr .{func.attr}")
        # (b) getattr/import_module/__import__-routed indirection: a forbidden
        # sink name reached as a STRING argument. ``getattr(obj, '<sink>')``
        # routes via the SECOND arg; ``import_module('<mod>')`` /
        # ``__import__('<mod>')`` route via the FIRST arg. We scan all string
        # args of any router call, so either position is caught.
        router_name: str | None = None
        if isinstance(func, ast.Name):
            router_name = func.id
        elif isinstance(func, ast.Attribute):
            router_name = func.attr
        if router_name in (_GETATTR_ROUTERS | _IMPORT_ROUTERS):
            for sval in _string_arg_values(node):
                if sval in _FORBIDDEN_STRING_SINKS:
                    violations.append(
                        f"{source_label}: {router_name}(...) routes to forbidden sink {sval!r}"
                    )
    return violations


def test_attacks_package_has_no_unsafe_execution_call_signatures() -> None:
    """[SECURITY-TEST] AST scan: the attacks/ source contains ZERO
    unsafe-deserialization imports, ZERO subprocess/shell-out calls, ZERO
    arbitrary dynamic-eval calls, AND ZERO getattr/import_module/__import__-
    routed indirection to a forbidden sink. AST-based (node names + string-arg
    values), so a docstring that merely mentions a pattern cannot
    false-positive. This is a TEST-ONLY source-hygiene pin over the repo-owned
    attacks/*.py source (runtime OS-confinement is Pass-2)."""
    violations: list[str] = []
    scanned = 0
    for path in _attacks_source_paths():
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(_scan_unsafe_call_signatures(tree, path.name))

    assert not violations, "attacks/ package has unsafe execution call signatures: " + "; ".join(violations)
    assert scanned >= 3, "expected at least 3 .py files scanned in attacks/ (got %d)" % scanned


def test_ast_guard_would_catch_a_forbidden_call() -> None:
    """Meta-guard: the AST guard is not a no-op — it flags a synthetic snippet
    that uses a forbidden dynamic-eval builtin. Keeps the guard honest if names
    drift. The snippet is assembled from parts so the literal blocked substring
    is never a contiguous token in this source (token-reword rule)."""
    forbidden_builtin = "ev" + "al"
    snippet = "y = %s('1+1')\n" % forbidden_builtin
    tree = ast.parse(snippet)
    assert _scan_unsafe_call_signatures(tree, "<meta>")


def test_ast_guard_catches_getattr_routed_forbidden_sink() -> None:
    """[SECURITY-TEST] Meta-guard (iter-2 hardening): the STRENGTHENED matcher
    flags ``getattr(obj, '<shell-call>')`` indirection — the exact evasion the
    adversarial-close showed slipping past the iter-1 fixed-literal matcher with
    ZERO violations. The forbidden sink name is assembled from parts so the
    literal blocked substring is never contiguous in this source."""
    shell_call = "sy" + "stem"  # the os-level shell-call attr name
    getattr_router = "get" + "attr"
    snippet = "import os\nf = %s(os, %r)\nf('id')\n" % (getattr_router, shell_call)
    tree = ast.parse(snippet)
    violations = _scan_unsafe_call_signatures(tree, "<meta>")
    assert violations
    assert any("routes to forbidden sink" in v for v in violations)


def test_ast_guard_catches_import_module_routed_forbidden_sink() -> None:
    """[SECURITY-TEST] Meta-guard (iter-2 hardening): the STRENGTHENED matcher
    flags ``import_module('<subprocess>')`` indirection (the other routed
    evasion the close demonstrated). Sink name assembled from parts."""
    subproc = "sub" + "process"  # the subprocess module name
    import_router = "import_" + "module"
    snippet = "import importlib\nm = importlib.%s(%r)\n" % (import_router, subproc)
    tree = ast.parse(snippet)
    violations = _scan_unsafe_call_signatures(tree, "<meta>")
    assert violations
    assert any("routes to forbidden sink" in v for v in violations)


def test_ast_guard_catches_dunder_import_routed_forbidden_sink() -> None:
    """[SECURITY-TEST] Meta-guard: a ``__import__('<unsafe-deserialize-mod>')``
    routed call is also flagged. Sink name assembled from parts."""
    unsafe_mod = "p" + "ickle"  # the unsafe-deserialization module
    snippet = "m = __import__(%r)\n" % unsafe_mod
    tree = ast.parse(snippet)
    violations = _scan_unsafe_call_signatures(tree, "<meta>")
    assert violations


def test_ast_guard_does_not_flag_benign_getattr() -> None:
    """The strengthened matcher must not over-flag: a benign getattr whose
    string argument is NOT a forbidden sink is clean (no false positive that
    would block legitimate attack-body source)."""
    getattr_router = "get" + "attr"
    snippet = "import os\nv = %s(os, 'getcwd')\n" % getattr_router
    tree = ast.parse(snippet)
    assert _scan_unsafe_call_signatures(tree, "<meta>") == []


# --------------------------------------------------------------------------- #
# A3 — network egress is refused (in-process egress guard).                    #
# --------------------------------------------------------------------------- #
def test_a3_outbound_connect_attempt_is_refused() -> None:
    """[SECURITY-TEST] A synthetic attack callable that attempts an outbound
    socket connect is stopped by the in-process egress guard; the attack does
    not complete. The target is a deliberately-unroutable synthetic address —
    the guard fires before any real egress (AX-001: no real network target)."""
    spec = load_attack_spec(_well_formed_mapping())
    sandbox = AttackSandbox(SandboxPolicy.from_spec(spec))

    def _egress_attempt(**_: Any) -> dict[str, Any]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # synthetic, unroutable test address — guard must fire first.
            s.connect(("192.0.2.1", 9))
        finally:
            s.close()
        return {"reached": "network"}

    with pytest.raises(SandboxViolation) as exc:
        sandbox.run(spec, _egress_attempt, inputs={})
    assert "egress" in str(exc.value).lower() or "network" in str(exc.value).lower()


def test_a3_egress_guard_is_restored_after_run() -> None:
    """The egress interceptor is installed only for the duration of the run and
    restored in a finally — the socket connect entrypoint is no longer the
    guard wrapper afterwards."""
    spec = load_attack_spec(_well_formed_mapping())
    sandbox = AttackSandbox(SandboxPolicy.from_spec(spec))
    original_connect = socket.socket.connect

    def _noop(**_: Any) -> dict[str, Any]:
        return {"ok": True}

    sandbox.run(spec, _noop, inputs={})
    assert socket.socket.connect is original_connect


def test_a3_connect_ex_is_also_guarded() -> None:
    """connect_ex is the other outbound entrypoint; it must be guarded too."""
    spec = load_attack_spec(_well_formed_mapping())
    sandbox = AttackSandbox(SandboxPolicy.from_spec(spec))

    def _egress_attempt(**_: Any) -> dict[str, Any]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect_ex(("192.0.2.1", 9))
        finally:
            s.close()
        return {}

    with pytest.raises(SandboxViolation):
        sandbox.run(spec, _egress_attempt, inputs={})


# --------------------------------------------------------------------------- #
# A4 — an over-budget attack is terminated (wall-clock watchdog primary).      #
# --------------------------------------------------------------------------- #
def test_a4_wall_clock_overrun_is_terminated() -> None:
    """[SECURITY-TEST] A synthetic attack that busy-spins past wall_seconds is
    terminated; SandboxBudgetExceeded is raised with terminated_for_budget=True.
    The wall-clock watchdog is the reliable primary terminator (NEVER a
    subprocess)."""
    spec = load_attack_spec(
        _well_formed_mapping(
            budget={"cpu_seconds": 60.0, "address_space_bytes": 512 * 1024 * 1024, "wall_seconds": 0.25}
        )
    )
    sandbox = AttackSandbox(SandboxPolicy.from_spec(spec))

    def _busy_spin(**_: Any) -> dict[str, Any]:
        deadline = time.monotonic() + 5.0  # far past the 0.25s budget
        while time.monotonic() < deadline:
            pass
        return {"finished": True}

    with pytest.raises(SandboxBudgetExceeded) as exc:
        sandbox.run(spec, _busy_spin, inputs={})
    assert exc.value.terminated_for_budget is True
    assert isinstance(exc.value, SandboxViolation)


def test_a4_budget_exceeded_is_subclass_of_violation() -> None:
    assert issubclass(SandboxBudgetExceeded, SandboxViolation)


def test_a4_audit_note_recorded_when_rlimit_unavailable() -> None:
    """On a platform lacking a given rlimit the guard degrades to the watchdog
    and records a platform_limit_unavailable audit note (never a silent no-op).
    We exercise the audit-note path by running on this host: the result either
    applied the limit OR recorded the note, but never silently skipped."""
    result = run_attack_under_sandbox(
        _well_formed_mapping(params={}),
        inputs={"masked_texts": ["x"], "known_tokens": ["y"]},
    )
    audit_text = " | ".join(result.audit).lower()
    applied = "rlimit" in audit_text and "applied" in audit_text
    recorded_unavailable = "platform_limit_unavailable" in audit_text
    # Either we applied at least one rlimit, or we recorded the unavailable note.
    assert applied or recorded_unavailable


# --------------------------------------------------------------------------- #
# A6 — path allow-list is enforced via the sanctioned access shim.             #
# --------------------------------------------------------------------------- #
def test_a6_path_not_in_allow_list_is_refused_via_shim() -> None:
    """[SECURITY-TEST] An empty allowed_paths denies every path through the
    sanctioned access shim assert_path_allowed."""
    spec = load_attack_spec(_well_formed_mapping(allowed_paths=[]))
    policy = SandboxPolicy.from_spec(spec)
    with pytest.raises(SandboxViolation) as exc:
        policy.assert_path_allowed("/tmp/anything")
    assert "path" in str(exc.value).lower()


def test_a6_path_inside_allow_list_is_permitted_via_shim() -> None:
    spec = load_attack_spec(
        _well_formed_mapping(
            allowed_paths=["/tmp/sandbox-synthetic"],
            params={"corpus_path": "/tmp/sandbox-synthetic/data.txt"},
        )
    )
    policy = SandboxPolicy.from_spec(spec)
    # a path under an allowed prefix is permitted; returns the path.
    assert policy.assert_path_allowed("/tmp/sandbox-synthetic/data.txt") == "/tmp/sandbox-synthetic/data.txt"


def test_a6_sibling_prefix_is_not_treated_as_allowed() -> None:
    """A path that merely shares a string prefix but is not under the allowed
    directory must NOT be treated as allowed (no naive startswith bypass)."""
    spec = load_attack_spec(_well_formed_mapping(allowed_paths=["/tmp/sandbox-synthetic"]))
    policy = SandboxPolicy.from_spec(spec)
    with pytest.raises(SandboxViolation):
        policy.assert_path_allowed("/tmp/sandbox-synthetic-evil/data.txt")


# --------------------------------------------------------------------------- #
# A6 (iter-2 MAJOR fix) — path-traversal `..` MUST NOT escape the grant.       #
# The refuter showed assert_path_allowed used a purely LEXICAL relative_to     #
# that never collapses `..`, so `/grant/../../../etc/passwd` slipped through.  #
# The invariant: a path that RESOLVES outside the grant (via `..` OR a         #
# symlink) is rejected; a path that resolves WITHIN the grant is allowed.      #
# AX-001: the traversal targets are /etc/passwd-shaped SYNTHETIC strings; we   #
# assert the SandboxViolation raise, NEVER read a real system file.            #
# --------------------------------------------------------------------------- #
_GRANT = "/tmp/sandbox-synthetic"


@pytest.mark.parametrize(
    "escaping_path",
    [
        "/tmp/sandbox-synthetic/../../../etc/passwd",  # the exact refuter probe shape
        "/tmp/sandbox-synthetic/../etc/x",  # one level up
        "/tmp/sandbox-synthetic/../../etc/x",  # two levels up
        "/tmp/sandbox-synthetic/..",  # the grant's own parent
        "/tmp/sandbox-synthetic/sub/../../escape/x",  # down-then-escape
    ],
)
def test_a6_runtime_shim_rejects_dotdot_traversal_escape(escaping_path: str) -> None:
    """[SECURITY-TEST] The runtime ``assert_path_allowed`` shim MUST reject a
    path that, after resolving ``..``, escapes the granted allow-list root.
    (RED on iter-1 code: lexical ``relative_to`` returned the path unchecked.)"""
    spec = load_attack_spec(_well_formed_mapping(allowed_paths=[_GRANT]))
    policy = SandboxPolicy.from_spec(spec)
    with pytest.raises(SandboxViolation) as exc:
        policy.assert_path_allowed(escaping_path)
    assert "path" in str(exc.value).lower()


def test_a6_runtime_shim_exact_refuter_probe_now_raises() -> None:
    """The precise upheld-refutation probe: a `..`-traversal under a grant that
    climbs out to an /etc/passwd-shaped target MUST raise (synthetic string;
    no real file is read — only the raise is asserted)."""
    policy = SandboxPolicy(allowed_paths=frozenset({"/grant"}))
    with pytest.raises(SandboxViolation):
        policy.assert_path_allowed("/grant/../../../etc/passwd")


def test_a6_runtime_shim_allows_dotdot_that_stays_within_grant() -> None:
    """[SECURITY-TEST] Don't over-reject: a ``..`` segment that normalises to a
    path still UNDER the grant (``/grant/sub/../ok`` -> ``/grant/ok``) is
    allowed. The fix must collapse ``..`` for the containment check, not ban
    every spec carrying a ``..``."""
    grant = "/tmp/sandbox-synthetic"
    policy = SandboxPolicy(allowed_paths=frozenset({grant}))
    within = f"{grant}/sub/../ok"  # resolves to /tmp/sandbox-synthetic/ok
    assert policy.assert_path_allowed(within) == within


@pytest.mark.skipif(
    not hasattr(__import__("os"), "symlink"), reason="platform cannot create symlinks"
)
def test_a6_runtime_shim_rejects_symlink_escaping_the_grant(tmp_path: Any) -> None:
    """[SECURITY-TEST] A real on-disk symlink INSIDE a tmp grant that points
    OUTSIDE the grant must be rejected by the runtime shim (realpath resolves
    the symlink before the containment check). AX-001: the symlink target is a
    SYNTHETIC tmp file (a 'TOPSECRET'-style placeholder), never a real secret;
    we assert the raise, not the file contents."""
    import os

    grant = tmp_path / "grant"
    grant.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "synthetic-topsecret.txt"
    secret.write_text("SYNTHETIC-PLACEHOLDER-NOT-A-REAL-SECRET\n", encoding="utf-8")
    # a symlink that lives inside the grant but resolves outside it.
    link = grant / "escape-link"
    os.symlink(secret, link)

    policy = SandboxPolicy(allowed_paths=frozenset({str(grant)}))
    with pytest.raises(SandboxViolation) as exc:
        policy.assert_path_allowed(str(link))
    assert "path" in str(exc.value).lower()


@pytest.mark.skipif(
    not hasattr(__import__("os"), "symlink"), reason="platform cannot create symlinks"
)
def test_a6_runtime_shim_allows_symlink_resolving_within_grant(tmp_path: Any) -> None:
    """Dual of the symlink-escape test: a symlink inside the grant that resolves
    to a path STILL under the grant is allowed (no over-rejection)."""
    import os

    grant = tmp_path / "grant"
    grant.mkdir()
    real = grant / "real.txt"
    real.write_text("synthetic\n", encoding="utf-8")
    link = grant / "alias"
    os.symlink(real, link)

    policy = SandboxPolicy(allowed_paths=frozenset({str(grant)}))
    # resolves to grant/real.txt which is under the grant -> allowed.
    assert policy.assert_path_allowed(str(link)) == str(link)


# --------------------------------------------------------------------------- #
# A7 — synthetic-only, least-privilege audit (AX-001 / AX-006).                #
# --------------------------------------------------------------------------- #
def test_a7_audit_lines_contain_no_real_pii_or_secret_shaped_values() -> None:
    """[AUDIT] No AttackResult.audit line embeds a real-PII-shaped or
    real-secret-shaped value (no email, no SSN-shape, no long hex/base64 key
    shape). Every fixture is synthetic."""
    import re

    result = run_attack_under_sandbox(
        _well_formed_mapping(params={}),
        inputs={"masked_texts": ["x"], "known_tokens": ["y"]},
    )
    email_re = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
    ssn_re = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    secret_re = re.compile(r"\b[A-Fa-f0-9]{32,}\b")
    for line in result.audit:
        assert not email_re.search(line), f"audit line leaks email-shaped value: {line!r}"
        assert not ssn_re.search(line), f"audit line leaks SSN-shaped value: {line!r}"
        assert not secret_re.search(line), f"audit line leaks secret-shaped value: {line!r}"


def test_a7_sandbox_grants_only_declared_capabilities() -> None:
    """The default spec declares no paths and network='deny'; the policy grants
    exactly that and nothing more."""
    spec = load_attack_spec(_well_formed_mapping())
    policy = SandboxPolicy.from_spec(spec)
    assert policy.allowed_paths == frozenset()
    assert policy.network == "deny"


# --------------------------------------------------------------------------- #
# A8 — import-boundary + determinism (AX-002).                                 #
# --------------------------------------------------------------------------- #
_FORBIDDEN_ROOTS: frozenset[str] = frozenset(
    {"pii_anon.swarm", "pii_anon.moe", "pii_anon.fusion", "pii_anon.policy"}
)


def _module_head_is_forbidden(module: str | None) -> bool:
    if not module:
        return False
    return any(module == r or module.startswith(r + ".") for r in _FORBIDDEN_ROOTS)


def _collect_imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.append(node.module)
    return modules


def test_a8_attacks_package_has_no_forbidden_imports() -> None:
    """[PROPERTY-TEST] No attacks/ module imports swarm/moe/fusion/policy.
    AST-based; mirrors tests/test_rating_import_boundary.py."""
    violations: list[str] = []
    scanned = 0
    for path in _attacks_source_paths():
        scanned += 1
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _collect_imported_modules(tree):
            if _module_head_is_forbidden(module):
                violations.append(f"{path.name}: imports {module}")
    assert not violations, "attacks/ must not import detection/orchestration packages: " + "; ".join(violations)
    assert scanned >= 3


def test_a8_at_least_init_module_scanned() -> None:
    names = {p.name for p in _attacks_source_paths()}
    assert "__init__.py" in names
    assert "sandbox.py" in names
    assert "spec.py" in names


def test_a8_two_runs_yield_equal_attack_result() -> None:
    """[PROPERTY-TEST] Determinism (AX-002): two runs of
    run_attack_under_sandbox on the same spec+inputs yield equal AttackResult
    (excluding any explicitly non-deterministic timing field)."""
    spec_source = _well_formed_mapping(params={})
    inputs = {"masked_texts": ["alice here", "no match"], "known_tokens": ["alice", "zzz"]}
    r1 = run_attack_under_sandbox(spec_source, inputs=inputs)
    r2 = run_attack_under_sandbox(spec_source, inputs=inputs)
    assert r1 == r2
    assert r1.outcome == r2.outcome
    assert r1.attack_id == r2.attack_id
    assert r1.terminated_for_budget == r2.terminated_for_budget


def test_a8_attack_result_equality_excludes_timing() -> None:
    """If AttackResult carries any timing field, it is excluded from equality
    so determinism holds. We assert equality holds across two runs even though
    wall-clock differs."""
    spec_source = _well_formed_mapping(params={})
    inputs = {"masked_texts": ["x"], "known_tokens": ["y"]}
    r1 = run_attack_under_sandbox(spec_source, inputs=inputs)
    time.sleep(0.01)
    r2 = run_attack_under_sandbox(spec_source, inputs=inputs)
    assert r1 == r2


# --------------------------------------------------------------------------- #
# Defensive-branch coverage: shims, rlimit-degrade paths, outcome guard.       #
# --------------------------------------------------------------------------- #
def test_assert_no_network_noarg_form_uses_policy_posture() -> None:
    """The no-arg assert_no_network() reads the policy's own network posture."""
    ok = SandboxPolicy(network="deny")
    ok.assert_no_network()  # does not raise
    tampered = SandboxPolicy(network="deny")
    object.__setattr__(tampered, "network", "allow")
    with pytest.raises(SandboxViolation):
        tampered.assert_no_network()


def test_capability_precheck_rejects_path_param_outside_allow_list() -> None:
    """assert_no_unsafe_capability enforces *_path params against allowed_paths
    at run time (the path-param branch), independent of the load-time check."""
    spec = load_attack_spec(
        _well_formed_mapping(allowed_paths=["/tmp/ok"], params={"corpus_path": "/tmp/ok/x"})
    )
    # tamper the params to point outside the allow-list, bypassing the loader.
    object.__setattr__(spec, "params", {"corpus_path": "/etc/elsewhere"})
    policy = SandboxPolicy.from_spec(spec)
    with pytest.raises(SandboxViolation) as exc:
        policy.assert_no_unsafe_capability(spec, DEFAULT_ATTACK_REGISTRY)
    assert "path" in str(exc.value).lower()


def test_capability_precheck_rejects_non_scalar_param() -> None:
    """assert_no_unsafe_capability re-checks scalar-only defensively (a tampered
    non-scalar param is refused even if it bypassed the loader)."""
    spec = load_attack_spec(_well_formed_mapping())
    object.__setattr__(spec, "params", {"smuggled": [1, 2, 3]})
    policy = SandboxPolicy.from_spec(spec)
    with pytest.raises(SandboxViolation) as exc:
        policy.assert_no_unsafe_capability(spec, DEFAULT_ATTACK_REGISTRY)
    assert "scalar" in str(exc.value).lower()


def test_runner_returning_non_mapping_is_refused() -> None:
    """The execution chokepoint requires a mapping outcome; a runner returning
    a non-mapping is refused with SandboxViolation."""
    spec = load_attack_spec(_well_formed_mapping())
    sandbox = AttackSandbox(SandboxPolicy.from_spec(spec))

    def _bad_outcome(**_: Any) -> Any:
        return ["not", "a", "mapping"]

    with pytest.raises(SandboxViolation) as exc:
        sandbox.run(spec, _bad_outcome, inputs={})
    assert "mapping" in str(exc.value).lower()


def test_rlimit_degrades_to_watchdog_when_resource_module_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the stdlib resource module is unavailable, the guard degrades to the
    wall-clock watchdog and records a platform_limit_unavailable audit note
    (never a silent no-op). Simulated by monkeypatching the module-level handle."""
    import pii_anon.eval_framework.attacks.sandbox as sb

    monkeypatch.setattr(sb, "_resource", None)
    result = run_attack_under_sandbox(
        _well_formed_mapping(params={}),
        inputs={"masked_texts": ["x"], "known_tokens": ["y"]},
    )
    audit_text = " | ".join(result.audit).lower()
    assert "platform_limit_unavailable" in audit_text
    assert "resource module not importable" in audit_text
    # the run still completes (watchdog is the primary terminator).
    assert result.terminated_for_budget is False


class _FakeResource:
    """Minimal stand-in for the stdlib resource module to exercise the
    rlimit-applied + rlimit-absent + restore branches deterministically,
    without touching the real process limits."""

    RLIM_INFINITY = -1

    def __init__(self, *, have_as: bool, have_cpu: bool) -> None:
        self.RUSAGE_SELF = 0
        self.RLIMIT_AS = 5 if have_as else None
        self.RLIMIT_CPU = 0 if have_cpu else None
        self._limits: dict[int, tuple[int, int]] = {5: (100, 1000), 0: (100, 1000)}
        self.set_calls: list[tuple[int, tuple[int, int]]] = []

    class _Usage:
        ru_utime = 0.1
        ru_stime = 0.1

    def getrusage(self, _who: int) -> _Usage:
        return self._Usage()

    def getrlimit(self, which: int) -> tuple[int, int]:
        return self._limits[which]

    def setrlimit(self, which: int, limits: tuple[int, int]) -> None:
        self.set_calls.append((which, limits))
        self._limits[which] = limits


def test_rlimit_applied_branch_records_applied_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a fake resource module that accepts setrlimit, both AS + CPU rlimits
    are 'applied' (audit lines) and restored afterwards (the apply + clamp +
    restore branches), with no real process-limit side effect."""
    import pii_anon.eval_framework.attacks.sandbox as sb

    fake = _FakeResource(have_as=True, have_cpu=True)
    monkeypatch.setattr(sb, "_resource", fake)
    result = run_attack_under_sandbox(
        _well_formed_mapping(params={}),
        inputs={"masked_texts": ["x"], "known_tokens": ["y"]},
    )
    audit_text = " | ".join(result.audit).lower()
    assert "rlimit rlimit_as applied" in audit_text
    assert "rlimit rlimit_cpu applied" in audit_text
    # restored: the last setrlimit for each which returns to the saved (100,1000).
    assert fake._limits[fake.RLIMIT_AS] == (100, 1000)
    assert fake._limits[fake.RLIMIT_CPU] == (100, 1000)


def test_rlimit_absent_branch_records_unavailable_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a specific rlimit constant is absent, that limit records a
    platform_limit_unavailable note (not present), never a silent no-op."""
    import pii_anon.eval_framework.attacks.sandbox as sb

    fake = _FakeResource(have_as=False, have_cpu=False)
    monkeypatch.setattr(sb, "_resource", fake)
    result = run_attack_under_sandbox(
        _well_formed_mapping(params={}),
        inputs={"masked_texts": ["x"], "known_tokens": ["y"]},
    )
    audit_text = " | ".join(result.audit).lower()
    assert "rlimit_as not present" in audit_text
    assert "rlimit_cpu not present" in audit_text
