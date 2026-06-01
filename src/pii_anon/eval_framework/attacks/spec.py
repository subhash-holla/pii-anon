"""S5-04 — safe structured attack-spec loader (no unsafe-deserialization).

DC-09 ``attacks/`` package — Security MUST #3 from the D6 design: "sandbox the
attack harness (ingests adversarial material)". An attack spec describes what an
adversarial attack (re-id / MIA) does and what it may touch. Because that
material is adversarial by definition, the loader that parses it is a
privilege-escalation surface. This module gives the spec a *safe structured*
representation:

- ``AttackSpec`` — a frozen dataclass with explicit typed fields.
- ``load_attack_spec`` — parses ONLY a safe structured form (a JSON document or
  an already-parsed mapping) via the stdlib structured parser. It NEVER uses an
  arbitrary-object deserializer (no unsafe-deserialization) and NEVER constructs
  objects named in the data; it reads scalar fields out of a plain mapping.

Fail-loud refusals (``SandboxViolation``): a ``runner`` not in the in-code
allow-list registry; a ``params`` value that is not a JSON scalar (a nested
object that could smuggle an object-construction / arbitrary-object-
reconstruction payload); ``network`` other than ``"deny"``; a declared
filesystem path outside ``allowed_paths``.

Dangerous patterns are referenced in prose via the reworded forms
"unsafe-deserialization" / "subprocess" / "shell-out" (token-reword rule);
the loader's own code uses only the stdlib structured parser and plain mapping
access — no dynamic-eval, no subprocess/shell-out, no unsafe-deserialization.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Final, Literal, get_args

# Re-exported from sandbox to avoid a circular import at module-load time;
# the canonical definitions live in sandbox.py. We import lazily inside the
# functions that need them. The exception type is defined here as the shared
# base so spec.py has no import dependency on sandbox.py.

AttackKind = Literal["reid", "mia", "reconstruction"]
NetworkPosture = Literal["deny"]

_VALID_KINDS: Final[frozenset[str]] = frozenset(get_args(AttackKind))
_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "attack_id",
    "attack_kind",
    "runner",
    "budget",
)
# Param keys with this suffix are interpreted as declared filesystem paths and
# validated against ``allowed_paths`` at load time (acceptance A6-load).
_PATH_PARAM_SUFFIX: Final[str] = "_path"


class SandboxViolation(Exception):
    """A spec requested a capability outside the sandbox, or is malformed.

    The shared fail-loud exception for both the load-time refusals (this module)
    and the execution-time guards (``sandbox.py``). ``sandbox.py`` defines
    ``SandboxBudgetExceeded`` as a subclass for the resource-budget terminator.
    """


@dataclass(frozen=True)
class ResourceBudget:
    """In-process resource budget for one attack run.

    Positive-bounded, frozen. Defaults are project-pinned. The actual rlimit
    application + the wall-clock watchdog live in ``sandbox.py``; this dataclass
    is the validated value object the loader produces.
    """

    cpu_seconds: float = 30.0
    address_space_bytes: int = 2 * 1024 * 1024 * 1024
    wall_seconds: float = 30.0

    # Project-pinned defaults (re-exported as module constants below).
    def __post_init__(self) -> None:
        if not (isinstance(self.cpu_seconds, (int, float)) and self.cpu_seconds > 0):
            raise SandboxViolation(f"ResourceBudget.cpu_seconds must be > 0 (got {self.cpu_seconds!r})")
        if not (isinstance(self.address_space_bytes, int) and not isinstance(self.address_space_bytes, bool) and self.address_space_bytes > 0):
            raise SandboxViolation(
                f"ResourceBudget.address_space_bytes must be a positive int (got {self.address_space_bytes!r})"
            )
        if not (isinstance(self.wall_seconds, (int, float)) and self.wall_seconds > 0):
            raise SandboxViolation(f"ResourceBudget.wall_seconds must be > 0 (got {self.wall_seconds!r})")


DEFAULT_CPU_SECONDS: Final[float] = ResourceBudget().cpu_seconds
DEFAULT_AS_BYTES: Final[int] = ResourceBudget().address_space_bytes
DEFAULT_WALL_SECONDS: Final[float] = ResourceBudget().wall_seconds


@dataclass(frozen=True)
class AttackSpec:
    """A frozen, fully-validated description of one adversarial attack run.

    Constructed only by :func:`load_attack_spec` from a safe structured source.
    ``runner`` is a dotted name resolved ONLY against an in-code allow-list
    registry (a plain dict) at execution time — never via a free dynamic import
    or arbitrary-object resolution.
    """

    attack_id: str
    attack_kind: AttackKind
    runner: str
    budget: ResourceBudget
    params: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    allowed_paths: frozenset[str] = field(default_factory=frozenset)
    network: NetworkPosture = "deny"


def _parse_structured(source: str | Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a plain mapping parsed via the stdlib structured parser.

    A ``str`` is parsed as a JSON document; a ``Mapping`` is taken as already
    parsed. Anything else, or a malformed document, or a non-mapping top level,
    is refused loudly. No arbitrary-object deserializer is ever used.
    """
    if isinstance(source, str):
        try:
            parsed = json.loads(source)
        except (json.JSONDecodeError, ValueError) as exc:
            raise SandboxViolation(f"attack spec is not a valid structured (JSON) document: {exc}") from exc
    elif isinstance(source, Mapping):
        parsed = source
    else:
        raise SandboxViolation(f"attack spec source must be a JSON document or a mapping (got {type(source).__name__})")

    if not isinstance(parsed, Mapping):
        raise SandboxViolation(f"attack spec top level must be a mapping (got {type(parsed).__name__})")
    return parsed


def _validate_scalar_params(raw_params: Any) -> dict[str, str | int | float | bool]:
    """Accept ONLY a mapping of JSON-scalar values.

    Any non-scalar value (nested mapping / list / tuple / None / object) is the
    headline unsafe-deserialization smuggling vector and is refused loudly,
    naming the offending field (acceptance A1). ``bool`` is a subclass of
    ``int`` and is explicitly allowed as a scalar.
    """
    if raw_params is None:
        return {}
    if not isinstance(raw_params, Mapping):
        raise SandboxViolation(f"params must be a mapping of scalar values (got {type(raw_params).__name__})")
    out: dict[str, str | int | float | bool] = {}
    for key, value in raw_params.items():
        if not isinstance(key, str):
            raise SandboxViolation(f"params key must be a string (got {type(key).__name__})")
        # NOTE: bool is a subclass of int; both are valid scalars. Reject
        # everything that is not exactly one of the scalar types. The explicit
        # tuple of literal types (not the module-level alias) lets the type
        # checker narrow ``value`` to the scalar union.
        if not isinstance(value, (str, int, float, bool)):
            raise SandboxViolation(
                f"params[{key!r}] must be a JSON scalar (str|int|float|bool); "
                f"a non-scalar value is refused (possible unsafe-deserialization smuggling): "
                f"got {type(value).__name__}"
            )
        out[key] = value
    return out


def _normalise_allowed_paths(raw: Any) -> frozenset[str]:
    if raw is None:
        return frozenset()
    if isinstance(raw, (str, bytes)):
        raise SandboxViolation("allowed_paths must be a list/sequence of path strings, not a single string")
    if not isinstance(raw, (list, tuple, frozenset, set)):
        raise SandboxViolation(f"allowed_paths must be a sequence of strings (got {type(raw).__name__})")
    paths: set[str] = set()
    for p in raw:
        if not isinstance(p, str):
            raise SandboxViolation(f"allowed_paths entries must be strings (got {type(p).__name__})")
        paths.add(p)
    return frozenset(paths)


def _is_within_allowed(path: str, allowed_paths: frozenset[str]) -> bool:
    """True iff ``path`` is one of, or strictly under, an allowed directory.

    Uses path-component containment (not a naive ``startswith``) so a sibling
    that merely shares a string prefix (``/a/b-evil`` vs allowed ``/a/b``) is
    NOT treated as allowed. This is the single source of truth for the
    path-allow-list semantics; ``sandbox.SandboxPolicy.assert_path_allowed``
    delegates to it so the two cannot drift.
    """
    target = PurePosixPath(path)
    for allowed in allowed_paths:
        base = PurePosixPath(allowed)
        if target == base:
            return True
        try:
            target.relative_to(base)
            return True
        except ValueError:
            continue
    return False


def _validate_declared_paths(params: Mapping[str, str | int | float | bool], allowed_paths: frozenset[str]) -> None:
    """Refuse any ``*_path``-suffixed param naming a path outside allowed_paths.

    A spec must not declare, in its own params, a filesystem path it has not
    granted itself via ``allowed_paths`` (acceptance A6 at load time).
    """
    for key, value in params.items():
        if key.endswith(_PATH_PARAM_SUFFIX) and isinstance(value, str):
            if not _is_within_allowed(value, allowed_paths):
                raise SandboxViolation(
                    f"params[{key!r}] declares path {value!r} outside allowed_paths {sorted(allowed_paths)!r}"
                )


def load_attack_spec(
    source: str | Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
) -> AttackSpec:
    """Parse + validate a safe structured attack spec into a frozen AttackSpec.

    Parses ONLY a JSON document or an already-parsed mapping via the stdlib
    structured parser. Refuses loudly (``SandboxViolation``) on: malformed /
    non-mapping document; missing required field; unknown ``attack_kind``; a
    ``runner`` not in the allow-list ``registry``; a non-scalar ``params`` value
    (unsafe-deserialization smuggling); ``network`` other than ``"deny"``; or a
    declared path outside ``allowed_paths``. NEVER constructs objects named in
    the data.

    ``registry`` defaults to ``DEFAULT_ATTACK_REGISTRY`` (imported lazily to
    avoid a circular import with ``sandbox.py``).
    """
    if registry is None:
        from pii_anon.eval_framework.attacks.sandbox import DEFAULT_ATTACK_REGISTRY

        registry = DEFAULT_ATTACK_REGISTRY

    data = _parse_structured(source)

    for required in _REQUIRED_FIELDS:
        if required not in data:
            raise SandboxViolation(f"attack spec is missing required field {required!r}")

    attack_id = data["attack_id"]
    if not isinstance(attack_id, str) or not attack_id:
        raise SandboxViolation("attack_id must be a non-empty string")

    attack_kind = data["attack_kind"]
    if attack_kind not in _VALID_KINDS:
        raise SandboxViolation(f"attack_kind must be one of {sorted(_VALID_KINDS)} (got {attack_kind!r})")

    runner = data["runner"]
    if not isinstance(runner, str):
        raise SandboxViolation(f"runner must be a string dotted-name (got {type(runner).__name__})")
    if runner not in registry:
        raise SandboxViolation(
            f"runner {runner!r} is not in the allow-list registry; "
            f"a spec may only select an in-code allow-listed runner "
            f"(no subprocess / shell-out / external-program target)"
        )

    # network — the only legal value is "deny" (acceptance A3 at load time).
    network = data.get("network")
    if network != "deny":
        raise SandboxViolation(f"network must be 'deny' (the only legal value in v1); got {network!r}")

    allowed_paths = _normalise_allowed_paths(data.get("allowed_paths"))
    params = _validate_scalar_params(data.get("params"))
    _validate_declared_paths(params, allowed_paths)

    budget = _coerce_budget(data["budget"])

    return AttackSpec(
        attack_id=attack_id,
        attack_kind=attack_kind,  # validated above against _VALID_KINDS
        runner=runner,
        budget=budget,
        params=params,
        allowed_paths=allowed_paths,
        network="deny",
    )


def _coerce_budget(raw: Any) -> ResourceBudget:
    """Build a ResourceBudget from a scalar mapping (validated positive in
    ``ResourceBudget.__post_init__``). Refuses a non-mapping loudly."""
    if isinstance(raw, ResourceBudget):
        return raw
    if not isinstance(raw, Mapping):
        raise SandboxViolation(f"budget must be a mapping of scalar limits (got {type(raw).__name__})")
    try:
        return ResourceBudget(
            cpu_seconds=float(raw.get("cpu_seconds", DEFAULT_CPU_SECONDS)),
            address_space_bytes=int(raw.get("address_space_bytes", DEFAULT_AS_BYTES)),
            wall_seconds=float(raw.get("wall_seconds", DEFAULT_WALL_SECONDS)),
        )
    except (TypeError, ValueError) as exc:
        raise SandboxViolation(f"budget limits must be numeric scalars: {exc}") from exc
