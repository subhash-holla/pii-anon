"""S5-04 — safe structured loader for attack specs (no unsafe-deserialization).

Security MUST #3 from the D6 design: "sandbox the attack harness (ingests
adversarial material)". An attack spec is adversarial material by definition;
the loader that parses it is a privilege-escalation surface. This module pins
the load-time refusals (acceptance A1 / A2-load / A3-load / A6-load) of
``pii_anon.eval_framework.attacks.spec``:

- ``AttackSpec`` is a frozen dataclass with explicit typed fields; the loader
  parses ONLY a safe structured form (a JSON document or an already-parsed
  mapping) via the stdlib structured parser. It NEVER uses an arbitrary-object
  deserializer and NEVER constructs objects named in the data.
- A ``runner`` not in the in-code allow-list registry is refused loudly.
- A ``params`` value that is not a JSON scalar (a nested object that could
  smuggle an object-construction / arbitrary-object-reconstruction payload) is
  refused loudly — ZERO such specs reach the runner.
- ``network`` may only be ``"deny"``.
- A path outside ``allowed_paths`` is refused at load time.

AX-001 synthetic-only: every fixture here is SYNTHETIC — no real PII, no real
key, no real network target. Dangerous patterns are referenced via the
reworded forms "unsafe-deserialization" / "subprocess" / "shell-out" per the
project token-reword rule; the literal serialization-module name and os-level
shell-call names never appear in source.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from pii_anon.eval_framework.attacks import (
    AttackSpec,
    ResourceBudget,
    SandboxViolation,
    load_attack_spec,
)


# --------------------------------------------------------------------------- #
# Synthetic well-formed spec fixtures (AX-001: no real PII / key / target).    #
# --------------------------------------------------------------------------- #
def _well_formed_mapping() -> dict[str, Any]:
    """A minimal, fully synthetic, well-formed spec mapping.

    ``reconstruction_attack_score`` is the one synthetic allow-listed runner
    registered for the end-to-end proof; it is pure over in-memory strings and
    touches nothing, hence empty ``allowed_paths`` and ``network="deny"``.
    """
    return {
        "attack_id": "synthetic-recon-001",
        "attack_kind": "reconstruction",
        "runner": "pii_anon.harness.attack.reconstruction_attack_score",
        "params": {"trials": 3, "label": "synthetic", "ratio": 0.5, "strict": True},
        "budget": {
            "cpu_seconds": 5.0,
            "address_space_bytes": 512 * 1024 * 1024,
            "wall_seconds": 5.0,
        },
        "allowed_paths": [],
        "network": "deny",
    }


# --------------------------------------------------------------------------- #
# Happy path: a well-formed structured spec loads into a frozen AttackSpec.    #
# --------------------------------------------------------------------------- #
def test_load_well_formed_mapping_returns_frozen_attack_spec() -> None:
    spec = load_attack_spec(_well_formed_mapping())
    assert isinstance(spec, AttackSpec)
    assert spec.attack_id == "synthetic-recon-001"
    assert spec.attack_kind == "reconstruction"
    assert spec.runner == "pii_anon.harness.attack.reconstruction_attack_score"
    assert spec.network == "deny"
    assert spec.allowed_paths == frozenset()
    assert isinstance(spec.budget, ResourceBudget)
    # frozen: field assignment must raise.
    with pytest.raises(Exception):
        spec.attack_id = "mutated"  # type: ignore[misc]


def test_load_accepts_equivalent_json_string_document() -> None:
    """The safe structured form may arrive as a JSON document string; it is
    parsed via the stdlib structured parser, NOT an arbitrary-object loader."""
    as_json = json.dumps(_well_formed_mapping())
    spec_from_str = load_attack_spec(as_json)
    spec_from_map = load_attack_spec(_well_formed_mapping())
    assert spec_from_str == spec_from_map


def test_params_preserves_only_json_scalars() -> None:
    spec = load_attack_spec(_well_formed_mapping())
    assert isinstance(spec.params, Mapping)
    assert spec.params["trials"] == 3
    assert spec.params["label"] == "synthetic"
    assert spec.params["ratio"] == 0.5
    assert spec.params["strict"] is True
    # all values are JSON scalars
    assert all(isinstance(v, (str, int, float, bool)) for v in spec.params.values())


# --------------------------------------------------------------------------- #
# A1 — unsafe-deserialization refusal (the headline security gate).           #
# A non-scalar / object-construction payload in params is rejected loudly.    #
# --------------------------------------------------------------------------- #
def test_a1_nested_object_in_params_is_refused() -> None:
    """A ``params`` value shaped to smuggle arbitrary-object reconstruction
    (a nested mapping) is refused; NO object named in the data is constructed."""
    bad = _well_formed_mapping()
    bad["params"] = {"payload": {"__reduce__": "os-level-callable"}}
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "payload" in str(exc.value)


def test_a1_list_in_params_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["params"] = {"seq": [1, 2, 3]}
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "seq" in str(exc.value)


def test_a1_none_value_in_params_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["params"] = {"maybe": None}
    with pytest.raises(SandboxViolation):
        load_attack_spec(bad)


@given(
    payload=st.one_of(
        st.dictionaries(st.text(min_size=1, max_size=8), st.integers(), min_size=1),
        st.lists(st.integers(), min_size=1),
        st.none(),
        st.tuples(st.integers()),
    )
)
def test_a1_property_every_nonscalar_params_value_is_rejected(payload: Any) -> None:
    """Property test: ANY non-scalar ``params`` value is rejected; ZERO such
    specs reach the runner. Generated payloads are synthetic structural noise
    (AX-001) — never real PII or secrets."""
    bad = _well_formed_mapping()
    bad["params"] = {"smuggled": payload}
    with pytest.raises(SandboxViolation):
        load_attack_spec(bad)


@given(scalar=st.one_of(st.text(max_size=16), st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.booleans()))
def test_a1_property_every_scalar_params_value_is_accepted(scalar: Any) -> None:
    """Dual of the refusal property: every JSON scalar is accepted (no false
    refusal that would block a legitimate synthetic attack)."""
    ok = _well_formed_mapping()
    ok["params"] = {"value": scalar}
    spec = load_attack_spec(ok)
    assert spec.params["value"] == scalar


# --------------------------------------------------------------------------- #
# A2 (load) — a runner not in the allow-list registry is refused loudly.      #
# --------------------------------------------------------------------------- #
def test_a2_unknown_runner_is_refused_at_load() -> None:
    bad = _well_formed_mapping()
    bad["runner"] = "pii_anon.harness.attack.nonexistent_runner"
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "runner" in str(exc.value).lower()


def test_a2_runner_naming_external_program_is_refused() -> None:
    """A spec cannot name an external program / shell-out target: any dotted
    name not in the in-code allow-list registry is refused. (The reworded
    forms 'subprocess' / 'shell-out' are used in prose per token-reword rule.)"""
    bad = _well_formed_mapping()
    bad["runner"] = "/bin/some-external-program"
    with pytest.raises(SandboxViolation):
        load_attack_spec(bad)


# --------------------------------------------------------------------------- #
# A3 (load) — network must be "deny"; anything else refused at load.          #
# --------------------------------------------------------------------------- #
def test_a3_network_allow_is_refused_at_load() -> None:
    bad = _well_formed_mapping()
    bad["network"] = "allow"
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "network" in str(exc.value).lower()


def test_a3_missing_network_field_is_refused() -> None:
    bad = _well_formed_mapping()
    del bad["network"]
    with pytest.raises(SandboxViolation):
        load_attack_spec(bad)


# --------------------------------------------------------------------------- #
# A6 (load) — a declared path outside allowed_paths is refused at load.       #
# --------------------------------------------------------------------------- #
def test_a6_params_path_outside_allowed_paths_is_refused() -> None:
    """A spec that declares a filesystem path (in a ``*_path``-suffixed scalar)
    outside its own ``allowed_paths`` is refused at load time."""
    bad = _well_formed_mapping()
    bad["allowed_paths"] = ["/tmp/sandbox-synthetic"]
    bad["params"] = {"corpus_path": "/etc/somewhere-else"}
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "corpus_path" in str(exc.value) or "path" in str(exc.value).lower()


def test_a6_params_path_inside_allowed_paths_is_accepted() -> None:
    ok = _well_formed_mapping()
    ok["allowed_paths"] = ["/tmp/sandbox-synthetic"]
    ok["params"] = {"corpus_path": "/tmp/sandbox-synthetic/data.txt"}
    spec = load_attack_spec(ok)
    assert "/tmp/sandbox-synthetic" in spec.allowed_paths


# --------------------------------------------------------------------------- #
# A6 (load, iter-2 MAJOR fix) — a `*_path` param using `..` traversal to       #
# escape allowed_paths MUST be refused at LOAD time. The refuter showed the    #
# lexical containment check let `/grant/../../../etc/passwd` through as a       #
# valid AttackSpec. Load-time validation must normalise `..` (os.path.normpath #
# — no filesystem dependence) and reject any path that escapes the normalised  #
# grant. AX-001: targets are /etc/passwd-shaped SYNTHETIC strings; we assert   #
# the SandboxViolation raise, never read a real file.                          #
# --------------------------------------------------------------------------- #
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
def test_a6_load_time_dotdot_traversal_path_is_refused(escaping_path: str) -> None:
    """[SECURITY-TEST] A ``*_path`` param that uses ``..`` to climb out of the
    declared ``allowed_paths`` is refused at load time naming the field/path.
    (RED on iter-1 code: lexical containment accepted the escaping path.)"""
    bad = _well_formed_mapping()
    bad["allowed_paths"] = ["/tmp/sandbox-synthetic"]
    bad["params"] = {"corpus_path": escaping_path}
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "corpus_path" in str(exc.value) or "path" in str(exc.value).lower()


def test_a6_load_time_exact_refuter_probe_now_raises() -> None:
    """The precise upheld-refutation probe at load time: a `..`-traversal
    `corpus_path` under the grant that climbs out to an /etc/passwd-shaped
    target MUST raise SandboxViolation (synthetic string; no real read)."""
    bad = _well_formed_mapping()
    bad["allowed_paths"] = ["/grant"]
    bad["params"] = {"corpus_path": "/grant/../../../etc/passwd"}
    with pytest.raises(SandboxViolation):
        load_attack_spec(bad)


def test_a6_load_time_dotdot_within_grant_is_accepted() -> None:
    """[SECURITY-TEST] Don't over-reject at load: a ``*_path`` whose ``..``
    normalises to a path STILL under the grant (``/grant/sub/../ok`` ->
    ``/grant/ok``) is accepted (the fix collapses ``..`` for the containment
    check; it does not ban every spec carrying a ``..``)."""
    ok = _well_formed_mapping()
    ok["allowed_paths"] = ["/tmp/sandbox-synthetic"]
    ok["params"] = {"corpus_path": "/tmp/sandbox-synthetic/sub/../ok"}
    spec = load_attack_spec(ok)
    assert "/tmp/sandbox-synthetic" in spec.allowed_paths


# --------------------------------------------------------------------------- #
# Structural validation: unknown kind / missing required fields refused.      #
# --------------------------------------------------------------------------- #
def test_unknown_attack_kind_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["attack_kind"] = "exfiltrate"
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "attack_kind" in str(exc.value)


@pytest.mark.parametrize("field", ["attack_id", "attack_kind", "runner", "budget"])
def test_missing_required_field_is_refused(field: str) -> None:
    bad = _well_formed_mapping()
    del bad[field]
    with pytest.raises(SandboxViolation):
        load_attack_spec(bad)


def test_malformed_json_string_is_refused() -> None:
    """A non-JSON / malformed structured document is refused loudly via the
    stdlib structured parser — never via an arbitrary-object deserializer."""
    with pytest.raises(SandboxViolation):
        load_attack_spec("{ this is : not json }")


def test_top_level_non_mapping_is_refused() -> None:
    with pytest.raises(SandboxViolation):
        load_attack_spec("[1, 2, 3]")


def test_budget_must_be_positive_bounded() -> None:
    bad = _well_formed_mapping()
    bad["budget"] = {"cpu_seconds": -1.0, "address_space_bytes": 1, "wall_seconds": 1.0}
    with pytest.raises(SandboxViolation):
        load_attack_spec(bad)


# --------------------------------------------------------------------------- #
# Defensive-branch coverage: the fail-loud refusals must each be pinned.       #
# --------------------------------------------------------------------------- #
def test_source_of_wrong_type_is_refused() -> None:
    """A source that is neither a JSON document string nor a mapping (e.g. an
    int) is refused loudly."""
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(123)  # type: ignore[arg-type]
    assert "json document or a mapping" in str(exc.value).lower()


def test_missing_params_and_allowed_paths_default_to_empty() -> None:
    """A spec omitting params + allowed_paths yields empty defaults (the
    None-handling branches in the validators)."""
    ok = _well_formed_mapping()
    del ok["params"]
    del ok["allowed_paths"]
    spec = load_attack_spec(ok)
    assert dict(spec.params) == {}
    assert spec.allowed_paths == frozenset()


def test_params_not_a_mapping_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["params"] = "not-a-mapping"
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "mapping" in str(exc.value).lower()


def test_params_with_non_string_key_is_refused() -> None:
    """A non-string params key (reachable only via a pre-parsed mapping, not
    JSON) is refused."""
    bad = _well_formed_mapping()
    bad["params"] = {123: "x"}
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "key" in str(exc.value).lower()


def test_allowed_paths_as_bare_string_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["allowed_paths"] = "/tmp/single"
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "list/sequence" in str(exc.value).lower() or "sequence" in str(exc.value).lower()


def test_allowed_paths_not_a_sequence_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["allowed_paths"] = 42
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "sequence" in str(exc.value).lower()


def test_allowed_paths_entry_not_a_string_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["allowed_paths"] = ["/tmp/ok", 99]
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "string" in str(exc.value).lower()


def test_empty_attack_id_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["attack_id"] = ""
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "attack_id" in str(exc.value)


def test_runner_not_a_string_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["runner"] = 123
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "runner" in str(exc.value).lower()


def test_budget_not_a_mapping_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["budget"] = "generous"
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "budget" in str(exc.value).lower()


def test_budget_with_non_numeric_value_is_refused() -> None:
    bad = _well_formed_mapping()
    bad["budget"] = {"cpu_seconds": "lots", "address_space_bytes": 1, "wall_seconds": 1.0}
    with pytest.raises(SandboxViolation) as exc:
        load_attack_spec(bad)
    assert "budget" in str(exc.value).lower() or "numeric" in str(exc.value).lower()


def test_budget_passed_as_resource_budget_instance_is_accepted() -> None:
    """A pre-parsed mapping may carry an already-built ResourceBudget; the
    coercer accepts it as-is (the isinstance fast-path)."""
    ok = _well_formed_mapping()
    ok["budget"] = ResourceBudget(cpu_seconds=2.0, address_space_bytes=1024, wall_seconds=2.0)
    spec = load_attack_spec(ok)
    assert spec.budget.cpu_seconds == 2.0


def test_path_equal_to_allowed_directory_is_accepted_at_load() -> None:
    """A declared path exactly equal to an allowed directory (not strictly
    under it) is accepted — the equality branch of the containment check."""
    ok = _well_formed_mapping()
    ok["allowed_paths"] = ["/tmp/sandbox-synthetic"]
    ok["params"] = {"corpus_path": "/tmp/sandbox-synthetic"}
    spec = load_attack_spec(ok)
    assert "/tmp/sandbox-synthetic" in spec.allowed_paths
