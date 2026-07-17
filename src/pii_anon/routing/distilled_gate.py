"""S2-02 — the runtime ``DistilledTopKGate`` (control-path, security-critical).

The first concrete :class:`~pii_anon.moe.RoutingGate` implementation (DC-02). A
``DistilledTopKGate`` advisorily re-weights the experts the static-softmax router
already selected, per ``(entity_type, expert_id)``, from a *verified* learned-gate
payload. Because ``gate_v1.json`` is a **control-path** artifact — an attacker who
can write it biases the detector, which is exactly why the S2-05 signer exists —
this module is deliberately hostile to any unverified construction path:

* The ONLY way to build a production gate is :func:`load_distilled_gate`, which
  goes through the fail-closed :func:`pii_anon.moe.load_verified_gate` seam with
  ``verify_on_load=True`` (the S2-01 gate's S2-02 entry-gate obligation, AX-006
  least-privilege). There is no verification-skipping escape here (no
  ``verify_on_load`` opt-out) and no raw-dict public constructor on the
  production seam.
* A present-but-unverifiable artifact raises
  :class:`~pii_anon.errors.GateSignatureError` (fail-closed, S2-05) — never a
  silent static fallback. An *absent* artifact returns ``None`` so the caller keeps
  the static ``entity_strengths`` softmax (NFR-026).
* A *verified-but-malformed* payload raises
  :class:`~pii_anon.errors.GatePayloadError` (the one loud shape error) —
  validated by :meth:`DistilledGatePayload.from_payload`.

The gate is **advisory on weighting/selection only** (AX-003): it can re-weight or
re-order experts already in ``base`` but it can NEVER drop a shared-layer span. The
recall-floor no-drop invariant is enforced downstream by
:class:`~pii_anon.routing.floor_fusion.FloorProjectingFusion` /
:class:`~pii_anon.routing.shared_layer.SharedLayerProjector`, which re-inject a
floored span post-fusion regardless of how the gate re-weighted it.

This module is **pure-stdlib**: it imports no xgboost (the trainer's optional
``swarm-ml`` concern), nothing from ``eval_framework``, and nothing that
introduces nondeterminism (no ``random`` / ``uuid`` / ``time`` / ``secrets``). A
call to :meth:`DistilledTopKGate.adjust` is a dict lookup plus a multiply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, TypeGuard

from pii_anon.errors import GatePayloadError
from pii_anon.moe import RouteContext, load_verified_gate
from pii_anon.moe_gate_signing import KeyRing

#: The gate payload schema this runtime understands. A payload stamped with any
#: other ``schema_version`` is a loud :class:`GatePayloadError` — a future schema
#: is an operator/version error, never a best-effort parse.
SUPPORTED_SCHEMA_VERSION = 1

# ── Canonical payload key names ──────────────────────────────────────────────
# Single source of truth for the distilled-gate payload field names, shared by
# the runtime validator (below) AND the offline producer
# (:mod:`pii_anon.routing.gate_distillation`), so the producer and consumer can
# never drift on a key string. The producer stamps exactly KEY_* ; the validator
# requires exactly :data:`_REQUIRED_KEYS`.
KEY_SCHEMA_VERSION = "schema_version"
KEY_GATE_FEATURE_VERSION = "gate_feature_version"
KEY_ORACLE_HASH = "oracle_hash"
KEY_TEMPERATURE = "temperature"
KEY_WEIGHTS = "weights"

#: The required top-level keys of a distilled-gate payload.
_REQUIRED_KEYS = (
    KEY_SCHEMA_VERSION,
    KEY_GATE_FEATURE_VERSION,
    KEY_ORACLE_HASH,
    KEY_TEMPERATURE,
    KEY_WEIGHTS,
)


def _is_finite_number(value: Any) -> TypeGuard[float]:
    """True iff *value* is a real, finite int/float — and NOT a bool.

    Python's ``bool`` is an ``int`` subclass, so ``True``/``False`` would otherwise
    sneak through an ``isinstance(value, (int, float))`` check and be silently
    treated as ``1.0``/``0.0``. A gate weight must be a genuine number, so bools are
    rejected here (mirrors the SDO gate's no-fabrication ``_is_finite_number``)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(value)


@dataclass(frozen=True)
class DistilledGatePayload:
    """A validated, immutable view over a *verified* ``gate_v1.json`` payload.

    Constructed exclusively through :meth:`from_payload`, which rejects a
    malformed/partial payload LOUD (:class:`GatePayloadError`). The signature is
    already checked upstream by :func:`pii_anon.moe.load_verified_gate`; this
    validates the SHAPE of the verified content (no non-finite / negative weights,
    no missing keys, a known ``schema_version``).

    Attributes
    ----------
    schema_version : int
        Must equal :data:`SUPPORTED_SCHEMA_VERSION`.
    gate_feature_version : int
        The feature-vector version the producing oracle was distilled against
        (DC-02 ``gate_feature_version``).
    oracle_hash : str
        sha256 of the teacher oracle's stable bytes (DC-02 oracle-hash).
    temperature : float
        The softmax temperature the producer used (recorded for provenance).
    weights : Mapping[str, Mapping[str, float]]
        Advisory per-``(entity_type, expert_id)`` routing factors. Read-only.
    """

    schema_version: int
    gate_feature_version: int
    oracle_hash: str
    temperature: float
    weights: Mapping[str, Mapping[str, float]]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DistilledGatePayload:
        """Validate a verified payload dict and return an immutable view.

        Raises
        ------
        GatePayloadError
            On a missing required key, a wrong ``schema_version``, a non-mapping
            ``weights`` (or a non-mapping inner cell), or a non-finite / negative /
            boolean advisory weight. A structurally-broken *signed* gate is a loud
            operator error — never coerced into a partial gate.
        """
        if not isinstance(payload, Mapping):
            raise GatePayloadError("gate payload must be a mapping")
        for key in _REQUIRED_KEYS:
            if key not in payload:
                raise GatePayloadError(f"gate payload missing required key: {key!r}")

        schema_version = payload[KEY_SCHEMA_VERSION]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise GatePayloadError("gate payload 'schema_version' must be an int")
        if schema_version != SUPPORTED_SCHEMA_VERSION:
            raise GatePayloadError(
                f"unsupported gate schema_version {schema_version!r} "
                f"(this runtime supports {SUPPORTED_SCHEMA_VERSION})"
            )

        gate_feature_version = payload[KEY_GATE_FEATURE_VERSION]
        if not isinstance(gate_feature_version, int) or isinstance(gate_feature_version, bool):
            raise GatePayloadError("gate payload 'gate_feature_version' must be an int")

        oracle_hash = payload[KEY_ORACLE_HASH]
        if not isinstance(oracle_hash, str) or not oracle_hash:
            raise GatePayloadError("gate payload 'oracle_hash' must be a non-empty string")

        temperature = payload[KEY_TEMPERATURE]
        if not _is_finite_number(temperature) or temperature <= 0.0:
            raise GatePayloadError("gate payload 'temperature' must be a finite number > 0")

        raw_weights = payload[KEY_WEIGHTS]
        if not isinstance(raw_weights, Mapping):
            raise GatePayloadError("gate payload 'weights' must be a mapping")

        validated: dict[str, Mapping[str, float]] = {}
        for entity_type, cell in raw_weights.items():
            if not isinstance(entity_type, str):
                raise GatePayloadError("gate payload 'weights' keys must be entity-type strings")
            if not isinstance(cell, Mapping):
                raise GatePayloadError(
                    f"gate payload 'weights[{entity_type!r}]' must be a mapping of expert->weight"
                )
            inner: dict[str, float] = {}
            for expert_id, weight in cell.items():
                if not isinstance(expert_id, str):
                    raise GatePayloadError(
                        f"gate payload 'weights[{entity_type!r}]' keys must be expert-id strings"
                    )
                if not _is_finite_number(weight):
                    raise GatePayloadError(
                        f"gate weight {entity_type!r}/{expert_id!r} is not a finite number"
                    )
                if weight < 0.0:
                    raise GatePayloadError(
                        f"gate weight {entity_type!r}/{expert_id!r} is negative"
                    )
                inner[expert_id] = float(weight)
            # Freeze each inner cell so the view is immutable end-to-end.
            validated[entity_type] = MappingProxyType(inner)

        return cls(
            schema_version=schema_version,
            gate_feature_version=gate_feature_version,
            oracle_hash=oracle_hash,
            temperature=float(temperature),
            weights=MappingProxyType(validated),
        )


class DistilledTopKGate:
    """Advisory MoE gate implementing the :class:`~pii_anon.moe.RoutingGate`
    Protocol from a verified distilled payload.

    :meth:`adjust` looks up ``weights[entity_type]`` and, for each
    ``(expert_id, base_weight)`` ALREADY in ``base``, multiplies the base weight by
    the gate's advisory factor for that expert (or ``1.0`` when the gate has no
    opinion on it). The router's existing ``_normalize_weights`` re-normalizes and
    validates the result downstream (finite, ≥0, Σ=1.0), so ``adjust`` returns a
    raw advisory vector — never NaN / negative.

    The gate is ``top_k``-aware only in that re-weighting may shift which experts
    dominate; it **never invents an expert** (it returns exactly the ids in
    ``base``, in order) and **never zeroes the floor** (the recall-floor is a
    downstream projection, not the gate's concern). An unknown ``entity_type`` or
    an entity the gate has no opinion on returns ``base`` unchanged.
    """

    def __init__(self, payload: DistilledGatePayload) -> None:
        self._payload = payload

    @property
    def payload(self) -> DistilledGatePayload:
        """The validated, immutable gate payload backing this gate."""
        return self._payload

    def adjust(
        self,
        entity_type: str,
        context: RouteContext | None,
        base: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Return advisory re-weighted ``(expert_id, weight)`` pairs.

        Unknown ``entity_type`` / no overlapping opinion ⇒ ``base`` unchanged. The
        ``context`` is accepted for Protocol conformance and future
        feature-conditioning; this distilled gate routes on ``entity_type`` alone,
        so it is deterministically independent of ``context`` today.
        """
        cell = self._payload.weights.get(entity_type)
        if not cell:
            # Unknown entity type, or an empty opinion ⇒ identity (no perturbation).
            return base
        # Multiply each base weight by the gate's advisory factor (1.0 = no opinion).
        # Only experts already in `base` are touched; never invent an expert.
        return [(expert_id, base_weight * cell.get(expert_id, 1.0)) for expert_id, base_weight in base]


def load_distilled_gate(
    path: str | Path,
    *,
    key_ring: KeyRing,
) -> DistilledTopKGate | None:
    """Construct a :class:`DistilledTopKGate` from a *verified* ``gate_v1.json``.

    This is the ONLY sanctioned construction path for a production gate. It routes
    the artifact through the fail-closed :func:`pii_anon.moe.load_verified_gate`
    seam with ``verify_on_load=True`` (S2-05 / AX-006) — there is no
    verification-skipping opt-out and no raw-dict constructor on this seam.

    * **Absent** ``path`` ⇒ ``None`` (the caller keeps the static
      ``entity_strengths`` softmax, NFR-026).
    * **Present-but-unverifiable** ⇒ :class:`~pii_anon.errors.GateSignatureError`
      (fail-closed, S2-05) — never a silent static fallback.
    * **Verified-but-malformed** ⇒ :class:`~pii_anon.errors.GatePayloadError` (the
      one loud shape error, via :meth:`DistilledGatePayload.from_payload`).

    Parameters
    ----------
    path : str | Path
        Filesystem path to the ``gate_v1.json`` envelope.
    key_ring : KeyRing
        The trust root for signature verification (S2-05).

    Returns
    -------
    DistilledTopKGate | None
        The constructed gate, or ``None`` if the artifact is absent.
    """
    verified = load_verified_gate(path, key_ring=key_ring, verify_on_load=True)
    if verified is None:
        return None
    return DistilledTopKGate(DistilledGatePayload.from_payload(verified))
