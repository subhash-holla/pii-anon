"""S2-02 — offline distillation that emits the signed ``gate_v1.json`` (DC-02).

This is the **producer** for the runtime :class:`~pii_anon.routing.distilled_gate.
DistilledTopKGate`. It is NOT on the hot path: it runs offline to distil the
*frozen* XGBoost survival oracle (the meta-learner, P(span is a true positive))
into a compact per-``(entity_type, expert)`` advisory gate, and signs that
artifact via the S2-05 :class:`~pii_anon.moe_gate_signing.KeyRing`.

Closed-form compact student (documented, not gradient-fit)
----------------------------------------------------------
For a table-shaped student — one constant routing logit per
``(entity_type, expert)`` cell — the BCE- and KL-minimizing per-cell value is
exactly the (weighted) MEAN of the teacher's soft survival targets in that cell.
Proof sketch: minimizing ``Σ_i w_i · CE(p, t_i)`` (or ``Σ_i w_i · KL(t_i || p)``)
over a single scalar ``p`` has the unique stationary point ``p* = Σ w_i t_i / Σ
w_i``. So we do not gradient-fit a student; we compute each cell's RRS-up-weighted
oracle-survival mean directly (deterministic, exact), then a temperature-softmax
across experts within each entity type yields the Switch/Mistral-style advisory
routing weights. The RRS up-weighting (``compute_sample_weights_from_records``)
sharpens the gate on Tier-3 hard cases without changing the closed form.

Determinism (NFR-005 / AX-002)
------------------------------
``distill_topk_gate`` uses no ``random`` / ``uuid`` / ``time`` / ``secrets`` — it
is grouped means + softmax (``math`` only). Records are grouped in first-seen
order and experts are sorted within each entity type before the softmax, so the
payload is a pure function of ``(oracle, records, temperature, seed,
feature_version)`` and ``canonical_gate_bytes(payload)`` is byte-identical across
replays. The ``seed`` is threaded for contract symmetry with the trainer (and
seeds the optional XGBoost oracle in tests); the closed-form distillation itself
samples nothing.

Dependency isolation (RISK-5)
-----------------------------
xgboost is NOT imported at module top level — the :class:`SurvivalOracle` Protocol
decouples the trainer from any concrete teacher (``XGBoostMetaLearner`` satisfies
it structurally; a synthetic oracle drives the deterministic teeth). xgboost stays
an optional ``swarm-ml`` extra.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pii_anon.moe_gate_signing import KeyRing
from pii_anon.routing.distilled_gate import SUPPORTED_SCHEMA_VERSION
from pii_anon.swarm_learner import compute_sample_weights_from_records, select_f2_threshold


@runtime_checkable
class SurvivalOracle(Protocol):
    """The survival teacher the gate is distilled from.

    Any object exposing ``predict(features) -> list[float]`` — the per-row
    probability that a span is a true positive — satisfies this Protocol
    structurally. :class:`~pii_anon.swarm_learner.XGBoostMetaLearner` already does,
    so the frozen meta-learner plugs in without any xgboost import here; a
    deterministic synthetic oracle drives the pure-determinism tests.
    """

    def predict(self, features: list[list[float]]) -> list[float]:
        """Return P(true positive) for each feature row (aligned with input)."""
        ...


def _oracle_hash(oracle: SurvivalOracle) -> str:
    """sha256 (hex) of the oracle's stable bytes — binds the artifact to its
    teacher (DC-02 oracle-hash).

    Uses ``oracle.model_bytes()`` when available (the real meta-learner can expose
    its serialized model); otherwise falls back to a stable ``repr``. Deterministic
    for a fixed oracle (NFR-005)."""
    model_bytes = getattr(oracle, "model_bytes", None)
    if callable(model_bytes):
        raw = model_bytes()
        if not isinstance(raw, (bytes, bytearray)):
            raw = repr(raw).encode("utf-8")
    else:
        raw = repr(oracle).encode("utf-8")
    return hashlib.sha256(bytes(raw)).hexdigest()


def _temperature_softmax(values: list[float], temperature: float) -> list[float]:
    """Numerically-stable softmax of ``values`` scaled by ``1/temperature``.

    Deterministic and order-preserving; subtracts the max for stability so large
    logits do not overflow ``math.exp``."""
    scaled = [v / temperature for v in values]
    hi = max(scaled)
    exps = [math.exp(v - hi) for v in scaled]
    total = sum(exps)
    if total <= 0.0:  # pragma: no cover - unreachable: max-subtraction forces an exp(0)=1.0 term
        # Degenerate (cannot happen for finite inputs) — fall back to uniform.
        n = len(values)
        return [1.0 / n for _ in values]
    return [e / total for e in exps]


def distill_topk_gate(
    oracle: SurvivalOracle,
    records: list[Any],
    *,
    temperature: float = 1.0,
    seed: int = 42,
    feature_version: int = 1,
) -> dict[str, Any]:
    """Distil ``oracle`` over ``records`` into a compact advisory gate payload.

    Groups ``records`` by ``(entity_type, expert_id)`` (each record must expose
    ``entity_type``, ``expert_id``, and a ``features`` row plus the Tier-3 RRS
    annotations :func:`compute_sample_weights_from_records` reads); computes each
    cell's RRS-up-weighted oracle-survival MEAN (the closed-form BCE/KL-optimal
    compact student); then a temperature-softmax across experts within each entity
    type. Stamps ``oracle_hash`` / ``gate_feature_version`` / ``schema_version`` /
    ``temperature`` and an advisory ``emission_threshold`` (via
    :func:`select_f2_threshold`).

    Deterministic: same ``(oracle, records, temperature, seed, feature_version)``
    ⇒ byte-identical ``canonical_gate_bytes(payload)`` (NFR-005 / AX-002).

    Parameters
    ----------
    oracle : SurvivalOracle
        The frozen survival teacher.
    records : list
        Training records, each with ``entity_type`` / ``expert_id`` / ``features``
        and optional ``re_identification_resistance_score`` / ``is_paired_profile``.
    temperature : float
        Softmax temperature (> 0). Lower = sharper routing.
    seed : int
        Threaded for contract symmetry / oracle seeding; the closed-form
        distillation samples nothing.
    feature_version : int
        The feature-vector version the oracle was distilled against (stamped as
        ``gate_feature_version``).

    Returns
    -------
    dict
        The unsigned gate payload (sign it with :func:`emit_signed_gate`).
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    # Group records by (entity_type, expert_id) in first-seen order (deterministic).
    cells: dict[str, dict[str, list[Any]]] = {}
    for record in records:
        entity_type = record.entity_type
        expert_id = record.expert_id
        cells.setdefault(entity_type, {}).setdefault(expert_id, []).append(record)

    # Per-cell RRS-up-weighted oracle-survival mean = the closed-form compact
    # student (the BCE/KL-minimizing per-cell constant for a table-shaped student).
    weights: dict[str, dict[str, float]] = {}
    all_scores: list[float] = []
    for entity_type, experts in cells.items():
        cell_means: dict[str, float] = {}
        for expert_id, recs in experts.items():
            survivals = oracle.predict([rec.features for rec in recs])
            sample_weights = compute_sample_weights_from_records(recs)
            wsum = sum(sample_weights)
            if wsum > 0.0:
                wmean = sum(w * s for w, s in zip(sample_weights, survivals)) / wsum
            else:  # pragma: no cover - sample weights are >= default_weight (1.0)
                wmean = sum(survivals) / len(survivals) if survivals else 0.0
            cell_means[expert_id] = wmean
            all_scores.extend(survivals)

        # Temperature-softmax across experts WITHIN this entity type. Sort experts
        # so the softmax (and the emitted weights) are order-stable (NFR-005).
        ordered_experts = sorted(cell_means)
        logits = [cell_means[e] for e in ordered_experts]
        softmaxed = _temperature_softmax(logits, temperature)
        weights[entity_type] = {e: softmaxed[i] for i, e in enumerate(ordered_experts)}

    # Advisory emission threshold (DC-02 reuses select_f2_threshold). Self-consistent
    # synthetic labels (survival >= 0.5 ⇒ positive) keep this deterministic and
    # advisory-only — it is recorded as metadata, never a hard cut here.
    labels = [1 if s >= 0.5 else 0 for s in all_scores]
    emission_threshold, emission_f_beta = select_f2_threshold(all_scores, labels)

    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "gate_feature_version": feature_version,
        "oracle_hash": _oracle_hash(oracle),
        "temperature": float(temperature),
        "weights": weights,
        "emission_threshold": float(emission_threshold),
        "emission_f_beta": float(emission_f_beta),
    }


def emit_signed_gate(
    payload: dict[str, Any],
    *,
    key_ring: KeyRing,
    path: str | Path,
) -> None:
    """Sign ``payload`` and write the ``{payload, signature}`` envelope to ``path``.

    Signs via :meth:`KeyRing.sign_payload` (S2-05 — always the CURRENT key, HMAC
    over ``canonical_gate_bytes(payload)``) and writes the envelope as compact,
    deterministic JSON. Re-emitting the same payload yields a byte-identical file
    (the signature is computed over canonical bytes), so the round-trip
    ``emit_signed_gate`` → :func:`~pii_anon.routing.distilled_gate.load_distilled_gate`
    is reproducible.

    The artifact is written as JSON only — it is never serialized through any
    unsafe object-deserialization path.
    """
    import json  # local import keeps the module's top-level surface minimal

    envelope = key_ring.sign_payload(payload)
    Path(path).write_text(
        json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
