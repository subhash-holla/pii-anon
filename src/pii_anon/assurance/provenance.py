"""Provenance + reproducibility gate.

A report's numbers are only publishable when the run is reproducible. This module
captures the environment, builds the :class:`Provenance` record, and exposes the
fail-closed :func:`provenance_gate`.

``HARNESS_CLOSE_CERTIFIED`` operationalizes the red-team's hard requirement that
*no value may carry the ``MEASURED`` label until a 0-upheld adversarial close has
certified this harness*. While it is ``False`` the runner demotes every
``MEASURED`` value to ``ADVISORY`` (with a stated reason). It is flipped to
``True`` only after the close passes, and the certification facts are recorded
below so the methodology can cite them.
"""

from __future__ import annotations

import hashlib
import platform
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any

from pii_anon.eval_framework.validation import is_nonblank_str

# --- harness certification (flipped True only after the adversarial close) ---
HARNESS_CLOSE_CERTIFIED: bool = True
HARNESS_CLOSE_REFERENCE: str = (
    "Phase 1 adversarial close PASSED 0-upheld-blocking on the CERTIFIED (MEASURED) path "
    "(2026-06-18) after 18 rounds of red-team hardening — 11 of them attacking the live "
    "MEASURED path; final confirmatory round 0-upheld. A follow-on PII-egress hardening pass "
    "(2026-06-18, close rounds 19–22; every blocking finding fixed, then 0-upheld) added a "
    "distinctiveness floor + token-boundary containment to the detector-surface layer (round 19) "
    "and a date-safe version-label exemption — the close caught year-first and decorated "
    "2-digit-year dates leaking via labels (rounds 20–21), resolved by deferring to the PII "
    "detector as the date oracle (round 22)."
)

_TRACKED_PACKAGES = ("pii-anon", "numpy", "regex")


@dataclass(frozen=True)
class Provenance:
    dataset_name: str
    dataset_fingerprint: str
    seed: int
    pii_anon_version: str
    python_version: str
    platform: str
    package_versions: dict[str, str]
    user_pipeline_name: str
    user_pipeline_version: str
    user_pipeline_deterministic: bool
    user_pipeline_output_hash: str | None = None
    user_pipeline_transform_hash: str | None = None
    full_dataset_fingerprint: str | None = None
    sample_size: int | None = None
    timestamp: str | None = None
    harness_close_certified: bool = HARNESS_CLOSE_CERTIFIED
    harness_close_reference: str = HARNESS_CLOSE_REFERENCE

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "seed": self.seed,
            "pii_anon_version": self.pii_anon_version,
            "python_version": self.python_version,
            "platform": self.platform,
            "package_versions": dict(self.package_versions),
            "user_pipeline": {
                "name": self.user_pipeline_name,
                "version": self.user_pipeline_version,
                "deterministic": self.user_pipeline_deterministic,
                "output_hash": self.user_pipeline_output_hash,
                "transform_output_hash": self.user_pipeline_transform_hash,
            },
            "sampling": {
                "full_dataset_fingerprint": self.full_dataset_fingerprint or self.dataset_fingerprint,
                "sample_size": self.sample_size,
                "sampled": self.sample_size is not None,
            },
            "timestamp": self.timestamp,
            "harness_close_certified": self.harness_close_certified,
            "harness_close_reference": self.harness_close_reference,
            "determinism_scope": "in-process probe only",
            "reproducibility_claim": (
                "Recomputable by a holder of the original dataset whose sha256 "
                f"matches {self.dataset_fingerprint[:16]}…; not independently "
                "reproducible without the data (AX-001). The `deterministic` flag "
                "reflects an IN-PROCESS probe; cross-process reproducibility is NOT "
                "asserted by the harness alone — an independent party must re-run on the "
                "same dataset and confirm output_hash + transform_output_hash match via "
                "`python -m pii_anon.assurance verify --report <published> --against <rerun>`. "
                "A synthetic mirror corpus is shipped for a real-data run so the "
                "methodology is reproducible end-to-end on non-real data; re-running on "
                "that shipped mirror reproduces the methodology and, being already "
                "non-real, regenerates no further mirror."
            ),
        }


def capture_environment() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _hfield(h: Any, raw: bytes) -> None:
    """Length-prefixed field update — no in-band byte can shift a field boundary."""
    h.update(len(raw).to_bytes(8, "big"))
    h.update(raw)


def output_hash(pred_by_record: Sequence[Sequence[tuple[str, int, int]]]) -> str:
    """A content hash of a pipeline's spans over the dataset — a strong, data-free
    descriptor proving which pipeline produced the run (and its determinism).

    Every field is length-prefixed so an entity type containing ':' / '|' / record
    separators cannot collide two distinct detectors to the same digest."""
    h = hashlib.sha256()
    for spans in pred_by_record:
        _hfield(h, len(spans).to_bytes(4, "big"))
        for etype, start, end in spans:
            _hfield(h, etype.encode("utf-8", "replace"))
            _hfield(h, int(start).to_bytes(8, "big", signed=True))
            _hfield(h, int(end).to_bytes(8, "big", signed=True))
    return h.hexdigest()


def transform_hash(outputs: Sequence[str | None]) -> str:
    """A content hash of a pipeline's TRANSFORM outputs over the dataset.

    Data-free: hashes the transformed text (which the pipeline already deems safe to
    emit) one-way; only the digest is recorded. A distinct type tag separates an
    absent transform (None) from any real string output (so they cannot collide)."""
    h = hashlib.sha256()
    for out in outputs:
        if out is None:
            _hfield(h, b"\x00none")
        else:
            _hfield(h, b"\x01str")
            _hfield(h, out.encode("utf-8", "replace"))
    return h.hexdigest()


def build_provenance(
    *,
    dataset_name: str,
    dataset_fingerprint: str,
    seed: int,
    pii_anon_version: str,
    user_pipeline_name: str,
    user_pipeline_version: str,
    user_pipeline_deterministic: bool,
    user_pipeline_output_hash: str | None,
    user_pipeline_transform_hash: str | None = None,
    full_dataset_fingerprint: str | None = None,
    sample_size: int | None = None,
    timestamp: str | None,
) -> Provenance:
    return Provenance(
        dataset_name=dataset_name,
        dataset_fingerprint=dataset_fingerprint,
        seed=seed,
        pii_anon_version=pii_anon_version,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        package_versions=capture_environment(),
        user_pipeline_name=user_pipeline_name,
        user_pipeline_version=user_pipeline_version,
        user_pipeline_deterministic=user_pipeline_deterministic,
        user_pipeline_output_hash=user_pipeline_output_hash,
        user_pipeline_transform_hash=user_pipeline_transform_hash,
        full_dataset_fingerprint=full_dataset_fingerprint,
        sample_size=sample_size,
        timestamp=timestamp,
        harness_close_certified=HARNESS_CLOSE_CERTIFIED,
        harness_close_reference=HARNESS_CLOSE_REFERENCE,
    )


def provenance_gate(prov: Provenance) -> tuple[bool, tuple[str, ...]]:
    """Is the run reproducible enough to permit a publishable (``MEASURED``) label?

    Fail CLOSED. Returns ``(ok, reasons)``; ``reasons`` non-empty only when blocked.
    """
    reasons: list[str] = []
    if not is_nonblank_str(prov.dataset_fingerprint):
        reasons.append("missing dataset fingerprint")
    if not is_nonblank_str(prov.pii_anon_version):
        reasons.append("missing pii-anon version")
    if not isinstance(prov.seed, int) or isinstance(prov.seed, bool):
        reasons.append("missing/invalid seed")
    # user pipeline descriptor: name + a BINDING descriptor (version, or a real content
    # hash of detect or transform output — a detect-less pipeline's output_hash is None,
    # so a transform-only pipeline must bind via its transform hash or an explicit version).
    if not is_nonblank_str(prov.user_pipeline_name):
        reasons.append("missing user-pipeline name")
    has_descriptor = (
        is_nonblank_str(prov.user_pipeline_version)
        or is_nonblank_str(prov.user_pipeline_output_hash or "")
        or is_nonblank_str(prov.user_pipeline_transform_hash or "")
    )
    if not has_descriptor:
        reasons.append("missing user-pipeline descriptor (version, output hash, or transform hash)")
    if prov.user_pipeline_deterministic is not True:
        reasons.append("user pipeline is non-deterministic: comparison capped at ADVISORY")
    return (not reasons, tuple(reasons))
