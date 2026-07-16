"""External reproduction verification.

The in-process determinism probe (``runner._build_substrate``) cannot observe
CROSS-process non-determinism (e.g. a ``PYTHONHASHSEED``-dependent pipeline). The
binding guarantee behind a ``MEASURED`` claim is therefore an INDEPENDENT re-run:
a holder of the original dataset re-runs the report (ideally in a fresh process /
environment) and confirms the recorded, data-bound provenance hashes match.

This module implements that mode (the design §18 / provenance "external
verification" the report text references). It compares ONLY one-way hashes
(``dataset_fingerprint`` + ``output_hash`` + ``transform_output_hash``) — never raw
data — so verification is itself PII-free. A mismatch means the published numbers
are NOT independently reproducible (fail closed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    reproduced: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"reproduced": self.reproduced, "checks": dict(self.checks), "details": list(self.details)}


def _hashes(report: Any) -> dict[str, Any]:
    prov = report.get("provenance") if isinstance(report, dict) else None
    if not isinstance(prov, dict):
        prov = {}
    up = prov.get("user_pipeline")
    if not isinstance(up, dict):
        up = {}
    return {
        "dataset_fingerprint": prov.get("dataset_fingerprint"),
        "output_hash": up.get("output_hash"),
        "transform_output_hash": up.get("transform_output_hash"),
    }


def verify_reproduction(published: dict[str, Any], rerun: dict[str, Any]) -> VerificationResult:
    """Compare two report dicts' data-bound provenance hashes.

    A hash matches only when it is present (non-null) in BOTH and equal — a missing
    descriptor never counts as a match (fail closed)."""
    a, b = _hashes(published), _hashes(rerun)
    checks: dict[str, bool] = {}
    details: list[str] = []
    for k in a:
        ok = a[k] is not None and a[k] == b[k]
        checks[k] = ok
        av = (a[k][:16] + "…") if isinstance(a[k], str) else a[k]
        bv = (b[k][:16] + "…") if isinstance(b[k], str) else b[k]
        details.append(f"{k}: {'MATCH' if ok else 'MISMATCH'} ({av} vs {bv})")
    return VerificationResult(reproduced=all(checks.values()), checks=checks, details=tuple(details))


def verify_report_files(published_path: str | Path, rerun_path: str | Path) -> VerificationResult:
    """Verify two on-disk ``report.json`` files reproduce each other.

    Fail CLOSED on any unreadable/malformed input (missing file, bad JSON, or a deeply
    nested doc that raises ``RecursionError`` rather than ``JSONDecodeError``) — return
    NOT REPRODUCED with a clean reason rather than a raw traceback."""
    try:
        published = json.loads(Path(published_path).read_text(encoding="utf-8"))
        rerun = json.loads(Path(rerun_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError) as exc:
        return VerificationResult(reproduced=False, details=(f"could not read/parse a report ({type(exc).__name__})",))
    return verify_reproduction(published, rerun)
