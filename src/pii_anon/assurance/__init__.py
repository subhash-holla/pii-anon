"""PII Handling Assurance Report.

Run a user's own PII pipeline AND pii-anon's own offering on the user's own
dataset, and produce an audit-grade, advertisable comparison report whose
load-bearing property is honesty under adversarial reading: every number is
either provably defensible (``MEASURED``) or visibly hedged (``ADVISORY`` /
``NOT_ASSESSABLE``), and the report structurally cannot overstate either system.

Design of record: ``docs/superpowers/specs/2026-06-17-pii-assurance-report-design.md``.

Heavy detection/transform machinery is imported only inside functions, so
importing this package pulls no models.

Example
-------
    from pii_anon.assurance import run_assurance_report, AssuranceConfig, PipelineAdapter

    report = run_assurance_report(AssuranceConfig(
        dataset="my_data.jsonl",
        pipeline=PipelineAdapter(detect=my_detect, transform=my_transform, name="acme-dlp"),
        dimensions=("detection", "leakage"),
        outputs=("json", "markdown"),
        out_dir="./assurance-out",
    ))
"""

from __future__ import annotations

from .adapters import Capabilities, PipelineAdapter, pii_anon_adapter
from .claim_strength import ClaimStrength, DimensionResult, MeasurementMode
from .config import AssuranceConfig
from .report import AssuranceReport, NoFabricationError
from .runner import run_assurance_report
from .verify import VerificationResult, verify_report_files, verify_reproduction

__all__ = [
    "AssuranceConfig",
    "AssuranceReport",
    "Capabilities",
    "ClaimStrength",
    "DimensionResult",
    "MeasurementMode",
    "NoFabricationError",
    "PipelineAdapter",
    "VerificationResult",
    "pii_anon_adapter",
    "run_assurance_report",
    "verify_report_files",
    "verify_reproduction",
]
