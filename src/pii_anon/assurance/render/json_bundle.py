"""JSON + reproducibility-bundle renderer."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any


def render_json_bundle(
    report_dict: dict[str, Any],
    *,
    mirror_rows: Sequence[dict[str, Any]],
) -> dict[str, str]:
    """Return the machine-readable report + the reproducibility bundle files."""
    files: dict[str, str] = {}
    # allow_nan=False: a NaN/inf that slipped past the numeric gate (e.g. an un-gated
    # top-level/provenance number) raises rather than emitting invalid JSON (`NaN`).
    files["report.json"] = json.dumps(
        report_dict, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    files["repro-bundle/provenance.json"] = json.dumps(
        report_dict["provenance"], indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
    )
    files["repro-bundle/synthetic-mirror.jsonl"] = (
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in mirror_rows) + "\n"
        if mirror_rows else ""
    )
    methodology = report_dict.get("methodology", [])
    limitations = report_dict.get("limitations", [])
    files["repro-bundle/methodology.md"] = (
        "# Methodology\n\n"
        + ("\n".join(f"- {m}" for m in methodology) or "_none_")
        + "\n\n# Limitations\n\n"
        + ("\n".join(f"- {limit}" for limit in limitations) or "_none_")
        + "\n"
    )
    return files
