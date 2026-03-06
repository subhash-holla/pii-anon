#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from importlib import metadata
from pathlib import Path
from typing import Any

COMPETITORS: dict[str, dict[str, str]] = {
    "presidio": {
        "package": "presidio-analyzer",
        "citation_url": "https://github.com/microsoft/presidio",
    },
    "scrubadub": {
        "package": "scrubadub",
        "citation_url": "https://github.com/LeapBeyond/scrubadub",
    },
    "gliner": {
        "package": "gliner",
        "citation_url": "https://github.com/urchade/GLiNER",
    },
}


def _license_from_metadata(info: Any) -> tuple[str | None, str | None]:
    get_all = getattr(info, "get_all", None)
    classifiers = get_all("Classifier") or [] if callable(get_all) else []
    for item in classifiers:
        if "License ::" in item:
            return item.rsplit("::", 1)[-1].strip(), "classifier"

    expr = str(info.get("License-Expression", "")).strip()
    if expr:
        return expr, "spdx"

    raw = str(info.get("License", "")).strip()
    if raw:
        return raw, "license-field"

    return None, None


def _qualify(license_name: str | None, license_source: str | None) -> tuple[bool, str]:
    if not license_name or not license_source:
        return False, "license metadata missing"

    if license_source == "classifier":
        # Classifier provides OSI evidence in standard package metadata.
        return True, "OSI classifier evidence"

    compact = license_name.upper()
    if any(token in compact for token in ("PROPRIETARY", "COMMERCIAL", "UNLICENSED")):
        return False, f"unqualified license expression: {license_name}"
    return True, f"{license_source} license evidence"


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and qualify configured OSS competitors")
    parser.add_argument("--output-json", default="artifacts/benchmarks/competitor-qualification.json")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for system, meta in COMPETITORS.items():
        package = meta["package"]
        try:
            info = metadata.metadata(package)
        except Exception as exc:
            rows.append(
                {
                    "system": system,
                    "package": package,
                    "qualification_status": "unavailable",
                    "qualified": False,
                    "reason": f"metadata unavailable: {exc}",
                    "license_name": None,
                    "license_source": None,
                    "citation_url": meta["citation_url"],
                }
            )
            continue

        license_name, license_source = _license_from_metadata(info)
        qualified, reason = _qualify(license_name, license_source)
        rows.append(
            {
                "system": system,
                "package": package,
                "qualification_status": "qualified" if qualified else "excluded",
                "qualified": qualified,
                "reason": reason,
                "license_name": license_name,
                "license_source": license_source,
                "citation_url": meta["citation_url"],
            }
        )

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"competitors": rows, "qualified_count": sum(1 for row in rows if row["qualified"])}
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
