#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pii_anon.evaluation.runtime_preflight import run_benchmark_runtime_preflight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate benchmark runtime prerequisites for canonical publish-grade runs"
    )
    parser.add_argument(
        "--strict-runtime",
        action="store_true",
        help="Require linux runtime with shared-memory support",
    )
    parser.add_argument(
        "--require-all-competitors",
        action="store_true",
        help="Require all configured competitors to be available",
    )
    parser.add_argument(
        "--require-native-competitors",
        action="store_true",
        help="Require native competitor initialization readiness (no proxy/fallback paths)",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write full JSON diagnostics",
    )
    args = parser.parse_args()

    report = run_benchmark_runtime_preflight(
        strict_runtime=bool(args.strict_runtime),
        require_all_competitors=bool(args.require_all_competitors),
        require_native_competitors=bool(args.require_native_competitors),
    )

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {output}")

    if not bool(report.get("ready", False)):
        failures = report.get("failures", [])
        if isinstance(failures, list) and failures:
            raise SystemExit("runtime preflight failed: " + "; ".join(str(item) for item in failures))
        raise SystemExit("runtime preflight failed")


if __name__ == "__main__":
    main()

