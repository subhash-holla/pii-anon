"""CLI: ``python -m pii_anon.assurance --dataset … --pipeline-entrypoint mod:func``."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Any

from .adapters import PipelineAdapter
from .config import ALL_DIMENSIONS, ALL_OUTPUTS, AssuranceConfig
from .runner import run_assurance_report


def _resolve(path: str) -> Any:
    """Resolve a ``module:function`` reference to the callable."""
    if ":" not in path:
        raise ValueError(f"entrypoint must be 'module:function', got {path!r}")
    mod_name, func_name = path.split(":", 1)
    module = importlib.import_module(mod_name)
    return getattr(module, func_name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m pii_anon.assurance",
        description="Generate a PII Handling Assurance Report comparing your pipeline to pii-anon.",
    )
    p.add_argument("--dataset", required=True, help="path to a JSONL dataset")
    p.add_argument("--pipeline-name", default="user-pipeline")
    p.add_argument("--pipeline-version", default="unknown")
    p.add_argument("--pipeline-entrypoint", help="detect callable, 'module:function' -> spans")
    p.add_argument("--transform-entrypoint", help="transform callable, 'module:function' -> text")
    p.add_argument("--non-deterministic", action="store_true",
                   help="declare the pipeline non-deterministic (caps results at ADVISORY)")
    p.add_argument("--dimensions", default="detection,leakage",
                   help=f"comma-separated; any of {','.join(ALL_DIMENSIONS)}")
    p.add_argument("--outputs", default="json,markdown",
                   help=f"comma-separated; any of {','.join(ALL_OUTPUTS)}")
    p.add_argument("--out", default="./assurance-out", help="output directory")
    p.add_argument("--sample-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260617)
    p.add_argument("--swarm", action="store_true", help="compare against the pii-anon swarm offering")
    return p


def _verify_main(argv: Sequence[str]) -> int:
    from .verify import verify_report_files

    p = argparse.ArgumentParser(
        prog="python -m pii_anon.assurance verify",
        description="Independently verify a published report reproduces a fresh re-run "
        "(matches the data-bound provenance hashes). PII-free — compares only hashes.",
    )
    p.add_argument("--report", required=True, help="the published report.json")
    p.add_argument("--against", required=True, help="a freshly re-run report.json to compare against")
    args = p.parse_args(argv)
    result = verify_report_files(args.report, args.against)
    print("REPRODUCED" if result.reproduced else "NOT REPRODUCED")
    for line in result.details:
        print(f"  {line}")
    return 0 if result.reproduced else 1


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "verify":
        return _verify_main(raw[1:])
    args = build_parser().parse_args(raw)
    if not args.pipeline_entrypoint and not args.transform_entrypoint:
        print("error: at least one of --pipeline-entrypoint / --transform-entrypoint is required",
              file=sys.stderr)
        return 2
    adapter = PipelineAdapter(
        name=args.pipeline_name,
        detect=_resolve(args.pipeline_entrypoint) if args.pipeline_entrypoint else None,
        transform=_resolve(args.transform_entrypoint) if args.transform_entrypoint else None,
        version=args.pipeline_version,
        deterministic=not args.non_deterministic,
    )
    config = AssuranceConfig(
        dataset=args.dataset,
        pipeline=adapter,
        dimensions=tuple(d.strip() for d in args.dimensions.split(",") if d.strip()),
        outputs=tuple(o.strip() for o in args.outputs.split(",") if o.strip()),
        out_dir=args.out,
        sample_size=args.sample_size,
        seed=args.seed,
        compare_swarm=args.swarm,
    )
    report = run_assurance_report(config)
    print(f"Assurance report written to {args.out}/")
    print(f"  dataset: {report.dataset_name} ({report.n_records} records, mode={report.mode})")
    print(f"  systems: {', '.join(report.systems)} (baseline {report.baseline})")
    for dim, by_system in report.dimensions.items():
        for system, res in by_system.items():
            val = "—" if res.value is None else f"{res.value:.4f}"
            print(f"  {dim:<12} {system:<18} {val:>8}  [{res.claim_strength.value}]")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
