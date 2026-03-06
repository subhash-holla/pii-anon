#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check wheel size threshold")
    parser.add_argument("--dist-dir", default="dist")
    parser.add_argument("--max-wheel-mb", type=float, default=1.5)
    parser.add_argument(
        "--package-name",
        default="pii_anon",
        help="Normalized package name prefix to validate (hyphen/underscore insensitive)",
    )
    args = parser.parse_args()

    dist_dir = Path(args.dist_dir)
    normalized = args.package_name.replace("-", "_")
    wheels = sorted(
        wheel
        for wheel in dist_dir.glob("*.whl")
        if wheel.name.replace("-", "_").startswith(f"{normalized}_")
    )
    if not wheels:
        raise SystemExit(f"No wheel files found in {dist_dir} for package `{args.package_name}`")

    wheel = wheels[-1]
    size_mb = wheel.stat().st_size / (1024 * 1024)
    if size_mb > args.max_wheel_mb:
        raise SystemExit(
            f"Wheel {wheel.name} exceeds threshold: {size_mb:.3f} MB > {args.max_wheel_mb:.3f} MB"
        )

    print(f"Wheel {wheel.name} size OK: {size_mb:.3f} MB <= {args.max_wheel_mb:.3f} MB")


if __name__ == "__main__":
    main()
