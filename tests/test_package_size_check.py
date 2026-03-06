from __future__ import annotations

import subprocess
import sys


def test_package_size_check_passes_for_small_wheel(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "pkg-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"x" * 1024)  # 1 KB

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_package_size.py",
            "--dist-dir",
            str(dist),
            "--max-wheel-mb",
            "1.5",
            "--package-name",
            "pkg",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_package_size_check_fails_for_large_wheel(tmp_path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "pkg-1.0.0-py3-none-any.whl"
    wheel.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB

    result = subprocess.run(
        [
            sys.executable,
            "scripts/check_package_size.py",
            "--dist-dir",
            str(dist),
            "--max-wheel-mb",
            "1.5",
            "--package-name",
            "pkg",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
