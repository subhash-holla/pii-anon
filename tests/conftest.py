from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# tests/ lives inside pii-anon-code/, so the repo root is one level up.
_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parent

SRC_PATH = _REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Set working directory to the repo root so tests that use relative paths
# (e.g., "README.md", "scripts/...", "docs/...") resolve correctly.
os.chdir(str(_REPO_ROOT))


# ── Dataset availability detection ──────────────────────────────────────
# pii-anon-datasets is a separate package that may not be installed in CI.
# Tests that require the benchmark dataset should use the `requires_dataset`
# marker so they skip gracefully rather than fail with FileNotFoundError.

def _dataset_available() -> bool:
    """Check if pii-anon-datasets is installed and the benchmark dataset exists."""
    try:
        from pii_anon.benchmarks.datasets import resolve_benchmark_dataset_path
        return resolve_benchmark_dataset_path("pii_anon_benchmark") is not None
    except Exception:
        return False


_DATASET_INSTALLED = _dataset_available()

requires_dataset = pytest.mark.skipif(
    not _DATASET_INSTALLED,
    reason="pii-anon-datasets not installed (benchmark dataset unavailable)",
)
