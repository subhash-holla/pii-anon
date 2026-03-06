from __future__ import annotations

import os
import sys
from pathlib import Path


# tests/ lives inside pii-anon-code/, so the repo root is one level up.
_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parent

SRC_PATH = _REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Set working directory to the repo root so tests that use relative paths
# (e.g., "README.md", "scripts/...", "docs/...") resolve correctly.
os.chdir(str(_REPO_ROOT))
