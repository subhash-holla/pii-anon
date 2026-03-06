#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_cloud_benchmark.sh — Bootstrap and run the canonical benchmark on a
# fresh cloud VM (Linux ARM64 or x86_64).
#
# Usage:
#   # On the cloud VM (after cloning the repos):
#   bash scripts/run_cloud_benchmark.sh
#
#   # With custom eval-data path:
#   EVAL_DATA_DIR=/path/to/pii-anon-eval-data bash scripts/run_cloud_benchmark.sh
#
# Prerequisites:
#   - Python 3.10+ installed
#   - git clone of pii-anon-code (this repo)
#   - git clone of pii-anon-eval-data (sibling directory by default)
#
# The script:
#   1. Creates a virtual environment and installs all dependencies
#   2. Downloads NLP models (spaCy, GLiNER)
#   3. Runs the full canonical publish-grade benchmark suite
#   4. Copies artifacts to artifacts/benchmarks/
#
# Environment variables:
#   EVAL_DATA_DIR   — Path to pii-anon-eval-data repo (default: ../pii-anon-eval-data)
#   BENCH_WORKERS   — Override max parallel workers (default: auto-detected)
#   PYTHON          — Python interpreter to use (default: python3)
#
# Flags:
#   --parallel      — Evaluate all 6 profiles in parallel (requires >=64 GB RAM).
#                     Uses run_cloud_benchmark_parallel.sh under the hood.
#                     Typical speedup: ~5-6× (60 min vs 360 min).
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$REPO_ROOT/../pii-anon-eval-data}"
PYTHON="${PYTHON:-python3}"
PARALLEL_MODE=false

# Check for --parallel flag
for arg in "$@"; do
    if [ "$arg" = "--parallel" ]; then
        PARALLEL_MODE=true
    fi
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }
err() { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
log "Preflight checks"

command -v "$PYTHON" >/dev/null 2>&1 || err "Python not found: $PYTHON"
PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$("$PYTHON" -c "import sys; print(sys.version_info.major)")
PY_MINOR=$("$PYTHON" -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
    err "Python 3.10+ required, found $PY_VERSION"
fi
echo "  Python: $PYTHON ($PY_VERSION)"

[ -d "$REPO_ROOT/src/pii_anon" ] || err "Not in pii-anon-code repo root: $REPO_ROOT"
echo "  Repo: $REPO_ROOT"

[ -d "$EVAL_DATA_DIR" ] || err "Eval-data repo not found: $EVAL_DATA_DIR (set EVAL_DATA_DIR)"
echo "  Eval data: $EVAL_DATA_DIR"

ARCH=$("$PYTHON" -c "import platform; print(platform.machine())")
OS=$("$PYTHON" -c "import platform; print(platform.system())")
CPUS=$("$PYTHON" -c "import os; print(os.cpu_count())")
MEM_GB=$("$PYTHON" -c "
import os
try:
    gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
except Exception:
    gb = -1
print(f'{gb:.1f}')
")
echo "  Platform: $OS $ARCH, ${CPUS} CPUs, ${MEM_GB} GB RAM"

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
VENV_DIR="$REPO_ROOT/.venv-cloud-bench"
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
else
    log "Reusing existing virtual environment at $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

log "Upgrading pip and build tools"
pip install --upgrade pip setuptools wheel build 2>&1 | tail -1

# ---------------------------------------------------------------------------
# Install pii-anon with all benchmark extras
# ---------------------------------------------------------------------------
log "Installing pii-anon with benchmark extras"
pip install -e "$REPO_ROOT[dev,cli,crypto,benchmark]" 2>&1 | tail -3

log "Installing pii-anon-eval-data (dataset package)"
pip install -e "$EVAL_DATA_DIR" 2>&1 | tail -1

# ---------------------------------------------------------------------------
# Download NLP models
# ---------------------------------------------------------------------------
log "Downloading NLP models"

echo "  spaCy en_core_web_sm..."
python -m spacy download en_core_web_sm 2>&1 | tail -1

echo "  GLiNER PII model..."
python -c "
import warnings
warnings.filterwarnings('ignore')
from gliner import GLiNER
GLiNER.from_pretrained('knowledgator/gliner-pii-base-v1.0')
print('  GLiNER model cached successfully')
"

# ---------------------------------------------------------------------------
# Run the canonical benchmark suite
# ---------------------------------------------------------------------------
log "Starting canonical benchmark suite"
echo "  Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Platform: $OS $ARCH, ${CPUS} CPUs, ${MEM_GB} GB RAM"
echo "  Parallel mode: $PARALLEL_MODE"
echo ""

BENCHMARK_START=$(date +%s)

if [ "$PARALLEL_MODE" = true ]; then
    # -----------------------------------------------------------------------
    # PARALLEL MODE: evaluate all profiles concurrently, then merge.
    # The publish-grade suite is NOT used in parallel mode — instead we
    # run the competitor benchmark directly via the parallel orchestrator,
    # then run post-benchmark steps (continuity, rendering) separately.
    # -----------------------------------------------------------------------
    log "Running parallel profile evaluation"
    bash "$REPO_ROOT/scripts/run_cloud_benchmark_parallel.sh" \
        --dataset pii_anon_benchmark_v1 \
        --matrix src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json \
        --artifacts-dir artifacts/benchmarks \
        --engine-tiers auto minimal standard full \
        --warmup-samples 100 \
        --measured-runs 3 \
        --dataset-source package-only \
        --strict-runtime \
        --require-all-competitors \
        --require-native-competitors

    log "Running post-benchmark steps (rendering, validation)"
    python "$REPO_ROOT/scripts/run_publish_grade_suite.py" \
        --reuse-current-env \
        --install-no-deps \
        --dataset pii_anon_benchmark_v1 \
        --engine-tiers auto minimal standard full \
        --matrix src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json \
        --artifacts-dir artifacts/benchmarks \
        --work-dir .publish-suite \
        --warmup-samples 100 \
        --measured-runs 3 \
        --dataset-source package-only \
        --strict-runtime \
        --require-all-competitors \
        --require-native-competitors \
        --include-end-to-end \
        --allow-core-native-engines \
        --enforce-floors \
        --enforce-publish-claims \
        --validate-readme-sync
else
    # -----------------------------------------------------------------------
    # SEQUENTIAL MODE (default): the publish-grade suite handles everything.
    # -----------------------------------------------------------------------
    python "$REPO_ROOT/scripts/run_publish_grade_suite.py" \
        --reuse-current-env \
        --install-no-deps \
        --dataset pii_anon_benchmark_v1 \
        --engine-tiers auto minimal standard full \
        --matrix src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json \
        --artifacts-dir artifacts/benchmarks \
        --work-dir .publish-suite \
        --warmup-samples 100 \
        --measured-runs 3 \
        --dataset-source package-only \
        --strict-runtime \
        --require-all-competitors \
        --require-native-competitors \
        --include-end-to-end \
        --allow-core-native-engines \
        --enforce-floors \
        --enforce-publish-claims \
        --validate-readme-sync
fi

BENCHMARK_END=$(date +%s)
ELAPSED=$(( BENCHMARK_END - BENCHMARK_START ))
HOURS=$(( ELAPSED / 3600 ))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

log "Benchmark complete!"
echo "  End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Total duration: ${HOURS}h ${MINUTES}m"
echo "  Artifacts: $REPO_ROOT/artifacts/benchmarks/"
echo ""
echo "  Key files:"
echo "    benchmark-results.json  - Full benchmark report"
echo "    benchmark-summary.md    - Human-readable summary"
echo "    floor-gate-report.md    - Floor gate analysis"
echo ""
echo "  To download artifacts from the cloud VM:"
echo "    scp -r <user>@<host>:$(realpath "$REPO_ROOT/artifacts/benchmarks/") ."
