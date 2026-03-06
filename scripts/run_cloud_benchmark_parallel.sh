#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_cloud_benchmark_parallel.sh — Run the canonical benchmark with all
# profiles evaluated in parallel, then merge into a single report.
#
# Usage:
#   # Default (6 profiles in parallel):
#   bash scripts/run_cloud_benchmark_parallel.sh
#
#   # Custom options:
#   bash scripts/run_cloud_benchmark_parallel.sh \
#     --dataset pii_anon_benchmark_v1 \
#     --matrix src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json \
#     --artifacts-dir artifacts/benchmarks
#
# Environment variables:
#   PYTHON              — Python interpreter (default: python3)
#   EVAL_DATA_DIR       — Path to pii-anon-eval-data repo
#   BENCH_WORKERS       — Max parallel profiles (default: 6)
#
# How it works:
#   1. Extracts profile names from the use-case matrix JSON
#   2. Launches one run_competitor_benchmark.py per profile in parallel
#   3. Each writes a checkpoint file to the shared checkpoint directory
#   4. After all profiles complete, runs merge-only mode to produce
#      the final benchmark-results.json and other artifacts
#
# Speedup:
#   Sequential: 6 profiles × ~50-60 min = 300-360 min
#   Parallel:   ~50-75 min (limited by the slowest profile)
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Defaults (match run_cloud_benchmark.sh / run_publish_grade_suite.py)
# ---------------------------------------------------------------------------
DATASET="pii_anon_benchmark_v1"
MATRIX="src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json"
ARTIFACTS_DIR="artifacts/benchmarks"
WARMUP_SAMPLES=100
MEASURED_RUNS=3
DATASET_SOURCE="package-only"
MAX_PARALLEL="${BENCH_WORKERS:-6}"

# Flags (empty = disabled)
STRICT_RUNTIME="--strict-runtime"
REQUIRE_ALL_COMPETITORS="--require-all-competitors"
REQUIRE_NATIVE_COMPETITORS="--require-native-competitors"
INCLUDE_END_TO_END="--include-end-to-end"
ALLOW_CORE_NATIVE_ENGINES="--allow-core-native-engines"
ENGINE_TIERS="auto minimal standard full"
ENFORCE_FLOORS=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)              DATASET="$2";               shift 2 ;;
        --matrix)               MATRIX="$2";                shift 2 ;;
        --artifacts-dir)        ARTIFACTS_DIR="$2";         shift 2 ;;
        --warmup-samples)       WARMUP_SAMPLES="$2";        shift 2 ;;
        --measured-runs)        MEASURED_RUNS="$2";          shift 2 ;;
        --dataset-source)       DATASET_SOURCE="$2";        shift 2 ;;
        --max-parallel)         MAX_PARALLEL="$2";          shift 2 ;;
        --strict-runtime)       STRICT_RUNTIME="$1";        shift ;;
        --no-strict-runtime)    STRICT_RUNTIME="";          shift ;;
        --require-all-competitors)
            REQUIRE_ALL_COMPETITORS="$1"; shift ;;
        --no-require-all-competitors)
            REQUIRE_ALL_COMPETITORS=""; shift ;;
        --require-native-competitors)
            REQUIRE_NATIVE_COMPETITORS="$1"; shift ;;
        --no-require-native-competitors)
            REQUIRE_NATIVE_COMPETITORS=""; shift ;;
        --no-include-end-to-end)
            INCLUDE_END_TO_END="--no-include-end-to-end"; shift ;;
        --no-allow-core-native-engines)
            ALLOW_CORE_NATIVE_ENGINES="--no-allow-core-native-engines"; shift ;;
        --engine-tiers)
            shift; ENGINE_TIERS=""
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENGINE_TIERS="$ENGINE_TIERS $1"; shift
            done
            ENGINE_TIERS="${ENGINE_TIERS# }"  # trim leading space
            ;;
        --enforce-floors)       ENFORCE_FLOORS="--enforce-floors"; shift ;;
        --max-samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        *)  echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

CHECKPOINT_DIR="${ARTIFACTS_DIR}/checkpoints/${DATASET}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }
err()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
info() { printf '\033[1;33m    %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
log "Preflight checks"

command -v "$PYTHON" >/dev/null 2>&1 || err "Python not found: $PYTHON"
[ -d "$REPO_ROOT/src/pii_anon" ] || err "Not in pii-anon-code repo root"
[ -f "$REPO_ROOT/$MATRIX" ] || err "Matrix file not found: $MATRIX"

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
echo "  Dataset: $DATASET"
echo "  Matrix: $MATRIX"
echo "  Artifacts: $ARTIFACTS_DIR"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Max parallel: $MAX_PARALLEL"
echo "  Engine tiers: $ENGINE_TIERS"

# ---------------------------------------------------------------------------
# Extract profile names from matrix
# ---------------------------------------------------------------------------
log "Extracting profile names from matrix"

PROFILES=$("$PYTHON" -c "
import json, sys
with open('$REPO_ROOT/$MATRIX') as f:
    data = json.load(f)
profiles = [p['profile'] for p in data.get('profiles', [])]
print(' '.join(profiles))
")

read -ra PROFILE_ARRAY <<< "$PROFILES"
PROFILE_COUNT=${#PROFILE_ARRAY[@]}
echo "  Found $PROFILE_COUNT profiles: $PROFILES"

if [ "$PROFILE_COUNT" -lt 1 ]; then
    err "No profiles found in matrix"
fi

ACTUAL_PARALLEL=$(( PROFILE_COUNT < MAX_PARALLEL ? PROFILE_COUNT : MAX_PARALLEL ))
echo "  Running $ACTUAL_PARALLEL profiles concurrently"

# ---------------------------------------------------------------------------
# Create directories
# ---------------------------------------------------------------------------
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$ARTIFACTS_DIR/logs"

# ---------------------------------------------------------------------------
# Build common flags for run_competitor_benchmark.py
# ---------------------------------------------------------------------------
COMMON_FLAGS=(
    "--dataset" "$DATASET"
    "--matrix" "$REPO_ROOT/$MATRIX"
    "--dataset-source" "$DATASET_SOURCE"
    "--warmup-samples" "$WARMUP_SAMPLES"
    "--measured-runs" "$MEASURED_RUNS"
    "--checkpoint-dir" "$CHECKPOINT_DIR"
)
if [ -n "$ENGINE_TIERS" ]; then
    COMMON_FLAGS+=("--engine-tiers" $ENGINE_TIERS)
fi
if [ -n "$STRICT_RUNTIME" ]; then
    COMMON_FLAGS+=("$STRICT_RUNTIME")
fi
if [ -n "$REQUIRE_ALL_COMPETITORS" ]; then
    COMMON_FLAGS+=("$REQUIRE_ALL_COMPETITORS")
fi
if [ -n "$REQUIRE_NATIVE_COMPETITORS" ]; then
    COMMON_FLAGS+=("$REQUIRE_NATIVE_COMPETITORS")
fi
COMMON_FLAGS+=("$INCLUDE_END_TO_END" "$ALLOW_CORE_NATIVE_ENGINES")
if [ -n "${MAX_SAMPLES:-}" ]; then
    COMMON_FLAGS+=("--max-samples" "$MAX_SAMPLES")
fi

# ---------------------------------------------------------------------------
# Launch parallel profile evaluations
# ---------------------------------------------------------------------------
log "Starting parallel profile evaluation"
echo "  Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  Per-profile logs: $ARTIFACTS_DIR/logs/<profile>.log"
echo ""

BENCHMARK_START=$(date +%s)
declare -A PROFILE_PIDS   # pid -> profile name
FAILED=()

for ((i=0; i<PROFILE_COUNT; i+=ACTUAL_PARALLEL)); do
    BATCH_END=$(( i + ACTUAL_PARALLEL ))
    if [ "$BATCH_END" -gt "$PROFILE_COUNT" ]; then
        BATCH_END=$PROFILE_COUNT
    fi

    log "Launching batch: profiles $((i+1))-${BATCH_END} of $PROFILE_COUNT"

    BATCH_PIDS=()
    for ((j=i; j<BATCH_END; j++)); do
        PROFILE="${PROFILE_ARRAY[$j]}"
        PROFILE_LOG="$ARTIFACTS_DIR/logs/${PROFILE}.log"

        info "[$((j+1))/$PROFILE_COUNT] $PROFILE -> $PROFILE_LOG"

        # Each profile writes to its own log file.  The aggregate
        # progress reporter (launched below) reads all log files and
        # renders a single unified progress bar to stdout.
        "$PYTHON" "$REPO_ROOT/scripts/run_competitor_benchmark.py" \
            "${COMMON_FLAGS[@]}" \
            --profiles "$PROFILE" \
            --output-json "$ARTIFACTS_DIR/profile-${PROFILE}.json" \
            --output-csv "$ARTIFACTS_DIR/profile-${PROFILE}.csv" \
            --output-floor-report "$ARTIFACTS_DIR/profile-${PROFILE}-floor.md" \
            > "$PROFILE_LOG" 2>&1 &

        PID=$!
        BATCH_PIDS+=("$PID")
        PROFILE_PIDS[$PID]="$PROFILE"
        info "  PID=$PID"
    done

    # Start the aggregate progress reporter for this batch.
    # It tails all log files and prints a single unified progress bar
    # every 60 seconds (same format as the sequential _ProgressReporter).
    "$PYTHON" "$REPO_ROOT/scripts/aggregate_progress.py" \
        "$ARTIFACTS_DIR/logs/" --interval 60 &
    AGGREGATOR_PID=$!

    # Wait for all profile processes in this batch
    for PID in "${BATCH_PIDS[@]}"; do
        PNAME="${PROFILE_PIDS[$PID]}"
        if wait "$PID"; then
            info "$PNAME (PID=$PID): completed successfully"
        else
            EXIT_CODE=$?
            info "$PNAME (PID=$PID): FAILED (exit code $EXIT_CODE)"
            FAILED+=("$PNAME")
        fi
    done

    # Stop the aggregator now that the batch is done
    kill "$AGGREGATOR_PID" 2>/dev/null || true
    wait "$AGGREGATOR_PID" 2>/dev/null || true
done

PROFILES_END=$(date +%s)
PROFILE_ELAPSED=$(( PROFILES_END - BENCHMARK_START ))

log "Profile evaluation phase complete"
echo "  Duration: $(( PROFILE_ELAPSED / 60 ))m $(( PROFILE_ELAPSED % 60 ))s"
echo "  Successful: $(( PROFILE_COUNT - ${#FAILED[@]} )) / $PROFILE_COUNT"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED: ${FAILED[*]}"
    echo "  Check full logs in $ARTIFACTS_DIR/logs/"
fi

# Verify checkpoints exist
CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -name "checkpoint_*.json" 2>/dev/null | wc -l)
echo "  Checkpoint files: $CHECKPOINT_COUNT"

if [ "$CHECKPOINT_COUNT" -lt 1 ]; then
    err "No checkpoint files found — all profile evaluations may have failed"
fi

# ---------------------------------------------------------------------------
# Merge phase: combine checkpoints into final report
# ---------------------------------------------------------------------------
log "Starting merge phase"

MERGE_LOG="$ARTIFACTS_DIR/logs/merge.log"

MERGE_CMD=(
    "$PYTHON" "$REPO_ROOT/scripts/run_competitor_benchmark.py"
    "--merge-only"
    "--dataset" "$DATASET"
    "--matrix" "$REPO_ROOT/$MATRIX"
    "--dataset-source" "$DATASET_SOURCE"
    "--warmup-samples" "$WARMUP_SAMPLES"
    "--measured-runs" "$MEASURED_RUNS"
    "--checkpoint-dir" "$CHECKPOINT_DIR"
    "--output-json" "$ARTIFACTS_DIR/benchmark-results.json"
    "--output-csv" "$ARTIFACTS_DIR/benchmark-raw.csv"
    "--output-floor-report" "$ARTIFACTS_DIR/floor-gate-report.md"
    "--output-diagnostics" "$ARTIFACTS_DIR/benchmark-diagnostics.json"
)
if [ -n "$ENGINE_TIERS" ]; then
    MERGE_CMD+=("--engine-tiers" $ENGINE_TIERS)
fi
if [ -n "$REQUIRE_ALL_COMPETITORS" ]; then
    MERGE_CMD+=("$REQUIRE_ALL_COMPETITORS")
fi
MERGE_CMD+=("$INCLUDE_END_TO_END" "$ALLOW_CORE_NATIVE_ENGINES")

info "Running merge: ${MERGE_CMD[*]}"
info "Merge log: $MERGE_LOG"

MERGE_OK=true
if ! "${MERGE_CMD[@]}" > "$MERGE_LOG" 2>&1; then
    MERGE_OK=false
    info "Merge phase FAILED (see $MERGE_LOG)"
fi

BENCHMARK_END=$(date +%s)
TOTAL_ELAPSED=$(( BENCHMARK_END - BENCHMARK_START ))
MERGE_ELAPSED=$(( BENCHMARK_END - PROFILES_END ))

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log "Parallel benchmark complete"
echo ""
echo "  Timing:"
echo "    Profile evaluation: $(( PROFILE_ELAPSED / 60 ))m $(( PROFILE_ELAPSED % 60 ))s"
echo "    Merge phase:        $(( MERGE_ELAPSED / 60 ))m $(( MERGE_ELAPSED % 60 ))s"
echo "    Total:              $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo ""
echo "  Artifacts:"
echo "    $ARTIFACTS_DIR/benchmark-results.json     (final report)"
echo "    $ARTIFACTS_DIR/benchmark-raw.csv          (system metrics)"
echo "    $ARTIFACTS_DIR/benchmark-diagnostics.json  (per-entity analysis)"
echo "    $ARTIFACTS_DIR/floor-gate-report.md        (floor gate results)"
echo "    $ARTIFACTS_DIR/logs/                       (per-profile logs)"
echo ""

if [ "$MERGE_OK" = false ]; then
    err "Merge phase failed — see $MERGE_LOG"
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    err "Some profile evaluations failed: ${FAILED[*]}"
fi

echo "  All $PROFILE_COUNT profiles evaluated and merged successfully."
