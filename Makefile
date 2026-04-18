ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
PLATFORM := windows
else
VENV_PYTHON := .venv/bin/python
UNAME_S := $(shell uname -s 2>/dev/null)
ifeq ($(UNAME_S),Darwin)
PLATFORM := macos
else
PLATFORM := linux
endif
endif

PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)
PYTHON_WINDOWS ?= py -3
RUFF ?= $(PYTHON) -m ruff
MYPY ?= $(PYTHON) -m mypy
PYTEST ?= $(PYTHON) -m pytest
TWINE ?= $(PYTHON) -m twine
# Auto-detect Docker Desktop path on macOS when 'docker' is not on PATH.
# Also export the Docker Desktop bin dir so helpers like docker-credential-desktop are found.
DOCKER_DESKTOP_DIR := /Applications/Docker.app/Contents/Resources/bin
DOCKER_DESKTOP := $(DOCKER_DESKTOP_DIR)/docker
DOCKER ?= $(if $(shell command -v docker 2>/dev/null),docker,$(if $(wildcard $(DOCKER_DESKTOP)),$(DOCKER_DESKTOP),docker))
export PATH := $(if $(wildcard $(DOCKER_DESKTOP_DIR)),$(DOCKER_DESKTOP_DIR):$(PATH),$(PATH))
BENCH_IMAGE ?= pii-anon-bench:latest

EVAL_DATA_DIR ?= ../pii-anon-eval-data
BENCH_DATASET ?= pii_anon_benchmark
BENCH_MATRIX ?= src/pii_anon/benchmarks/matrix/use_case_matrix.json
BENCH_ARTIFACTS ?= artifacts/benchmarks
BENCH_WORKDIR ?= .publish-suite
BENCH_WARMUP ?= 100
BENCH_RUNS ?= 3

# Swarm training configuration
SWARM_DATASETS ?= pii_anon_eval
SWARM_MAX_RECORDS ?= 10000
SWARM_KFOLD ?= 5
SWARM_WORKERS ?= 4

PORTABLE_SUITE_FLAGS = \
	--dataset $(BENCH_DATASET) \
	--matrix $(BENCH_MATRIX) \
	--artifacts-dir $(BENCH_ARTIFACTS) \
	--work-dir $(BENCH_WORKDIR) \
	--warmup-samples $(BENCH_WARMUP) \
	--measured-runs $(BENCH_RUNS) \
	--reuse-current-env \
	--install-no-deps \
	--no-strict-runtime \
	--no-require-all-competitors \
	--no-require-native-competitors \
	--no-include-end-to-end \
	--no-allow-core-native-engines \
	--no-enforce-floors \
	--no-enforce-publish-claims \
	--no-validate-readme-sync \
	--dataset-source auto

CANONICAL_SUITE_FLAGS = \
	--dataset $(BENCH_DATASET) \
	--matrix $(BENCH_MATRIX) \
	--artifacts-dir $(BENCH_ARTIFACTS) \
	--work-dir $(BENCH_WORKDIR) \
	--warmup-samples $(BENCH_WARMUP) \
	--measured-runs $(BENCH_RUNS) \
	--dataset-source package-only \
	--strict-runtime \
	--require-all-competitors \
	--require-native-competitors \
	--include-end-to-end \
	--allow-core-native-engines \
	--no-enforce-floors \
	--no-enforce-publish-claims \
	--validate-readme-sync

.PHONY: bootstrap install-dev setup-swarm lint type test perf build twine-check package-size train-swarm benchmark-all benchmark-doctor benchmark-full benchmark compare-benchmark benchmark-preflight benchmark-publish-suite benchmark-portable benchmark-portable-macos benchmark-portable-linux benchmark-portable-windows benchmark-canonical benchmark-canonical-linux benchmark-canonical-macos benchmark-canonical-macos-native benchmark-canonical-cloud benchmark-canonical-windows benchmark-docker-build benchmark-native-setup readme-benchmark-check cli-smoke docs-smoke all

bootstrap:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[dev,cli,crypto,engines]'

install-dev: bootstrap

lint:
	$(RUFF) check src tests

type:
	$(MYPY) src/pii_anon

test:
	PYTHONPATH=src $(PYTEST)

perf:
	$(PYTEST) -m performance --no-cov

build:
	$(PYTHON) -m build --outdir dist

twine-check:
	$(TWINE) check dist/*

package-size:
	$(PYTHON) scripts/check_package_size.py --dist-dir dist --max-wheel-mb 1.5 --package-name pii_anon

benchmark-preflight:
	$(PYTHON) scripts/check_benchmark_runtime.py --strict-runtime --require-all-competitors --require-native-competitors --output-json artifacts/benchmarks/runtime-preflight.json

compare-benchmark:
	$(PYTHON) scripts/run_competitor_benchmark.py --dataset $(BENCH_DATASET) --dataset-source package-only --matrix $(BENCH_MATRIX) --strict-runtime --require-all-competitors --require-native-competitors --enforce-floors --output-json benchmark-results.json --output-csv benchmark-raw.csv --output-floor-report floor-gate-report.md --output-baseline artifacts/benchmarks/floor-baseline.json --preflight-output-json artifacts/benchmarks/runtime-preflight.json

benchmark:
	$(PYTHON) scripts/generate_benchmark_dataset.py
	$(PYTHON) scripts/check_benchmark_runtime.py --strict-runtime --require-all-competitors --require-native-competitors --output-json artifacts/benchmarks/runtime-preflight.json
	$(PYTHON) scripts/run_competitor_benchmark.py --dataset $(BENCH_DATASET) --dataset-source package-only --matrix $(BENCH_MATRIX) --strict-runtime --require-all-competitors --require-native-competitors --enforce-floors --output-json benchmark-results.json --output-csv benchmark-raw.csv --output-floor-report floor-gate-report.md --output-baseline artifacts/benchmarks/floor-baseline.json --preflight-output-json artifacts/benchmarks/runtime-preflight.json
	$(PYTHON) scripts/render_benchmark_summary.py --input-json benchmark-results.json --output-markdown docs/benchmark-summary.md --require-floor-pass

benchmark-publish-suite:
	$(PYTHON) scripts/run_publish_grade_suite.py --dataset $(BENCH_DATASET) --matrix $(BENCH_MATRIX) --artifacts-dir $(BENCH_ARTIFACTS)

benchmark-portable:
	$(PYTHON) scripts/run_publish_grade_suite.py $(PORTABLE_SUITE_FLAGS)

benchmark-portable-macos: benchmark-portable

benchmark-portable-linux: benchmark-portable

benchmark-portable-windows:
	$(PYTHON_WINDOWS) scripts/run_publish_grade_suite.py $(PORTABLE_SUITE_FLAGS)

benchmark-canonical: benchmark-canonical-linux

benchmark-canonical-linux:
	$(PYTHON) scripts/run_publish_grade_suite.py $(CANONICAL_SUITE_FLAGS)

# Build the benchmark Docker image once (slow, but cached for subsequent runs).
# Re-run only when pyproject.toml dependencies change.
benchmark-docker-build:
	$(DOCKER) build --platform linux/amd64 -f Dockerfile.benchmark -t $(BENCH_IMAGE) .

# Run the canonical benchmark inside the pre-built image (fast on repeat runs).
# Requires: make benchmark-docker-build (first time or after dependency changes)
#
# Competitor evaluations run in parallel inside a single Docker container.
# The --shm-size=4g flag provides sufficient shared memory for
# multiprocessing workers (spawn context).
# tqdm provides interactive progress bars; use TERM=xterm if they don't render.
benchmark-canonical-macos: benchmark-docker-build
	$(DOCKER) run --rm --platform linux/amd64 --shm-size=4g \
		-e PYTHONUNBUFFERED=1 -e TERM=$(TERM) \
		-e HF_HUB_DISABLE_PROGRESS_BARS=1 \
		-e TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
		-e TOKENIZERS_PARALLELISM=false \
		-v "$$PWD":/work -v "$$(cd $(EVAL_DATA_DIR) && pwd)":/eval-data -w /work $(BENCH_IMAGE) \
		bash -lc "pip install -q --no-cache-dir --no-deps -e /eval-data/ && pip install -q --no-cache-dir --no-deps -e '.[datasets]' && python scripts/run_publish_grade_suite.py --reuse-current-env --install-no-deps $(CANONICAL_SUITE_FLAGS)"

# Install all benchmark dependencies natively on macOS (no Docker).
# Run once before using benchmark-canonical-macos-native.
benchmark-native-setup:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e '.[dev,cli,crypto,benchmark]'
	$(PYTHON) -m pip install -e $(EVAL_DATA_DIR)
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -c "from gliner import GLiNER; GLiNER.from_pretrained('knowledgator/gliner-pii-base-v1.0')"

# Native macOS canonical benchmark — runs directly on Apple Silicon without
# Docker/Rosetta overhead (~2-3x faster than the Docker path).
# Requires: make benchmark-native-setup (first time or after dependency changes)
benchmark-canonical-macos-native:
	$(PYTHON) scripts/run_publish_grade_suite.py \
		--reuse-current-env --install-no-deps \
		$(CANONICAL_SUITE_FLAGS)

# Cloud benchmark — runs on a fresh Linux cloud VM (ARM64 or x86_64).
# Bootstraps the environment, installs all deps, and runs the full suite.
# Usage: bash scripts/run_cloud_benchmark.sh  (or: make benchmark-canonical-cloud)
benchmark-canonical-cloud:
	bash scripts/run_cloud_benchmark.sh

benchmark-canonical-windows:
	$(DOCKER) run --rm --platform linux/amd64 --shm-size=4g \
		-e PYTHONUNBUFFERED=1 -e TERM=$(TERM) \
		-e HF_HUB_DISABLE_PROGRESS_BARS=1 \
		-e TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
		-e TOKENIZERS_PARALLELISM=false \
		-v "$$PWD":/work -v "$$(cd $(EVAL_DATA_DIR) && pwd)":/eval-data -w /work $(BENCH_IMAGE) \
		bash -lc "pip install -q --no-cache-dir --no-deps -e /eval-data/ && pip install -q --no-cache-dir --no-deps -e '.[datasets]' && python scripts/run_publish_grade_suite.py --reuse-current-env --install-no-deps $(CANONICAL_SUITE_FLAGS)"

# ── High-Level Automation Targets ──────────────────────────────────────

# One-time setup for swarm training: install local packages + swarm deps.
# Run this before train-swarm on a fresh checkout.
#   make setup-swarm
#
# Platform notes:
#   - macOS: libomp is required by XGBoost.  Installed via Homebrew if the
#     user has it.  Otherwise the user is prompted to install it manually.
#   - Linux: XGBoost ships manylinux wheels with OpenMP bundled; no extra step.
#   - Windows: XGBoost ships with Visual C++ runtime; no extra step.
setup-swarm:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e $(EVAL_DATA_DIR)
ifeq ($(PLATFORM),macos)
	@command -v brew >/dev/null 2>&1 && brew list libomp >/dev/null 2>&1 \
		|| { command -v brew >/dev/null 2>&1 \
		     && { echo "Installing libomp (required by XGBoost on macOS)..."; brew install libomp; } \
		     || echo "WARNING: libomp missing and Homebrew not found. Install libomp manually or XGBoost will fail on import."; }
endif
	$(PYTHON) -m pip install -e '.[dev,cli,crypto,benchmark,datasets,swarm-ml,swarm-train]'
	$(PYTHON) -m spacy download en_core_web_sm
	$(PYTHON) -c "import stanza; stanza.download('en')"
	$(PYTHON) -c "from gliner import GLiNER; GLiNER.from_pretrained('knowledgator/gliner-pii-base-v1.0')"
	@echo ""
	@echo "Setup complete (platform: $(PLATFORM)).  Run:  make train-swarm"

# Train the swarm offering and deploy artifacts.
# Output: ~/.pii_anon/swarm/ (ds_params.json, temperature.json, etc.)
#
# Usage:
#   make train-swarm                              # 10K records, 5-fold CV
#   make train-swarm SWARM_MAX_RECORDS=0           # ALL records (unlimited)
#   make train-swarm SWARM_KFOLD=3                 # 3-fold CV (faster)
#   make train-swarm SWARM_KFOLD=10                # 10-fold CV (more robust)
#   make train-swarm SWARM_KFOLD=1                 # No CV (fastest, single holdout)
#   make train-swarm SWARM_DATASETS=pii_anon_eval,ai4privacy,conll2003  # multiple datasets
#   make train-swarm SWARM_DATASETS=pii_anon_eval,ai4privacy SWARM_MAX_RECORDS=0 SWARM_KFOLD=5
train-swarm:
	$(PYTHON) scripts/train_swarm.py --datasets $(SWARM_DATASETS) --max-records $(SWARM_MAX_RECORDS) --kfold $(SWARM_KFOLD) --workers $(SWARM_WORKERS)

# Run the full benchmark on M1 Mac and update all documentation.
# This is the single command to run before a release.
# Output: benchmark-results.json, updated README, updated docs/benchmark-summary.md
benchmark-full:
	$(PYTHON) scripts/run_full_benchmark.py --dataset $(BENCH_DATASET) --dataset-source auto

# ── Cross-platform community benchmark ─────────────────────────────────
# ``benchmark-all`` is the friendly entry-point for library users on any
# OS.  It runs the full suite, tolerates missing competitor engines
# (produces a partial leaderboard rather than failing), and updates both
# the benchmark-summary section AND the pii-rate-elo value section in
# the README.  Routed to the right underlying target based on the host
# platform:
#
#   Linux:    benchmark-all → benchmark-portable-linux
#   macOS:    benchmark-all → benchmark-portable-macos
#   Windows:  benchmark-all → benchmark-portable-windows
#
# For publish-grade canonical runs (strict competitor set, reproducible
# environment), use ``benchmark-canonical`` instead.
benchmark-all:
ifeq ($(PLATFORM),windows)
	@$(MAKE) benchmark-portable-windows
else ifeq ($(PLATFORM),macos)
	@$(MAKE) benchmark-portable-macos
else
	@$(MAKE) benchmark-portable-linux
endif
	$(PYTHON) scripts/render_pii_rate_elo_value.py \
		--input-json $(BENCH_ARTIFACTS)/benchmark-results.json \
		--readme README.md

# Print platform + environment diagnostics for users debugging a
# benchmark failure on an unfamiliar OS.  Always safe to run.
# Delegates to a Python script so the checks are portable and easy to
# test — the Makefile layer would otherwise need shell-specific
# ``try/except`` gymnastics that differ between POSIX sh and Windows cmd.
benchmark-doctor:
	@echo "Platform:        $(PLATFORM)"
	@echo "Python:          $(PYTHON)"
	@echo "Eval data dir:   $(EVAL_DATA_DIR)"
	@$(PYTHON) scripts/benchmark_doctor.py --eval-data-dir "$(EVAL_DATA_DIR)"

# ── Utility Targets ───────────────────────────────────────────────────

readme-benchmark-check:
	$(PYTHON) scripts/check_readme_benchmark.py --readme README.md --summary docs/benchmark-summary.md --report-json benchmark-results.json

cli-smoke:
	pii-anon version
	pii-anon health --output json

docs-smoke:
	jupyter nbconvert --to notebook --execute notebooks/llm_pipeline_quickstart.ipynb --output /tmp/llm_pipeline_quickstart.executed.ipynb

all: lint type test perf build twine-check package-size cli-smoke
