ifeq ($(OS),Windows_NT)
VENV_PYTHON := .venv/Scripts/python.exe
else
VENV_PYTHON := .venv/bin/python
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
BENCH_DATASET ?= pii_anon_benchmark_v1
BENCH_ENGINE_TIERS ?= auto minimal standard full
BENCH_MATRIX ?= src/pii_anon/benchmarks/matrix/use_case_matrix.v1.json
BENCH_ARTIFACTS ?= artifacts/benchmarks
BENCH_WORKDIR ?= .publish-suite
BENCH_WARMUP ?= 100
BENCH_RUNS ?= 3

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
	--engine-tiers $(BENCH_ENGINE_TIERS) \
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
	--enforce-floors \
	--enforce-publish-claims \
	--validate-readme-sync

.PHONY: bootstrap install-dev lint type test perf build twine-check package-size benchmark compare-benchmark benchmark-preflight benchmark-publish-suite benchmark-portable benchmark-portable-macos benchmark-portable-linux benchmark-portable-windows benchmark-canonical benchmark-canonical-linux benchmark-canonical-macos benchmark-canonical-macos-native benchmark-canonical-cloud benchmark-canonical-windows benchmark-docker-build benchmark-native-setup benchmark-multi-tier benchmark-comprehensive readme-benchmark-check cli-smoke docs-smoke all

bootstrap:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev,cli,crypto,engines]

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
	$(PYTHON) scripts/run_competitor_benchmark.py --dataset $(BENCH_DATASET) --dataset-source package-only --matrix $(BENCH_MATRIX) --strict-runtime --require-all-competitors --require-native-competitors --enforce-floors --output-json benchmark-results.json --output-csv benchmark-raw.csv --output-floor-report floor-gate-report.md --output-baseline artifacts/benchmarks/floor-baseline.v1.0.0.json --preflight-output-json artifacts/benchmarks/runtime-preflight.json

benchmark:
	$(PYTHON) scripts/generate_benchmark_dataset.py
	$(PYTHON) scripts/check_benchmark_runtime.py --strict-runtime --require-all-competitors --require-native-competitors --output-json artifacts/benchmarks/runtime-preflight.json
	$(PYTHON) scripts/run_competitor_benchmark.py --dataset $(BENCH_DATASET) --dataset-source package-only --matrix $(BENCH_MATRIX) --strict-runtime --require-all-competitors --require-native-competitors --enforce-floors --output-json benchmark-results.json --output-csv benchmark-raw.csv --output-floor-report floor-gate-report.md --output-baseline artifacts/benchmarks/floor-baseline.v1.0.0.json --preflight-output-json artifacts/benchmarks/runtime-preflight.json
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

# Quick local run with all engine tiers against a single dataset.
# Useful for verifying tier-level differences without running the full matrix.
benchmark-multi-tier:
	$(PYTHON) scripts/run_publish_grade_suite.py $(PORTABLE_SUITE_FLAGS) --engine-tiers $(BENCH_ENGINE_TIERS)

# Full matrix: all engine tiers against the unified dataset.
benchmark-comprehensive:
	$(PYTHON) scripts/run_publish_grade_suite.py $(PORTABLE_SUITE_FLAGS) --engine-tiers $(BENCH_ENGINE_TIERS)

readme-benchmark-check:
	$(PYTHON) scripts/check_readme_benchmark.py --readme README.md --summary docs/benchmark-summary.md --report-json benchmark-results.json

cli-smoke:
	pii-anon version
	pii-anon health --output json

docs-smoke:
	jupyter nbconvert --to notebook --execute notebooks/llm_pipeline_quickstart.ipynb --output /tmp/llm_pipeline_quickstart.executed.ipynb

all: lint type test perf build twine-check package-size cli-smoke
