# pii-anon: Local Evaluation, Testing, and Release Guide

This guide walks you through everything end-to-end on your MacBook Pro: setting up the repos, running the full competitor evaluation, verifying tests, pushing to GitHub, publishing to TestPyPI, validating, and promoting to PyPI.

---

## Part 1: Prerequisites (One-Time Mac Setup)

### 1.1 Install system tools

```bash
# Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 3.12 (IMPORTANT: use 3.10–3.12; do NOT use 3.14 — spaCy/thinc lack wheels)
brew install python@3.12

# Git and GitHub CLI
brew install git gh

# Docker Desktop (only needed for canonical x86_64 benchmarks on Apple Silicon)
brew install --cask docker
```

> **Python version warning:** spaCy and thinc do not publish pre-built
> wheels for Python 3.14 (or 3.13 in some cases). If `pip install` fails with
> `Failed to build thinc / spacy`, switch to **Python 3.12** — it has the broadest
> wheel coverage for ML/NLP packages. Recreate your venv with `python3.12 -m venv .venv`.

Verify everything is in place:

```bash
python3.12 --version   # Should print 3.12.x
git --version
gh --version
```

### 1.2 Authenticate with GitHub

```bash
gh auth login
# Choose: GitHub.com → HTTPS → Login with a web browser
# Follow the browser flow to complete authentication
```

Verify:

```bash
gh auth status
# Should show: Logged in to github.com account subhash-holla
```

---

## Part 2: Repository Setup

You have three repos. The main library repo (`pii-anon`) contains both the source code and the test suite.

| Local folder | GitHub repo | What it holds |
|---|---|---|
| `pii-anon-code/` | `subhash-holla/pii-anon` | Library source (`src/`), tests (`tests/`), CI, docs, benchmarks |
| `pii-anon-doc/` | `subhash-holla/pii-anon-doc` | Standalone documentation site |
| `pii-anon-eval-data/` | `subhash-holla/pii-anon-eval-data` | `pii-anon-datasets` Python package (JSONL benchmark data — v1.3.0: 159,891 records, 63 entity types, 60 languages, Tier 3 behavioral signals) |

### 2.1 Create the GitHub repos (first time only)

```bash
gh repo create subhash-holla/pii-anon \
    --public \
    --description "OSS PII detection, anonymization, and evaluation library for Python" \
    --license Apache-2.0

gh repo create subhash-holla/pii-anon-doc \
    --public \
    --description "Documentation for pii-anon"

gh repo create subhash-holla/pii-anon-eval-data \
    --public \
    --description "Benchmark datasets for pii-anon evaluation" \
    --license Apache-2.0
```

If a repo already exists, `gh repo create` will tell you — that's fine, just move on.

### 2.2 Set up local workspace

If you're starting fresh on a new machine:

```bash
mkdir -p ~/projects/pii-anon && cd ~/projects/pii-anon

# Option A: Clone from GitHub (if repos already have content)
git clone https://github.com/subhash-holla/pii-anon.git          pii-anon-code
git clone https://github.com/subhash-holla/pii-anon-doc.git       pii-anon-doc
git clone https://github.com/subhash-holla/pii-anon-eval-data.git pii-anon-eval-data

# Option B: Point existing local folders to GitHub (if you built locally first)
cd ~/projects/pii-anon/pii-anon-code
git init
git remote add origin https://github.com/subhash-holla/pii-anon.git

cd ~/projects/pii-anon/pii-anon-doc
git init
git remote add origin https://github.com/subhash-holla/pii-anon-doc.git

cd ~/projects/pii-anon/pii-anon-eval-data
git init
git remote add origin https://github.com/subhash-holla/pii-anon-eval-data.git
```

### 2.3 Set up the Python development environment

```bash
cd ~/projects/pii-anon/pii-anon-code

# Create and activate a virtual environment (use python3.12 explicitly)
python3.12 -m venv .venv
source .venv/bin/activate

# One-command setup: installs local eval-data, all deps, and NLP models
make setup-swarm
```

Or manually step by step:

```bash
pip install --upgrade pip setuptools wheel

# Install pii-anon-datasets from the local eval-data repo FIRST
pip install -e ../pii-anon-eval-data/

# Install PyTorch CPU-only FIRST (LLM Guard, GLiNER depend on it)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install pii-anon with all extras (quote brackets for zsh)
pip install -e '.[dev,cli,crypto,benchmark,datasets,swarm-ml,swarm-train]'
```

> **zsh note:** Always quote the extras brackets — `pip install 'pii-anon[swarm-train]'` not `pip install pii-anon[swarm-train]`. Unquoted brackets cause `zsh: no matches found`.

**Note:** The `datasets` extra requires `pii-anon-datasets`. Until that package is published to PyPI, you must install it from the local `pii-anon-eval-data/` repo first. Once it's on PyPI, the single `pip install -e ".[dev,cli,crypto,benchmark,datasets]"` line will work on its own.

### 2.4 Download NLP models

The benchmark compares pii-anon against three competitors: **Presidio**, **scrubadub**, and **GLiNER**. Some require downloaded models.

pii-anon's own ensemble also includes **spaCy** and **Stanza** as core detection engines (not competitors). Their models need to be downloaded separately.

```bash
# --- Core engine models (pii-anon's own ensemble) ---

# spaCy model (also required internally by Presidio competitor)
python3 -m spacy download en_core_web_sm

# Stanza model (pii-anon core engine, not a benchmark competitor)
python3 -c "import stanza; stanza.download('en')"

# --- Competitor benchmark models ---

# GLiNER PII model (downloads ~500 MB)
python3 -c "from gliner import GLiNER; GLiNER.from_pretrained('knowledgator/gliner-pii-base-v1.0')"
```

### 2.5 Access benchmark datasets

The benchmark evaluation requires the `pii-anon-datasets` package which provides the JSONL evaluation data. Three options:

**Option A: Install from PyPI (after first release)**
```bash
pip install pii-anon-datasets
```

**Option B: Install from local clone**
```bash
pip install -e ../pii-anon-eval-data/
```

**Option C: Set environment variable**
```bash
export PII_ANON_DATASET_ROOT=/path/to/pii-anon-eval-data/src/pii_anon_datasets/data
```

The dataset resolution order is: installed `pii_anon_datasets` package → `PII_ANON_DATASET_ROOT` env var → sibling `pii-anon-eval-data/` repo → monorepo layout fallback. Both the canonical v1.1+ filename (`data/pii_anon.jsonl.gz`) and the legacy `eval_framework/data/pii_anon_eval_v1.jsonl.gz` layout are probed automatically.

**Dataset v1.3.0 brings Tier 3 resources.** Every record now carries a
`behavioral_signals` block, a per-record Re-identification Resistance Score
(RRS), and an `anonymized_llm_sanitized` text variant. In addition, 2,500
paired pseudonymous/real personas and ~4,500 ESRC-attack evaluation records
are available for Tier 3 experiments. The loader
(`pii_anon.eval_framework.datasets.schema.load_eval_dataset`) reads these
fields into `EvalBenchmarkRecord` attributes (`behavioral_signal_density`,
`re_identification_resistance_score`, `tier3_risk_level`, `is_paired_profile`,
`esrc_attack_target`, `context_preservation`). These feed directly into the
Tier 3 inputs of `compute_composite(...)`
(`reidentification_recall`, `reidentification_precision`,
`quasi_identifiers_removed`, `behavioral_signal_similarity`) — see
`src/pii_anon/eval_framework/metrics/composite.py`.

### 2.6 Verify the dev environment

```bash
# Quick health check
pii-anon version        # Should print the installed version
pii-anon health --output json

# Verify competitor engines load for benchmarking
pii-anon benchmark-preflight --output json
```

You should see `presidio`, `scrubadub`, and `gliner` all reported as available. If any are missing, revisit the install steps above.

---

## Part 3: Quality Gates

Run the full CI pipeline locally before anything else:

```bash
cd ~/projects/pii-anon/pii-anon-code
make all
```

This runs: lint (ruff) → type check (mypy) → test suite (2301+ tests, 85%+ coverage) → performance SLAs → build → twine check → package size check → CLI smoke test.

If any step fails, fix it before continuing.

---

## Part 4: Train the Swarm

Train the pii-anon-swarm offering. This produces Dawid-Skene confusion matrices, temperature scaling parameters, and informativeness scores.

```bash
# Quick training (10K records, 5-fold CV, ~5 min)
make train-swarm

# Train on ALL pii-anon-eval-data records (0 = unlimited; v1.3.0 → ~160K)
make train-swarm SWARM_MAX_RECORDS=0

# Faster iteration: 3-fold CV
make train-swarm SWARM_KFOLD=3

# No CV (fastest, single holdout split)
make train-swarm SWARM_KFOLD=1

# More robust: 10-fold CV
make train-swarm SWARM_KFOLD=10

# Train on multiple datasets
make train-swarm SWARM_DATASETS=pii_anon_eval,ai4privacy,conll2003

# Full production training: all datasets, all records, 5-fold CV
make train-swarm SWARM_DATASETS=pii_anon_eval,ai4privacy,conll2003 SWARM_MAX_RECORDS=0 SWARM_KFOLD=5
```

The training pipeline:
1. Loads training data (0 = all records, >0 = cap per dataset)
2. Runs stratified K-fold cross-validation (entity type distribution preserved across folds)
3. For each fold: trains Dawid-Skene, temperature scaling, informativeness; evaluates on held-out test fold
4. Reports per-fold and mean F1/precision/recall **and F2** with standard deviation
5. Retrains final model on ALL data (not just one fold's training set)
6. Runs the F2 threshold sweep to pick `emission_threshold` (maximises F2 on held-out split, per paper v10)
7. Deploys final artifacts to `~/.pii_anon/swarm/`

The K-fold results — including the chosen F2 threshold and the
21-feature `FEATURE_VERSION` — are saved in `manifest.json` so you can
review per-fold performance. See
[docs/swarm-architecture.md](swarm-architecture.md) for the full four-layer
pipeline, Tier 3 sample-weighting rationale, and the when-to-retrain
checklist. For iterating on configuration *before* a retrain, see
[docs/autoresearch-integration.md](autoresearch-integration.md).

Verify training succeeded:

```bash
ls ~/.pii_anon/swarm/
# Should contain: ds_params.json, temperature.json, informativeness.json, manifest.json
cat ~/.pii_anon/swarm/manifest.json | python -m json.tool
```

---

## Part 5: Run the Full Benchmark

Run the complete competitor evaluation and update all documentation:

```bash
make benchmark-full
```

This runs `scripts/run_full_benchmark.py` which:
1. Verifies all competitor engines are available (preflight check)
2. Runs pii-anon, pii-anon-swarm, GLiNER, Presidio, and Scrubadub against pii-anon-eval-data
3. Renders the benchmark summary markdown
4. Updates the README benchmark section (between `<!-- BENCHMARK_SUMMARY_START -->` and `<!-- BENCHMARK_SUMMARY_END -->` markers)
5. Renders the complex mode pseudonymization example
6. Validates that the README stays in sync with the benchmark data

For a quick benchmark (faster, fewer records):

```bash
python scripts/run_full_benchmark.py --max-samples 5000
```

The benchmark produces these artifacts:

| File | Description |
|------|-------------|
| `benchmark-results.json` | Raw results for all systems |
| `benchmark-raw.csv` | Per-record results |
| `floor-gate-report.md` | Pass/fail per use-case profile |
| `artifacts/benchmarks/floor-baseline.json` | Floor baseline for regression detection |
| `docs/benchmark-summary.md` | Rendered markdown summary |
| `README.md` | Updated benchmark section |

### Understanding the results

The benchmark evaluates 5 systems across 6 use-case profiles (3 accuracy, 3 speed):

| System | Type | Description |
|--------|------|-------------|
| **pii-anon** | Core | Fast regex engine with checksum validators |
| **pii-anon-swarm** | Premium | Four-layer pipeline (regex + NER + Dawid-Skene + meta-learner) |
| **GLiNER** | Competitor | Zero-shot transformer NER |
| **Presidio** | Competitor | Microsoft PII detection framework |
| **Scrubadub** | Competitor | Rule-based PII scrubbing |

Key metrics reported:
- **F1 / Precision / Recall** — entity-level with strict span matching
- **F2 (β=2)** — privacy-first detection score (recall weighted 2×, per TAB 2022)
- **Composite** — pii-rate-elo weighted score; `CompositeConfig.f2_privacy_first()` and
  `CompositeConfig.for_deployment(profile)` expose the recommended weight presets
- **Tier 3 (optional)** — Re-identification Resistance Score (RRS), Quasi-Identifier
  Coverage (QIC), Behavioral Signal Leakage (BSL); enabled when the benchmark
  result forwards `reidentification_recall`, `quasi_identifiers_removed`, or
  `behavioral_signal_similarity`
- **Elo** — tournament rating from round-robin on composite scores
- **Bootstrap 95% CI** — confidence intervals from 1000 resamples
- **Per-entity F1** — breakdown by entity type (in JSON artifacts)

---


## Part 6: Push to GitHub

```bash
cd ~/projects/pii-anon/pii-anon-code

git status
git diff --stat

git add -A
git commit -m "Release: pii-anon with swarm pipeline and updated benchmarks"
git push origin main
```

Push the other repos if they changed:

```bash
# Datasets repo
cd ~/projects/pii-anon/pii-anon-eval-data
git add -A && git commit -m "Update benchmark datasets" && git push origin main

# Documentation repo
cd ~/projects/pii-anon/pii-anon-doc
git add -A && git commit -m "Update documentation" && git push origin main
```

---

## Part 7: Tag and Publish

### 7.1 Create the release tag

```bash
cd ~/projects/pii-anon/pii-anon-code

# Tag with the version from pyproject.toml
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
git tag -a "v${VERSION}" -m "pii-anon ${VERSION}"
git push origin "v${VERSION}"
```

The `v*` tag push triggers the `publish-release` GitHub Action which:
1. Runs quality gates (ruff, mypy, pytest, performance SLAs)
2. Builds the wheel and sdist
3. Publishes to PyPI (for `v*` tags without `-rc`)

### 7.2 Tag the datasets repo

```bash
cd ~/projects/pii-anon/pii-anon-eval-data
git tag -a "v${VERSION}" -m "pii-anon-datasets ${VERSION}"
git push origin "v${VERSION}"
```

### 7.3 Create GitHub Releases

```bash
cd ~/projects/pii-anon/pii-anon-code
gh release create "v${VERSION}" --title "pii-anon ${VERSION}" \
    --notes "See README for details."

cd ~/projects/pii-anon/pii-anon-eval-data
gh release create "v${VERSION}" --title "pii-anon-datasets ${VERSION}" \
    --notes "Benchmark datasets for pii-anon evaluation."
```

---

## Part 8: Verify on PyPI

```bash
# Create a clean venv and install from PyPI
python3.12 -m venv /tmp/pii-anon-verify && source /tmp/pii-anon-verify/bin/activate
pip install "pii-anon[cli,datasets]"

# Smoke tests
pii-anon version
pii-anon health --output json
pii-anon detect "Contact alice@example.com"

# Verify dataset is accessible
python -c "import pii_anon_datasets; print(f'Records: {len(pii_anon_datasets.load_dataset())}')"

# Clean up
deactivate && rm -rf /tmp/pii-anon-verify
```

---

## Release Checklist

```
1. SETUP
   [ ] Python 3.12 venv activated
   [ ] All dependencies installed (pip install -e ".[dev,cli,crypto,benchmark,datasets]")
   [ ] NLP models downloaded (spaCy, Stanza, GLiNER)

2. QUALITY GATES
   [ ] make all passes (lint + type + tests + perf + build + checks)

3. SWARM TRAINING
   [ ] make train-swarm completes successfully
   [ ] Artifacts deployed to ~/.pii_anon/swarm/
   [ ] manifest.json shows expected datasets and F1

4. BENCHMARK
   [ ] make benchmark-full completes successfully
   [ ] README benchmark section updated automatically
   [ ] benchmark-results.json produced
   [ ] Results reviewed (pii-anon and pii-anon-swarm both outperform competitors)
   [ ] pii-anon-datasets ≥ v1.3.0 installed (Tier 3 signals populated)
   [ ] summarize_eval_dataset reports `avg_re_identification_resistance_score` (sanity check: non-null)
   [ ] Swarm `manifest.json` shows `feature_version=2` and non-default `emission_threshold` (F2-selected)
   [ ] Industry-leadership bar check: `FloorGateConfig.industry_leadership()` passes on the published `pii-anon` / `pii-anon-swarm` scorecards

5. PUBLISH
   [ ] All repos pushed to GitHub
   [ ] Version tag created and pushed
   [ ] GitHub Action publish-release triggered
   [ ] PyPI package installable and smoke tests pass
```
