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
| `pii-anon-eval-data/` | `subhash-holla/pii-anon-eval-data` | `pii-anon-datasets` Python package (JSONL benchmark data: pii_anon_benchmark_v1, eval_framework_v1) |

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

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install pii-anon-datasets from the local eval-data repo FIRST
# (it isn't on PyPI yet, so pip can't resolve it otherwise)
pip install -e ../pii-anon-eval-data/

# Install PyTorch CPU-only FIRST (LLM Guard, GLiNER depend on it)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Now install pii-anon with all extras including benchmark competitors
pip install -e ".[dev,cli,crypto,benchmark,datasets]"
```

The `benchmark` extra installs all three competitor engines (Presidio, scrubadub, GLiNER) plus their transitive dependencies (spaCy, Stanza, transformers, tqdm). PyTorch must be installed first because pip cannot resolve it from the CPU-only index automatically.

**Note:** The `datasets` extra requires `pii-anon-datasets>=1.0.0`. Until that package is published to PyPI, you must install it from the local `pii-anon-eval-data/` repo first. Once it's on PyPI, the single `pip install -e ".[dev,cli,crypto,benchmark,datasets]"` line will work on its own.

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

The dataset resolution order is: installed `pii_anon_datasets` package → `PII_ANON_DATASET_ROOT` env var → sibling `pii-anon-eval-data/` repo → monorepo layout fallback.

### 2.6 Verify the dev environment

```bash
# Quick health check
pii-anon version        # Should print 1.0.0
pii-anon health --output json

# Verify competitor engines load for benchmarking
pii-anon benchmark-preflight --output json
```

You should see `presidio`, `scrubadub`, and `gliner` all reported as available. If any are missing, revisit the install steps above.

---

## Part 3: Running All Tests

### 3.1 Run the full test suite with coverage

```bash
cd ~/projects/pii-anon/pii-anon-code

# Tests live in tests/ — pytest discovers them via pyproject.toml testpaths
python3 -m pytest tests/ \
    --cov=pii_anon \
    --cov-report=term-missing \
    --cov-fail-under=85 \
    -v
```

Expected result: **1052+ tests passing**, coverage around **90%**, well above the 85% gate.

### 3.2 Run individual test categories

```bash
# Research rigor enhancements (dataset generator, bootstrap CI, Cohen's kappa, etc.)
python3 -m pytest tests/test_research_rigor_enhancements.py -v --no-cov

# File ingestion (CSV, JSON, JSONL, TXT readers/writers)
python3 -m pytest tests/test_ingestion_*.py -v --no-cov

# Streaming chunker (large-text splitting)
python3 -m pytest tests/test_streaming_chunker.py -v --no-cov

# Composite scoring and governance gates
python3 -m pytest tests/test_composite_scoring.py tests/test_governance_thresholds.py -v --no-cov

# Benchmark pipeline (summary rendering, competitor comparison)
python3 -m pytest tests/test_benchmark_summary_render.py tests/test_competitor_compare_unit.py -v --no-cov

# Multi-tier and multi-dataset benchmark tests
python3 -m pytest tests/test_multi_dataset_benchmark.py -v --no-cov

# Performance SLA tests (excluded from the default run; run them separately)
python3 -m pytest tests/performance/ -m performance --no-cov -v
```

### 3.3 Run lint and type checks

```bash
make lint    # ruff check src tests
make type    # mypy src/pii_anon
```

### 3.4 Run the entire CI pipeline locally

```bash
make all
```

This runs in order: lint, type check, test suite, performance SLAs, build, twine check, package size check, CLI smoke test, and docs smoke test.

### 3.5 Verify the build artifacts

```bash
make build          # Builds sdist + wheel into dist/
make twine-check    # Validates the packages
make package-size   # Ensures wheel stays under 1.5 MB
```

---

## Part 4: Running the Full Competitor Evaluation

### 4.1 Preflight check

```bash
cd ~/projects/pii-anon/pii-anon-code

pii-anon benchmark-preflight \
    --strict-runtime \
    --require-all-competitors \
    --require-native-competitors \
    --output json
```

All four competitors should show as available.

### 4.2 Portable benchmark (quick, no strict gates)

```bash
make benchmark-portable
```

This runs the evaluation with relaxed flags against the default single dataset. Results land in `artifacts/benchmarks/`.

### 4.3 Multi-tier benchmark (all engine tiers, single dataset)

```bash
make benchmark-multi-tier
```

Evaluates all four pii-anon engine tiers (`auto`, `minimal`, `standard`, `full`) as separate systems alongside competitors. Each tier uses a different engine configuration — `minimal` is regex+presidio (fastest), `standard` adds GLiNER (best F1), and `full` includes all available engines. Results show how each tier stacks up against competitors.

### 4.4 Comprehensive benchmark (all tiers × all datasets)

```bash
make benchmark-comprehensive
```

Runs the full matrix: all four engine tiers evaluated against all datasets (`pii_anon_benchmark_v1`, `eval_framework_v1`). Produces per-dataset artifacts plus a combined cross-dataset report (`benchmark-combined.json`) with aggregated metrics and an interpretation section. This is the best way to see a complete picture of where each tier succeeds and where it falls short.

The suite uses an **evaluations-first** pipeline: all time-consuming evaluations (per-dataset benchmarks + continuity) run to completion before any post-processing. If one dataset fails its floor gate, the remaining datasets still finish. A final summary table shows PASS/FAIL per dataset, and floor enforcement is deferred until after all results are rendered — so you always get the full picture.

### 4.5 Canonical benchmark (publication-grade, all gates enforced)

On an Apple Silicon Mac, the fastest path runs natively without Docker (~2-3x faster than the Docker path):

```bash
# First time only — install all benchmark dependencies natively:
make benchmark-native-setup

# Run the canonical benchmark natively on ARM64:
make benchmark-canonical-macos-native
```

Alternatively, for full Docker-based reproducibility (slower due to Rosetta x86_64 emulation):

```bash
make benchmark-canonical-macos
```

Both targets run the full suite with strict runtime checks, floor gates, and README sync validation. The canonical suite evaluates all engine tiers across all datasets by default.

If you're on an Intel Mac:

```bash
make benchmark-canonical
```

#### Resuming an interrupted benchmark run

The canonical benchmark automatically saves **per-profile checkpoints** to `artifacts/benchmarks/checkpoints/<dataset>/` as each profile completes. If the run is interrupted (machine restart, Docker crash, `Ctrl+C`), simply re-run the same `make` command:

```bash
# Just re-run — completed profiles are loaded from checkpoint
make benchmark-canonical-macos
```

The pipeline will detect existing checkpoint files, skip already-completed profiles (crediting their work to the progress bar), and continue from where it left off. This can save hours on long runs where slow systems (like `pii-anon-standard` or `gliner`) take a long time per profile.

To **force a fresh run** (discarding checkpoints), delete the checkpoint directory first:

```bash
rm -rf artifacts/benchmarks/checkpoints/
make benchmark-canonical-macos
```

You can also pass `--checkpoint-dir` explicitly when running the suite script directly:

```bash
python scripts/run_publish_grade_suite.py \
    --checkpoint-dir /tmp/my-checkpoints \
    ...other flags...
```

### 4.6 Cloud benchmark (fastest option)

Running the canonical benchmark on a cloud VM is the fastest way to get publication-grade results. A benchmark that takes 4-5 days locally (Docker/Rosetta) or 1.5-2 days natively on a MacBook Pro can complete in 6-12 hours on a cloud instance with enough cores.

#### Cloud provider recommendations

| Provider | Instance | Arch | vCPUs | RAM | On-Demand $/hr | Spot $/hr | Est. Runtime | Est. Cost (spot) |
|---|---|---|---|---|---|---|---|---|
| **AWS** | **c7g.8xlarge** | ARM64 (Graviton3) | 32 | 64 GB | ~$1.16 | ~$0.35 | 8-10h | **$3-4** |
| **AWS** | **c8g.8xlarge** | ARM64 (Graviton4) | 32 | 64 GB | ~$1.28 | ~$0.38 | 6-8h | **$2-3** |
| AWS | c7g.16xlarge | ARM64 (Graviton3) | 64 | 128 GB | ~$2.32 | ~$0.70 | 5-7h | $4-5 |
| GCP | c4a-standard-32 | ARM64 (Axion) | 32 | 128 GB | ~$1.32 | ~$0.40 | 8-10h | $3-4 |
| Hetzner | CAX41 | ARM64 (Ampere) | 16 | 32 GB | ~€0.038 | N/A | 16-20h | **€0.60-0.76** |

**Recommended: AWS c7g.8xlarge or c8g.8xlarge spot instance.** Best balance of speed and cost. Graviton ARM64 runs the same Python/NumPy/PyTorch code natively without emulation, 32 vCPUs give up to 12 parallel workers (memory-aware scaling), and spot pricing makes a full canonical run cost $2-4. Use us-east-1 for lowest spot prices.

**Budget option: Hetzner CAX41.** At €0.038/hr (~$0.04/hr), a full run costs under €1 total. However, with only 16 vCPUs and 32 GB RAM, it takes 16-20h. Still faster than local Docker on Apple Silicon, and by far the cheapest option.

#### Cost comparison: cloud vs. local

| Approach | Wall-Clock Time | Hardware Cost | Electricity | Total Cost |
|---|---|---|---|---|
| Local: Docker on M1 Pro (Rosetta) | 4-5 days | $0 (owned) | ~$3-5 | ~$3-5 |
| Local: Native on M1 Pro | 1.5-2 days | $0 (owned) | ~$1.5-3 | ~$1.5-3 |
| Local: Native on Mac Studio M2 Ultra | 18-24h | $0 (owned) | ~$1-2 | ~$1-2 |
| **Cloud: AWS c7g.8xlarge spot** | **8-10h** | **$3-4** | $0 | **$3-4** |
| **Cloud: AWS c8g.8xlarge spot** | **6-8h** | **$2-3** | $0 | **$2-3** |
| Cloud: Hetzner CAX41 | 16-20h | €0.60-0.76 | $0 | **<$1** |

Cloud wins on speed even when native macOS is available. A spot c8g.8xlarge cuts wall-clock time from 1.5-2 days to 6-8 hours for ~$3 — worth it when iterating on benchmark results. Hetzner wins on pure cost if you can wait 16-20 hours.

#### Step-by-step: AWS (recommended)

**One-time setup (from your Mac):**

1. **Install the AWS CLI:**

```bash
brew install awscli
aws configure
# Enter: AWS Access Key ID, Secret Access Key, region (us-east-1), output format (json)
```

Your IAM user needs `AmazonEC2FullAccess` permissions. Attach it via the AWS Console: IAM > Users > your-user > Add permissions > Attach policies directly > AmazonEC2FullAccess.

2. **Create an SSH key pair:**

```bash
aws ec2 create-key-pair \
    --key-name pii-anon-bench \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/pii-anon-bench.pem
chmod 400 ~/.ssh/pii-anon-bench.pem
```

3. **Create a security group for SSH access:**

```bash
aws ec2 create-security-group \
    --group-name pii-anon-bench-sg \
    --description "SSH access for benchmark runs"
# Note the GroupId from the output (e.g., sg-0abc123...)

aws ec2 authorize-security-group-ingress \
    --group-id <your-group-id> \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0
```

**Launch and run the benchmark:**

4. **Find the latest Ubuntu 22.04 ARM64 AMI:**

```bash
aws ec2 describe-images \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-arm64-server-*" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text
```

The owner ID `099720109477` is Canonical's (Ubuntu's) official AWS account.

5. **Launch an ARM64 spot instance:**

```bash
aws ec2 run-instances \
    --image-id <ami-from-step-4> \
    --instance-type c7g.8xlarge \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --key-name pii-anon-bench \
    --security-group-ids <sg-from-step-3> \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=pii-anon-benchmark}]'
```

6. **Get the public IP** (wait ~30 seconds for the instance to start):

```bash
aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=pii-anon-benchmark" "Name=instance-state-name,Values=running,pending" \
    --query 'Reservations[0].Instances[0].[InstanceId,PublicIpAddress]' \
    --output text
```

7. **SSH in:**

```bash
ssh -i ~/.ssh/pii-anon-bench.pem ubuntu@<instance-ip>
```

8. **Install Python 3.12 and clone repos** (on the cloud VM):

```bash
# Ubuntu 22.04 ships Python 3.10; add deadsnakes PPA for 3.12
sudo apt-get update && sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git

# Clone both repos
git clone https://github.com/subhash-holla/pii-anon.git pii-anon-code
git clone https://github.com/subhash-holla/pii-anon-eval-data.git
```

9. **Run the benchmark:**

**Sequential mode** (default — simpler, lower memory):

```bash
cd pii-anon-code
PYTHON=python3.12 bash scripts/run_cloud_benchmark.sh
```

The script creates a virtual environment, installs all dependencies, downloads NLP models, and runs the full canonical suite. Progress is logged to stdout at 1-minute intervals. On c7g.8xlarge the benchmark uses up to 12 parallel workers (memory-aware scaling: RAM_GB / 4, capped at 12). Duration: 3-4 days.

**Parallel mode** (recommended on c7g.8xlarge with 64 GB RAM — ~5-6× faster):

```bash
cd pii-anon-code
PYTHON=python3.12 bash scripts/run_cloud_benchmark.sh --parallel
```

This evaluates all 6 profiles in parallel (one process per profile), then merges results. Each process loads its own NLP models (~10 GB each, ~60 GB total). Duration: 4-8 hours instead of 3-4 days.

You can also limit parallelism on smaller VMs: `BENCH_WORKERS=3 bash scripts/run_cloud_benchmark.sh --parallel` runs 3 profiles at a time (2 batches of 3). Per-profile logs go to `artifacts/benchmarks/logs/`.

**Running overnight (auto-shutdown):** Append a shutdown command so the instance stops when the benchmark finishes (avoids paying for idle hours):

```bash
cd pii-anon-code
PYTHON=python3.12 bash scripts/run_cloud_benchmark.sh --parallel; sudo shutdown -h now
```

A stopped instance still has its EBS volume attached (~$0.08/GB/month, pennies for 50 GB). Terminate it in the morning to stop all charges.

10. **Download artifacts when done** (from your Mac):

```bash
# Create a local directory for the results
mkdir -p ~/projects/pii-anon/cloud-benchmark-results

# Download all benchmark artifacts
scp -i ~/.ssh/pii-anon-bench.pem -r \
    ubuntu@<instance-ip>:~/pii-anon-code/artifacts/benchmarks/ \
    ~/projects/pii-anon/cloud-benchmark-results/
```

If the instance was auto-stopped (shutdown), start it first to download:

```bash
aws ec2 start-instances --instance-ids <instance-id>
# Wait ~30 seconds, then get the new public IP:
aws ec2 describe-instances \
    --instance-ids <instance-id> \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text
# Download, then terminate
```

11. **Terminate the instance** (important -- stop all charges!):

```bash
aws ec2 terminate-instances --instance-ids <instance-id>
```

12. **Integrate results locally** (update README and docs on your Mac):

The cloud VM produced the benchmark JSON artifacts. The rendering and README integration steps are lightweight and run locally -- they only need the JSON files, not the dataset.

```bash
cd ~/projects/pii-anon/pii-anon-code

# Copy cloud artifacts into the local artifacts directory
cp -r ~/projects/pii-anon/cloud-benchmark-results/* artifacts/benchmarks/

# Render the benchmark summary and update README
python3 scripts/render_benchmark_summary.py \
    --input-json artifacts/benchmarks/benchmark-results.json \
    --output-markdown artifacts/benchmarks/benchmark-summary.md \
    --update-readme README.md

# Render complex mode example and update README
python3 scripts/render_complex_mode_example.py \
    --output-markdown docs/complex-mode-example.md \
    --update-readme README.md

# If you ran a multi-dataset benchmark (benchmark-combined.json exists):
# python3 scripts/render_marketing_narrative.py \
#     --input-json artifacts/benchmarks/benchmark-combined.json \
#     --output-markdown artifacts/benchmarks/marketing-narrative.md \
#     --update-readme README.md

# Validate everything is in sync
python3 scripts/check_readme_benchmark.py \
    --readme README.md \
    --summary artifacts/benchmarks/benchmark-summary.md \
    --complex-summary docs/complex-mode-example.md \
    --report-json artifacts/benchmarks/benchmark-results.json
```

**What runs locally vs. what needs the cloud:**

| Step | Runs Locally? | Needs Dataset? |
|---|---|---|
| Render benchmark summary | Yes | No (uses JSON artifacts) |
| Render complex mode example | Yes | No (uses hardcoded demo) |
| Render marketing narrative | Yes | No (uses JSON artifacts) |
| Validate README sync | Yes | No (compares markdown files) |
| Run competitor benchmark | No | Yes (needs 50K-record dataset) |
| Run continuity benchmark | No | Yes (needs dataset) |

After the validation passes, commit the updated README and artifacts:

```bash
git add README.md docs/complex-mode-example.md artifacts/benchmarks/
git commit -m "Update benchmark results from cloud run

- Canonical benchmark on AWS c7g.8xlarge (ARM64 Graviton3)
- All 6 profiles passed, all competitors evaluated
- README benchmark section updated"
git push origin main
```

#### Step-by-step: Hetzner (budget option)

1. **Create a CAX41 server** via the [Hetzner Cloud Console](https://console.hetzner.cloud/) or CLI:

```bash
hcloud server create \
    --name pii-anon-bench \
    --type cax41 \
    --image ubuntu-22.04 \
    --ssh-key your-key \
    --location nbg1
```

2. **SSH in, install Python 3.12, clone, and run:**

```bash
ssh root@<server-ip>
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.12 python3.12-venv python3.12-dev git
git clone https://github.com/subhash-holla/pii-anon.git pii-anon-code
git clone https://github.com/subhash-holla/pii-anon-eval-data.git
cd pii-anon-code
PYTHON=python3.12 bash scripts/run_cloud_benchmark.sh
```

3. **Download artifacts and delete the server:**

```bash
scp -r root@<server-ip>:~/pii-anon-code/artifacts/benchmarks/ ./cloud-benchmark-results/
hcloud server delete pii-anon-bench
```

#### Using checkpoint/resume on cloud

The benchmark saves per-profile checkpoints to `artifacts/benchmarks/checkpoints/`. If a spot instance is interrupted, simply launch a new instance, clone the repos, restore the artifacts directory, and re-run the script -- completed profiles are automatically skipped.

```bash
# Before launching: upload previous partial results
scp -i ~/.ssh/pii-anon-bench.pem -r ./cloud-benchmark-results/checkpoints/ ubuntu@<new-ip>:~/pii-anon-code/artifacts/benchmarks/

# Then re-run — will resume from last checkpoint
cd pii-anon-code
PYTHON=python3.12 bash scripts/run_cloud_benchmark.sh
```

#### Tips for fastest cloud runs

- **Use `--parallel` mode** -- On c7g.8xlarge (64 GB RAM), `--parallel` evaluates all 6 profiles concurrently, cutting wall time from 3-4 days to 4-8 hours. Cost is actually *lower* because you pay for fewer instance-hours.
- **Use spot/preemptible instances** -- 60-70% cheaper than on-demand, and the checkpoint system handles interruptions.
- **Pick ARM64 over x86** -- Graviton/Axion instances are 20-40% cheaper with better per-core performance for Python workloads.
- **Use gp3 SSD storage** -- The benchmark is CPU-bound, but fast storage helps with model loading. 50 GB gp3 is sufficient.
- **Check spot pricing across regions** -- us-east-1 and us-west-2 typically have the lowest spot prices.
- **Worker scaling** -- On a 64 GB RAM cloud VM, the benchmark auto-detects 12 parallel workers (RAM_GB / 4, capped at 12). In `--parallel` mode each profile gets its own process with ~5-6 within-profile threads. The `BENCH_WORKERS` env var controls how many profiles run concurrently (default: 6).
- **Don't over-provision** -- 32 vCPUs is the sweet spot for c7g.8xlarge. With 12 workers, cores beyond 32 provide diminishing returns. 64 GB RAM handles 12 concurrent NLP model instances comfortably.

### 4.7 Quick CLI evaluation (no Makefile needed)

```bash
pii-anon compare-competitors \
    --dataset pii_anon_benchmark_v1 \
    --warmup-samples 50 \
    --measured-runs 3 \
    --include-end-to-end \
    --require-all-competitors \
    --require-native-competitors \
    --enforce-floors \
    --output json
```

To evaluate multiple engine tiers in a single run:

```bash
python3 scripts/run_competitor_benchmark.py \
    --dataset pii_anon_benchmark_v1 \
    --engine-tiers auto minimal standard full \
    --warmup-samples 50 \
    --measured-runs 3 \
    --output-json benchmark-results.json \
    --output-csv benchmark-raw.csv \
    --output-floor-report floor-gate-report.md
```

This produces four `pii-anon` variants (`pii-anon`, `pii-anon-minimal`, `pii-anon-standard`, `pii-anon-full`) alongside competitors. Competitors are evaluated only once regardless of how many tiers are requested.

### 4.8 Evaluation framework assessment

```bash
pii-anon eval-framework \
    --dataset eval_framework_v1 \
    --max-records 5000 \
    --output json
```

### 4.9 Understanding benchmark output columns

The benchmark summary table includes these columns per system:

- **Composite** — Two-tier weighted score combining accuracy (Tier 1: β-weighted F1 from precision/recall) and operational metrics (Tier 2: latency, throughput, entity coverage).
- **F1 / Precision / Recall** — Standard detection accuracy metrics computed against ground-truth labels.
- **p50 Latency (ms)** — Median per-document detection latency.
- **Docs/hour** — Throughput extrapolated from total elapsed time.
- **Elo** — Tournament rating from 3-round round-robin using composite scores.
- **Per-entity precision/recall** — Breakdown by entity type (in JSON artifacts, not in summary table).
- **Entity coverage** — Count of entity types with non-zero recall vs total types in ground truth.

When multiple engine tiers are evaluated, the summary includes one row per tier (`pii-anon`, `pii-anon-minimal`, `pii-anon-standard`, `pii-anon-full`). The floor gate uses only the canonical `pii-anon` (auto) tier; other tier variants are informational.

When multiple datasets are evaluated, the combined report (`benchmark-combined.json`, schema `2026-02-19.v3`) adds a cross-dataset summary with sample-weighted averages per system, best/worst dataset per system, and a per-tier F1 breakdown by dataset. The rendered markdown includes an interpretation section explaining what it means to succeed on one dataset but fall short on another.

### 4.10 Regenerate the README benchmark section

After a benchmark run, update the docs and verify they stay in sync:

```bash
# Single-dataset mode (original behavior)
python3 scripts/render_benchmark_summary.py \
    --input-json benchmark-results.json \
    --output-markdown docs/benchmark-summary.md \
    --require-floor-pass

# Multi-dataset mode (pass combined report + per-dataset reports)
python3 scripts/render_benchmark_summary.py \
    --input-json artifacts/benchmarks/benchmark-combined.json \
    --input-json artifacts/benchmarks/benchmark-results-pii_anon_benchmark_v1.json \
    --input-json artifacts/benchmarks/benchmark-results-eval_framework_v1.json \
    --output-markdown docs/benchmark-summary.md

python3 scripts/render_complex_mode_example.py

python3 scripts/check_readme_benchmark.py \
    --readme README.md \
    --summary docs/benchmark-summary.md \
    --complex-summary docs/complex-mode-example.md \
    --report-json benchmark-results.json
```

---

## Part 5: Pushing to GitHub

### 5.1 Push the main library

```bash
cd ~/projects/pii-anon/pii-anon-code

# Check what's changed
git status
git diff --stat

# Stage and commit
git add -A
git commit -m "v1.0.0: Production release with research-grade evaluation framework

- 36 PII entity types with context-aware confidence scoring and checksum validation
- Multi-engine ensemble detection (regex, Presidio, scrubadub, spaCy, stanza)
- Modular regex engine with declarative PatternSpec registry
- Two-tier composite scoring (β-weighted F1 composition + governance gates)
- PII-Rate-Elo tournament ratings with per-entity precision and entity coverage
- Sync engine fast-path bypassing asyncio overhead for single-engine benchmarks
- 10K-record evaluation dataset with 52-language taxonomy support
- Bootstrap CI, paired significance testing, Cohen's kappa
- Context-preserving pseudonymization with entity linking
- File ingestion (CSV, JSON, JSONL, TXT) and streaming chunker
- 1052 tests at 90% code coverage
- NIST, GDPR, ISO, HIPAA, CCPA, PCI-DSS compliance validation"

git branch -M main
git push -u origin main
```

### 5.2 Push the documentation repo

```bash
cd ~/projects/pii-anon/pii-anon-doc
git add -A
git commit -m "v1.0.0: Documentation for pii-anon release"
git branch -M main
git push -u origin main
```

### 5.3 Push the datasets repo

```bash
cd ~/projects/pii-anon/pii-anon-eval-data
git add -A
git commit -m "v1.0.0: Benchmark datasets for pii-anon evaluation

- pii_anon_benchmark_v1 dataset (50,000 records, 22 entity types, 12 languages, 7 evaluation dimensions)
- eval_framework_v1 dataset (50,000 records, 52 languages)"

git branch -M main
git push -u origin main
```

### 5.4 Tag releases

```bash
cd ~/projects/pii-anon/pii-anon-code
git tag -a v1.0.0 -m "v1.0.0: First public release"
git push origin v1.0.0

cd ~/projects/pii-anon/pii-anon-eval-data
git tag -a v1.0.0 -m "v1.0.0: Benchmark datasets"
git push origin v1.0.0
```

### 5.5 Create GitHub Releases

```bash
cd ~/projects/pii-anon/pii-anon-code
gh release create v1.0.0 \
    --title "pii-anon v1.0.0" \
    --notes "First public release. See README for full details. Documentation: https://github.com/subhash-holla/pii-anon-doc"

cd ~/projects/pii-anon/pii-anon-eval-data
gh release create v1.0.0 \
    --title "pii-anon-datasets v1.0.0" \
    --notes "Benchmark datasets for pii-anon evaluation. Install independently or auto-included with pii-anon[datasets]."

cd ~/projects/pii-anon/pii-anon-doc
gh release create v1.0.0 \
    --title "pii-anon Documentation v1.0.0" \
    --notes "Comprehensive documentation for pii-anon library, evaluation framework, and benchmarks."
```

### 5.6 Configure GitHub Actions for Trusted Publishing (OIDC)

The CI workflows in `.github/workflows/release.yml` use **Trusted Publishing (OIDC)** via `pypa/gh-action-pypi-publish` for secure, keyless authentication to PyPI and TestPyPI.

#### Step 1: Create PyPI and TestPyPI OIDC tokens

**TestPyPI OIDC Setup:**

1. Go to https://test.pypi.org/manage/account/#api-tokens
2. Click "Add API token"
3. Token name: `pii-anon-oidc`
4. Scope: "Project" → select/create `pii-anon-datasets` and `pii-anon` projects (or use entire account for first setup)
5. **Important:** Choose "OIDC" as the token type (not API token)
6. Trusted publishers: Add your GitHub repository (`subhash-holla/pii-anon`)
7. Copy and save the token

**PyPI OIDC Setup:**

1. Go to https://pypi.org/manage/account/#api-tokens
2. Click "Add API token"
3. Token name: `pii-anon-oidc`
4. Scope: "Project" → select/create `pii-anon-datasets` and `pii-anon` projects
5. **Important:** Choose "OIDC" as the token type
6. Trusted publishers: Add `subhash-holla/pii-anon`
7. Copy and save the token

Alternatively, if OIDC isn't available yet, you can use repository secrets instead (see fallback option below).

#### Step 2: Create GitHub repository secrets (Optional fallback)

If your PyPI/TestPyPI accounts don't support OIDC yet, you can use API tokens instead:

```bash
cd ~/projects/pii-anon/pii-anon-code

# Set the TestPyPI token
gh secret set TEST_PYPI_API_TOKEN

# Set the PyPI token
gh secret set PYPI_API_TOKEN
```

Each command will prompt you to paste the token. See Part 6.2 for how to generate API tokens.

#### Step 3: Create GitHub Environments

Create two GitHub Environments for the workflow:

1. Go to https://github.com/subhash-holla/pii-anon/settings/environments
2. Create environment `testpypi`
   - Add repository secret `TEST_PYPI_API_TOKEN` if using API tokens (optional, OIDC preferred)
3. Create environment `pypi`
   - Add repository secret `PYPI_API_TOKEN` if using API tokens (optional, OIDC preferred)
   - (Optional) Add a "Required reviewers" protection rule for safety on production releases

**OIDC Advantages:**
- No credentials stored in GitHub
- Time-limited tokens (15 minutes)
- Automatic token refresh
- Audit trail through GitHub Actions

**Workflow Behavior:**
- Tags like `v1.0.0-rc1` → publishes to TestPyPI (requires `testpypi` environment)
- Tags like `v1.0.0` (no `-rc`) → publishes to PyPI (requires `pypi` environment)
- Manual dispatch via Actions UI can target either TestPyPI or PyPI

---

## Part 6: Publishing to TestPyPI

### 6.1 Create your TestPyPI and PyPI accounts (if needed)

If you don't already have accounts, register at both (use the same email for simplicity):

- TestPyPI: https://test.pypi.org/account/register/
- PyPI: https://pypi.org/account/register/

Enable 2FA on both accounts for security.

### 6.2 Create API tokens (fallback option if not using OIDC)

If you're not using OIDC (see Part 5.6), you'll need API tokens for local uploads or as GitHub repository secrets.

**TestPyPI:**

1. Go to https://test.pypi.org/manage/account/#api-tokens
2. Click "Add API token"
3. Token name: `pii-anon-upload`
4. Scope: "Entire account" (first upload) — you can scope it to the project later
5. Copy the token (starts with `pypi-`)

**PyPI:** Same process at https://pypi.org/manage/account/#api-tokens

**Note:** OIDC is preferred for GitHub Actions (see Part 5.6). API tokens are mainly needed for local `twine` uploads.

### 6.3 Configure ~/.pypirc for local uploads

```bash
cat > ~/.pypirc << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
EOF

chmod 600 ~/.pypirc
```

Replace the placeholder tokens with the real ones you generated.

### 6.4 Build the packages

Build in the correct order — datasets first, since the main library depends on it:

```bash
# Build pii-anon-datasets
cd ~/projects/pii-anon/pii-anon-eval-data
rm -rf dist/ build/ src/*.egg-info
python3 -m build --outdir dist
twine check dist/*

# Build pii-anon
cd ~/projects/pii-anon/pii-anon-code
rm -rf dist/ build/ src/*.egg-info
python3 -m build --outdir dist
twine check dist/*
python3 scripts/check_package_size.py --dist-dir dist --max-wheel-mb 1.5 --package-name pii_anon
```

**Verify versions match in both projects:**
- `pii-anon-code/pyproject.toml` → `version = "1.0.0"`
- `pii-anon-eval-data/pyproject.toml` → `version = "1.0.0"`
- Both should have matching version numbers for the release

### 6.5 Upload to TestPyPI

Upload datasets first since the main library depends on it:

```bash
# Upload pii-anon-datasets
cd ~/projects/pii-anon/pii-anon-eval-data
twine upload --repository testpypi dist/*

# Upload pii-anon
cd ~/projects/pii-anon/pii-anon-code
twine upload --repository testpypi dist/*
```

---

## Part 6.5+ Distribution Artifacts for v1.0.0 Release

Before verification, understand what artifacts are being released:

### Artifact Overview

| Artifact | Repository | Package | Destination | Access |
|---|---|---|---|---|
| **pii-anon library** | `pii-anon-code` | `pii-anon` | PyPI | `pip install pii-anon` |
| **Evaluation datasets** | `pii-anon-eval-data` | `pii-anon-datasets` | PyPI | `pip install pii-anon-datasets` or auto-included via `pip install pii-anon[datasets]` |
| **Documentation** | `pii-anon-doc` | (static docs) | GitHub | https://github.com/subhash-holla/pii-anon-doc |
| **GitHub Releases** | All 3 repos | (release notes) | GitHub | GitHub Releases for each repo |

### Dataset Availability for Evaluation

The `pii-anon-datasets` package provides JSONL benchmark data for anyone running evaluations. It's distributed via PyPI and can be accessed in three ways:

#### Option 1: Auto-included with pii-anon[datasets] (Recommended)

```bash
# Installs both pii-anon and pii-anon-datasets automatically
pip install "pii-anon[datasets]"
```

This is the easiest approach for users who want evaluation capability out-of-the-box.

#### Option 2: Install datasets separately

```bash
# Install just the datasets package
pip install pii-anon-datasets

# Then import and use in your code
from pii_anon.eval_framework import load_dataset
dataset = load_dataset('pii_anon_benchmark_v1')
```

This is useful for evaluation-only workflows that don't need the full pii-anon library.

#### Option 3: Access via environment variable (development/offline)

```bash
# During development or in environments without PyPI access
export PII_ANON_DATASET_ROOT=/path/to/pii-anon-eval-data/src/pii_anon_datasets/data
python your_eval_script.py
```

### Dataset Resolution Order

When pii-anon looks for evaluation data, it checks in this order:

1. **Installed `pii_anon_datasets` package** (from PyPI or local install)
2. **`PII_ANON_DATASET_ROOT` environment variable** (if set)
3. **Sibling `pii-anon-eval-data/` repo** (local development)
4. **Monorepo fallback** (legacy layout)

This allows both production (PyPI) and development (local repo) workflows.

### Documentation Links

The main `pii-anon` repository references the documentation repo:

- **Primary docs:** https://github.com/subhash-holla/pii-anon-doc
- **In pyproject.toml:** `Documentation = "https://github.com/subhash-holla/pii-anon-doc"`
- **In README.md:** Link to detailed docs in the doc repo
- **In GitHub Release:** Release notes include link to documentation

Ensure the `pii-anon-doc` repo is fully updated before the v1.0.0 release with:
- Complete API documentation
- Evaluation framework guide
- Benchmark setup and interpretation
- Entity type registry
- Compliance mapping (GDPR, HIPAA, NIST, etc.)

---

## Part 7: Verifying on TestPyPI

### 7.1 Install in a clean environment

```bash
python3 -m venv /tmp/pii-anon-test-env
source /tmp/pii-anon-test-env/bin/activate
pip install --upgrade pip

# Install datasets first from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pii-anon-datasets

# Then install main library with datasets extra (includes datasets dependency)
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    "pii-anon[cli,datasets]"
```

The `--extra-index-url` fallback is essential — TestPyPI won't have `pydantic`, `typer`, and other transitive dependencies.

**Verification of dataset auto-inclusion:**

```bash
# Verify pii-anon-datasets is installed as a dependency
pip show pii-anon
# Should list "pii-anon-datasets" in the Requires field

# Verify dataset access
python3 -c "from pii_anon.eval_framework import load_dataset; \
    ds = load_dataset('pii_anon_benchmark_v1'); \
    print(f'Dataset loaded: {len(ds)} records')"
```

### 7.2 Run smoke tests

```bash
# Version check
python3 -c "import pii_anon; print(f'Version: {pii_anon.__version__}')"
# Should print: Version: 1.0.0

# CLI smoke
pii-anon version
pii-anon health --output json

# Detection
pii-anon detect "Contact alice@example.com and call 415-555-0100"

# Orchestrator
python3 -c "
from pii_anon import PIIOrchestrator
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan
orch = PIIOrchestrator(token_key='test-key')
result = orch.run(
    {'text': 'Contact alice@example.com and 555-12-3456'},
    profile=ProcessingProfileSpec(profile_id='default', mode='weighted_consensus', language='en'),
    segmentation=SegmentationPlan(enabled=False),
    scope='etl', token_version=1,
)
print('Findings:', len(result['ensemble_findings']))
print('Output:', result['transformed_payload'][:80])
"

# Evaluation framework
python3 -c "
from pii_anon.eval_framework import PII_TAXONOMY, SUPPORTED_LANGUAGES, EVIDENCE_REGISTRY
print(f'Entity types: {len(PII_TAXONOMY)}')
print(f'Languages: {len(SUPPORTED_LANGUAGES)}')
print(f'Research references: {len(EVIDENCE_REGISTRY)}')
"

# Composite scoring
python3 -c "
from pii_anon.eval_framework import compute_composite
score = compute_composite(f1=0.612, precision=0.580, recall=0.647, latency_ms=6.942, docs_per_hour=514300)
print(f'Composite: {score.score:.4f}')
print(f'Tier 1 (accuracy): {score.tier1_score:.4f}')
print(f'Tier 2 (operational): {score.tier2_score:.4f}')
"

# Governance thresholds
python3 -c "
from pii_anon.eval_framework import GovernanceThresholds, evaluate_governance
thresholds = GovernanceThresholds()
result = evaluate_governance(f1=0.90, precision=0.91, recall=0.88, latency_p50_ms=20.0, docs_per_hour=12000)
print(f'Governance pass: {result.passed}')
print(f'Checks: {len(result.checks)}')
"
```

### 7.3 Verify on the TestPyPI website

```bash
open https://test.pypi.org/project/pii-anon/
open https://test.pypi.org/project/pii-anon-datasets/
```

Check: README renders correctly, version shows 1.0.0, classifiers and project links are right.

### 7.4 Clean up

```bash
deactivate
rm -rf /tmp/pii-anon-test-env
```

---

## Part 8: Promoting to PyPI (Production)

Only proceed here after TestPyPI verification passes cleanly.

### 8.1 Option A: Upload manually with twine

```bash
# Datasets first
cd ~/projects/pii-anon/pii-anon-eval-data
twine upload dist/*

# Main library
cd ~/projects/pii-anon/pii-anon-code
twine upload dist/*
```

### 8.2 Option B: Trigger via GitHub Actions (Recommended with OIDC)

If you set up the environments in Part 5.6, pushing a version tag triggers the release workflow automatically.

**Automatic (tag-based):**

```bash
# In pii-anon-code repo
git tag v1.0.0-rc1
git push origin v1.0.0-rc1
# → Automatically publishes to TestPyPI

# Later, when ready for production:
git tag v1.0.0
git push origin v1.0.0
# → Automatically publishes to PyPI
```

**Manual (workflow dispatch):**

1. Go to https://github.com/subhash-holla/pii-anon/actions/workflows/release.yml
2. Click "Run workflow"
3. Choose target environment: `testpypi` or `pypi`
4. Click "Run workflow"

The workflow will:
- Build the package (runs full test suite, benchmarks, and validations)
- Publish to the chosen PyPI registry using OIDC (or API token fallback)
- Verify package integrity and README rendering

**For pii-anon-datasets:** The datasets package is published separately. Each repo (`pii-anon-code` and `pii-anon-eval-data`) can be released independently or together, depending on whether the datasets change.

### 8.3 Verify on PyPI

```bash
python3 -m venv /tmp/pii-anon-prod-verify
source /tmp/pii-anon-prod-verify/bin/activate
pip install --upgrade pip

# Install from production PyPI
pip install "pii-anon[cli,datasets]"

# Run the same smoke tests from Part 7.2
python3 -c "import pii_anon; print(f'Version: {pii_anon.__version__}')"
pii-anon version
pii-anon health --output json
pii-anon detect "Contact alice@example.com and call 415-555-0100"

# Verify dataset is included and working
python3 -c "from pii_anon.eval_framework import load_dataset; \
    ds = load_dataset('pii_anon_benchmark_v1'); \
    print(f'Dataset accessible: {len(ds)} records')"

deactivate
rm -rf /tmp/pii-anon-prod-verify
```

### 8.4 Verify on the PyPI website

```bash
open https://pypi.org/project/pii-anon/
open https://pypi.org/project/pii-anon-datasets/
```

---

## Quick Reference Checklist

```
1. MAC SETUP
   [ ] Homebrew, Python 3.12, Git, gh CLI installed
   [ ] gh auth login completed

2. REPO SETUP
   [ ] Three GitHub repos created (pii-anon, pii-anon-doc, pii-anon-eval-data)
   [ ] Local folders cloned or linked to remotes
   [ ] venv created and activated
   [ ] PyTorch CPU installed, then pip install -e ".[dev,cli,crypto,benchmark,datasets]" succeeded
   [ ] NLP models downloaded (spaCy en_core_web_sm, Stanza en, GLiNER)
   [ ] pii-anon benchmark-preflight shows all 3 competitors

3. DOCUMENTATION (pii-anon-doc repo)
   [ ] API documentation complete and current
   [ ] Evaluation framework guide up-to-date
   [ ] Benchmark setup and interpretation documented
   [ ] Entity type registry documented
   [ ] Compliance mapping (GDPR, HIPAA, NIST, PCI-DSS, CCPA, ISO) complete
   [ ] All examples working and tested
   [ ] README links to documentation repo

4. TESTS
   [ ] make all passes (lint + type + tests + perf + build + checks)
   [ ] 1052+ tests passing, ~90% coverage, 85% gate met

5. EVALUATION
   [ ] make benchmark-portable (or benchmark-canonical-macos for full gates)
   [ ] make benchmark-comprehensive (all 4 engine tiers × all datasets)
   [ ] All 3 competitors evaluated (presidio, scrubadub, gliner)
   [ ] All engine tiers evaluated (auto, minimal, standard, full)
   [ ] All datasets evaluated (pii_anon_benchmark_v1, eval_framework_v1)
   [ ] Cross-dataset combined report reviewed (benchmark-combined.json)
   [ ] Per-tier performance differences reviewed
   [ ] Results reviewed in artifacts/benchmarks/

6. VERSION SYNC
   [ ] pii-anon-code/pyproject.toml → version = "1.0.0"
   [ ] pii-anon-eval-data/pyproject.toml → version = "1.0.0"
   [ ] pii-anon-code/src/pii_anon/__init__.py → __version__ = "1.0.0"
   [ ] pii-anon-code/pyproject.toml dependencies → pii-anon-datasets>=1.0.0

7. GITHUB
   [ ] All three repos pushed to main (pii-anon-code, pii-anon-doc, pii-anon-eval-data)
   [ ] Tags created for both packages (v1.0.0 in both pii-anon-code and pii-anon-eval-data)
   [ ] GitHub Releases created for all three repos with descriptive notes
   [ ] Documentation links included in release notes
   [ ] OIDC/Trusted Publishing configured (Part 5.6)
   [ ] Environments created (testpypi, pypi) with OIDC or API token secrets

8. BUILD & VALIDATION
   [ ] Both packages built in order: pii-anon-datasets, then pii-anon
   [ ] twine check dist/* passes for both packages
   [ ] Package sizes validated (pii-anon wheel < 1.5 MB)

9. TESTPYPI
   [ ] pii-anon-datasets uploaded to TestPyPI
   [ ] pii-anon uploaded to TestPyPI
   [ ] Installed in clean venv from TestPyPI with [cli,datasets] extras
   [ ] Smoke tests pass:
      - import pii_anon (check version is 1.0.0)
      - pii-anon CLI commands work (version, health, detect)
      - Dataset auto-included and accessible via load_dataset()
      - Orchestrator works
      - Eval framework works
      - Composite scoring works
      - Governance thresholds work
   [ ] TestPyPI project pages look correct for both packages
   [ ] README renders without errors on TestPyPI

10. PYPI (only after TestPyPI is verified)
    [ ] pii-anon-datasets uploaded to PyPI
    [ ] pii-anon uploaded to PyPI
    [ ] Installed in clean venv from PyPI with [cli,datasets] extras
    [ ] All smoke tests from step 9 pass on production PyPI
    [ ] pypi.org project pages look correct for both packages
    [ ] Users can install with: pip install "pii-anon[datasets]"
    [ ] Users can install just datasets with: pip install pii-anon-datasets

11. POST-RELEASE
    [ ] Monitor for any issues reported on GitHub
    [ ] Update links/references in related projects
    [ ] Announce release (email list, social media, etc.)
    [ ] Archive benchmark results
```

---

## Part 9: Complete v1.0.0 Artifact Checklist

After all publishing steps are complete, verify that all v1.0.0 artifacts are available and accessible:

### Artifacts to Verify

#### 1. PyPI Artifacts

**pii-anon library:**
```bash
# Visit https://pypi.org/project/pii-anon/
# Verify:
# - Version: 1.0.0
# - Package size: wheel < 1.5 MB, reasonable sdist size
# - README renders correctly
# - Project URLs point to:
#   - Homepage: https://github.com/subhash-holla/pii-anon
#   - Documentation: https://github.com/subhash-holla/pii-anon-doc
#   - Repository: https://github.com/subhash-holla/pii-anon
#   - Issues: https://github.com/subhash-holla/pii-anon/issues
# - Classifiers include: Development Status :: 5 - Production/Stable
# - Dependencies list: pydantic>=2.8
# - Optional dependencies (datasets) available in extras

pip install "pii-anon[cli,datasets]" --dry-run
# Should resolve successfully
```

**pii-anon-datasets package:**
```bash
# Visit https://pypi.org/project/pii-anon-datasets/
# Verify:
# - Version: 1.0.0
# - README shows datasets available
# - Project URLs point to:
#   - Homepage: https://github.com/subhash-holla/pii-anon-eval-data
#   - Repository: https://github.com/subhash-holla/pii-anon-eval-data
# - Classifiers include: Development Status :: 5 - Production/Stable

pip install pii-anon-datasets --dry-run
# Should resolve successfully
```

#### 2. GitHub Releases

**pii-anon library release:**
```bash
# Visit https://github.com/subhash-holla/pii-anon/releases/tag/v1.0.0
# Verify:
# - Tag: v1.0.0
# - Release title: "pii-anon v1.0.0"
# - Release notes include link to documentation: https://github.com/subhash-holla/pii-anon-doc
# - No release assets needed (packages on PyPI, not GitHub)
```

**pii-anon-datasets release:**
```bash
# Visit https://github.com/subhash-holla/pii-anon-eval-data/releases/tag/v1.0.0
# Verify:
# - Tag: v1.0.0
# - Release title: "pii-anon-datasets v1.0.0"
# - Release notes mention installation via pip and auto-inclusion with pii-anon[datasets]
```

**pii-anon-doc release:**
```bash
# Visit https://github.com/subhash-holla/pii-anon-doc/releases/tag/v1.0.0
# Verify:
# - Tag: v1.0.0
# - Release title: "pii-anon Documentation v1.0.0"
# - Release notes include overview of documentation content
```

#### 3. Documentation

**pii-anon-doc repository:**
```bash
# Visit https://github.com/subhash-holla/pii-anon-doc
# Verify main branch contains:
# - API reference documentation
# - Evaluation framework guide
# - Benchmark setup and interpretation
# - Entity type registry
# - Compliance mapping documentation
# - Usage examples
# - Architecture diagrams
```

### Integration Verification

Test the full integration workflow:

```bash
# Test 1: Fresh install with datasets
python3 -m venv /tmp/final-check
source /tmp/final-check/bin/activate
pip install --upgrade pip

pip install "pii-anon[cli,datasets]"

# Verify library works
pii-anon version  # Should print: 1.0.0
pii-anon capabilities --output json | head -50

# Verify datasets are included
python3 << 'EOF'
from pii_anon.eval_framework import load_dataset
datasets = ['pii_anon_benchmark_v1', 'eval_framework_v1']
for ds_name in datasets:
    try:
        ds = load_dataset(ds_name)
        print(f"✓ {ds_name}: {len(ds)} records")
    except Exception as e:
        print(f"✗ {ds_name}: {e}")
EOF

deactivate

# Test 2: Datasets-only install
python3 -m venv /tmp/datasets-only
source /tmp/datasets-only/bin/activate
pip install --upgrade pip

pip install pii-anon-datasets

python3 << 'EOF'
from pii_anon_datasets import get_dataset_path
path = get_dataset_path('pii_anon_benchmark_v1')
print(f"✓ Dataset path accessible: {path}")
EOF

deactivate

# Test 3: Verify documentation link
python3 << 'EOF'
import pii_anon
# Check that metadata points to documentation
import importlib.metadata
meta = importlib.metadata.metadata('pii-anon')
doc_url = meta.get('Project-URL', '').split(',')
print(f"✓ Documentation URL available in package metadata")
EOF

rm -rf /tmp/final-check /tmp/datasets-only
```

### Distribution Accessibility Summary

After v1.0.0 release, users can access the artifacts via:

1. **Library (pii-anon):** `pip install pii-anon[datasets]` from PyPI
2. **Datasets (pii-anon-datasets):**
   - Via PyPI: `pip install pii-anon-datasets`
   - Auto-included: `pip install pii-anon[datasets]`
3. **Documentation:** https://github.com/subhash-holla/pii-anon-doc (GitHub repo)
4. **GitHub Releases:** All three repos have v1.0.0 releases with notes

---

## Troubleshooting

**`gh repo create` says "already exists":**
That's fine. The repo is already set up — just make sure your local remote points to it: `git remote -v`. If not, run `git remote add origin https://github.com/subhash-holla/pii-anon.git`.

**`git push` is rejected (non-fast-forward):**
If the remote has content you don't have locally: `git pull --rebase origin main` first, resolve any conflicts, then push again.

**"spaCy model not found" during Presidio evaluation:**
Run `python3 -m spacy download en_core_web_sm`. If that fails behind a firewall, download manually from the spaCy releases page and install with `pip install /path/to/en_core_web_sm-x.x.x.tar.gz`.

**"stanza model not found":**
Run `python3 -c "import stanza; stanza.download('en')"`. Models download to `~/stanza_resources/`.

**`Failed to build thinc` / `Failed to build spacy`:**
This means your Python version is too new for pre-built wheels. spaCy and thinc need **Python 3.12 or lower**. Fix: `deactivate && rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate` then re-run the install steps.

**TestPyPI `pip install` fails on dependencies:**
Always use `--extra-index-url https://pypi.org/simple/` so pip can find transitive dependencies (pydantic, typer, etc.) on real PyPI.

**"Package already exists" on PyPI or TestPyPI:**
You cannot overwrite a published version. Bump the version in both `pyproject.toml` and `src/pii_anon/__init__.py`, rebuild, and re-upload.

**Docker not found (canonical macOS benchmark):**
Install Docker Desktop: `brew install --cask docker`. Start Docker Desktop from Applications. The canonical benchmark uses Docker to run on linux/amd64 for reproducibility on Apple Silicon.

**`docker` command runs a Node.js tool instead of Docker Engine:**
If you see errors referencing `node_modules/docker/` or `highlight.js`, you have an npm package called `docker` (a documentation generator) shadowing the real Docker CLI. Fix: `npm uninstall -g docker`. Verify with `which docker` — it should point to `/usr/local/bin/docker` or `/opt/homebrew/bin/docker`, not an nvm path. Alternatively, override the path: `make benchmark-canonical-macos DOCKER=/opt/homebrew/bin/docker`.

**Docker benchmark is very slow on Apple Silicon:**
Use `make benchmark-canonical-macos-native` instead — it runs natively on ARM64 without Rosetta emulation and is 2-3x faster. If you must use Docker, ensure Docker Desktop → Settings → General → "Use Rosetta for x86_64/amd64 emulation on Apple Silicon" is enabled. Allocate at least 8 GB RAM and 4 CPUs to Docker (Settings → Resources).

**tqdm progress bar not rendering in Docker:**
The Makefile passes `-e TERM=$(TERM)` and `-e PYTHONUNBUFFERED=1` to Docker. If progress bars still don't render, try `TERM=xterm make benchmark-canonical-macos`.

**`pip install gliner` fails with build errors:**
GLiNER requires a C compiler for some transitive deps. On macOS, ensure Xcode command line tools are installed: `xcode-select --install`. If the build still fails, try `pip install --no-build-isolation gliner`.

**Coverage below 85%:**
The `pyproject.toml` enforces `--cov-fail-under=85`. Current coverage is ~90%. If it drops, run `python3 -m pytest tests/ --cov=pii_anon --cov-report=html` and open `htmlcov/index.html` to inspect uncovered lines.

**GitHub Actions release workflow fails:**
Check that both environment secrets (`TEST_PYPI_API_TOKEN`, `PYPI_API_TOKEN`) are set and that the `testpypi` and `pypi` environments exist in your repo settings. If using OIDC, verify trusted publishers are configured correctly in PyPI/TestPyPI account settings.

**Documentation repo release:**
The `pii-anon-doc` repo doesn't publish packages to PyPI (it contains static documentation). To release it:

```bash
cd ~/projects/pii-anon/pii-anon-doc
git add -A
git commit -m "v1.0.0: Documentation update for pii-anon release"
git branch -M main
git push -u origin main
git tag -a v1.0.0 -m "v1.0.0: Complete documentation for pii-anon library and evaluation framework"
git push origin v1.0.0
gh release create v1.0.0 \
    --title "pii-anon Documentation v1.0.0" \
    --notes "Comprehensive documentation for pii-anon library, evaluation framework, and benchmarks. See README for content overview."
```

The documentation repo is primarily accessed via GitHub (https://github.com/subhash-holla/pii-anon-doc), not PyPI.

**"Inconsistent versions across repos":**
If PyPI publishes succeed but versions are inconsistent, verify:
- `pii-anon-code/pyproject.toml` version matches git tag
- `pii-anon-eval-data/pyproject.toml` version matches git tag
- Both packages list matching versions in dependencies (e.g., `pii-anon-datasets>=1.0.0`)
- `pii-anon-code/src/pii_anon/__init__.py` has `__version__ = "1.0.0"`

Rebuild and re-upload if there's a mismatch.
