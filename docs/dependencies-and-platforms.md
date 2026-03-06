# Dependencies and Platform Support

## Supported runtimes

- Operating systems:
  - macOS
  - Linux
  - Windows
- Python:
  - Supported: `3.10`, `3.11`, `3.12`, `3.13`
  - Experimental: `3.14`

## Canonical publish-grade benchmark runtime

Benchmark claims intended for release publication require:
- Linux or macOS runtime
- shared-memory support (`/dev/shm` on Linux; kernel `shm_open` on macOS)
- all configured competitors installed and native-ready
- installed datasets package resources (`pii-anon-datasets`)

Use these commands before publishing metrics:

```bash
pii-anon benchmark-preflight --output json
pii-anon benchmark-publish-suite --artifacts-dir artifacts/benchmarks --output json
```

## Recommended for all OS: use a virtual environment

Use `venv` on every operating system to keep dependency resolution and benchmark behavior consistent.
The OS-specific install fallback section below should only be used when `venv` cannot be used in your environment.

### Create and activate a venv

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Windows CMD:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

## OS-specific fallback (only if `venv` cannot be used)

These commands install into the user site-packages and are less isolated than `venv`.

macOS / Linux fallback:

```bash
python3.11 -m pip install --user --upgrade pip
python3.11 -m pip install --user pii-anon
```

Windows PowerShell fallback:

```powershell
py -3.11 -m pip install --user --upgrade pip
py -3.11 -m pip install --user pii-anon
```

Windows CMD fallback:

```bat
py -3.11 -m pip install --user --upgrade pip
py -3.11 -m pip install --user pii-anon
```

## Install profiles

Core library only:

```bash
pip install pii-anon
```

Core + CLI + crypto:

```bash
pip install "pii-anon[cli,crypto]"
```

Core + optional engines:

```bash
pip install "pii-anon[engines]"
```

Core + datasets package:

```bash
pip install "pii-anon[datasets]"
```

Benchmark/full stack including `llm-guard`:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2,<2.6"
pip install "pii-anon[engines,llm-guard,cli,crypto,datasets,dev]"
```

## Optional engine dependencies

- `regex-oss`: no extra dependency.
- `presidio-compatible`: `presidio-analyzer`.
- `scrubadub-compatible`: `scrubadub`.
- `spacy-ner-compatible`: `spacy`.
- `stanza-ner-compatible`: `stanza`.
- `llm-guard-compatible`: `llm-guard` and `torch`.

Recommended model bootstrap for better coverage:

```bash
python -m spacy download xx_ent_wiki_sm
python -c "import stanza; stanza.download('en')"
```

For additional Stanza languages:

```bash
python -c "import stanza; [stanza.download(lang) for lang in ('es','fr','de','ja','ar','hi','zh','ko')]"
```

## Verify installation

```bash
python - <<'PY'
import pii_anon
from pii_anon import PIIOrchestrator
print("pii_anon version:", pii_anon.__version__)
orch = PIIOrchestrator(token_key="check")
print("engines:", orch.list_engines())
print("capabilities:", orch.capabilities().keys())
PY
```

CLI check:

```bash
pii-anon version
pii-anon health --output json
```

## Notes for `llm-guard`

- `llm-guard` availability depends on `torch` wheel support for your Python/OS.
- If `llm-guard` cannot be installed on your platform, benchmark/comparison runs will mark it as unavailable and continue with diagnostics.
- For canonical publish-grade workflows, all configured competitors (including `llm-guard`) are required; run these gates in Linux with Python `3.11` and preinstall `torch` before extras.

## Common build issues

- If a wheel is unavailable and pip tries to build from source:
  - macOS: install Xcode command line tools (`xcode-select --install`).
  - Linux: install build tools (`build-essential`, `python3-dev`).
  - Windows: install Microsoft C++ Build Tools.
