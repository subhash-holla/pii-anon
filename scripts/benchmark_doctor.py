"""Cross-platform environment diagnostics for ``make benchmark-*`` targets.

Prints Python version, OS, and the availability of every optional
benchmark dependency (pii-anon-datasets, Presidio, scrubadub, GLiNER,
spaCy, Stanza, XGBoost).  Exits 0 when the environment is usable for
`make benchmark-all` (the friendly target that tolerates missing
competitors), and surfaces actionable install hints for anything that
is missing.

Invoked from the Makefile ``benchmark-doctor`` target but also
runnable standalone — helpful when debugging a benchmark failure on a
machine you've never configured before.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

# (module_name, label, optional, install_hint_when_missing)
_DEPS: list[tuple[str, str, bool, str]] = [
    ("pii_anon", "pii-anon library", False, "pip install -e ."),
    ("pydantic", "pydantic (core dep)", False, "pip install pydantic>=2.8"),
    ("pii_anon_datasets", "pii-anon-datasets", True, None),  # hint depends on eval-data-dir
    ("presidio_analyzer", "presidio-analyzer (competitor)", True, "pip install 'pii-anon[engines]'"),
    ("scrubadub", "scrubadub (competitor)", True, "pip install scrubadub"),
    ("gliner", "GLiNER (competitor)", True, "pip install 'pii-anon[engines]'"),
    ("spacy", "spaCy (core NER)", True, "pip install 'pii-anon[engines]' && python -m spacy download en_core_web_sm"),
    ("stanza", "Stanza (core NER)", True, "pip install 'pii-anon[engines]'"),
    ("xgboost", "XGBoost (swarm meta-learner)", True, "pip install 'pii-anon[swarm-ml]'"),
]


def _check_module(mod_name: str) -> tuple[bool, str | None]:
    """Return ``(available, version)`` for *mod_name*."""
    try:
        mod = importlib.import_module(mod_name)
    except Exception:
        return False, None
    version = getattr(mod, "__version__", None)
    if isinstance(version, str):
        return True, version
    return True, None


def _hint_for_pii_anon_datasets(eval_data_dir: Path) -> str:
    if eval_data_dir.exists():
        return f"pip install -e {eval_data_dir}  # or: pip install pii-anon-datasets"
    return "pip install pii-anon-datasets  # once published, or clone pii-anon-eval-data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark environment diagnostics")
    parser.add_argument(
        "--eval-data-dir",
        type=Path,
        default=Path("../pii-anon-eval-data"),
        help="Path to the pii-anon-eval-data checkout (when using a local clone).",
    )
    args = parser.parse_args()

    print(f"{'sys.executable:':<18}{sys.executable}")
    print(f"{'Python version:':<18}{sys.version.split()[0]}")
    print(f"{'OS:':<18}{platform.platform()}")
    print(f"{'Machine:':<18}{platform.machine()}")
    print(f"{'Eval data dir:':<18}{args.eval_data_dir}  ({'exists' if args.eval_data_dir.exists() else 'NOT FOUND'})")
    print()
    print(f"{'Component':<38}{'Status':<24}{'Version'}")
    print("-" * 78)

    missing_required: list[tuple[str, str]] = []
    missing_optional: list[tuple[str, str]] = []

    for mod_name, label, optional, install_hint in _DEPS:
        available, version = _check_module(mod_name)
        if available:
            status = "installed"
            ver_str = version or "—"
        else:
            status = "MISSING (optional)" if optional else "MISSING (REQUIRED)"
            ver_str = "—"
            hint = install_hint
            if mod_name == "pii_anon_datasets":
                hint = _hint_for_pii_anon_datasets(args.eval_data_dir)
            if optional:
                missing_optional.append((label, hint or "see docs/release-guide.md"))
            else:
                missing_required.append((label, hint or "see docs/release-guide.md"))
        print(f"{label:<38}{status:<24}{ver_str}")

    print()
    if missing_required:
        print("REQUIRED dependencies missing — the benchmark will NOT run:")
        for label, hint in missing_required:
            print(f"  - {label}")
            print(f"    Fix: {hint}")
        return 1

    if missing_optional:
        print("Optional dependencies missing — the benchmark will still run,")
        print("but the affected systems will be excluded from the leaderboard:")
        for label, hint in missing_optional:
            print(f"  - {label}  →  {hint}")
        print()
        print("Run `make benchmark-all` anyway — it tolerates missing competitors.")
    else:
        print("All optional dependencies present — ready for `make benchmark-canonical`.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
