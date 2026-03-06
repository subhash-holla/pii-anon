from __future__ import annotations

from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_dependencies_doc_recommends_venv_for_all_os() -> None:
    doc = _read("docs/dependencies-and-platforms.md")
    assert "Recommended for all OS: use a virtual environment" in doc
    assert "Use `venv` on every operating system" in doc


def test_dependencies_doc_has_non_venv_fallback_section() -> None:
    doc = _read("docs/dependencies-and-platforms.md")
    assert "OS-specific fallback (only if `venv` cannot be used)" in doc
    assert "macOS / Linux fallback:" in doc
    assert "Windows PowerShell fallback:" in doc
    assert "Windows CMD fallback:" in doc
    assert "python3.11 -m pip install --user pii-anon" in doc
    assert "py -3.11 -m pip install --user pii-anon" in doc


def test_dependencies_doc_keeps_venv_activation_commands() -> None:
    doc = _read("docs/dependencies-and-platforms.md")
    assert "python3.11 -m venv .venv" in doc
    assert "source .venv/bin/activate" in doc
    assert ".venv\\Scripts\\Activate.ps1" in doc
    assert ".venv\\Scripts\\activate.bat" in doc


def test_readme_references_venv_and_fallback_doc() -> None:
    readme = _read("README.md")
    assert "Use a virtual environment (`venv`) on any OS" in readme
    assert "docs/dependencies-and-platforms.md" in readme
