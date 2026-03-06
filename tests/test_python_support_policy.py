from __future__ import annotations

from pathlib import Path


def test_pyproject_python_classifiers_match_support_policy() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert '"Programming Language :: Python :: 3.10"' in text
    assert '"Programming Language :: Python :: 3.11"' in text
    assert '"Programming Language :: Python :: 3.12"' in text
    assert '"Programming Language :: Python :: 3.13"' in text
    assert '"Programming Language :: Python :: 3.14"' not in text
    assert 'requires-python = ">=3.10"' in text


def test_ci_has_blocking_matrix_for_3_10_to_3_13() -> None:
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'python-version: ["3.10", "3.11", "3.12", "3.13"]' in ci


def test_ci_has_non_blocking_python_3_14_experimental_job() -> None:
    ci = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "python-3-14-experimental:" in ci
    assert "continue-on-error: true" in ci
    assert 'python-version: "3.14"' in ci
    assert "allow-prereleases: true" in ci
