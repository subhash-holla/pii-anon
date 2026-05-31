"""Import-boundary CI guard for S3-05 (NFR eval-integrity isolation).

Design trace: D-EVAL DECISION 2 — the rating layer must be import-isolated
from the detection/orchestration layers. No module under
``pii_anon/eval_framework/rating/`` may import from ``pii_anon.swarm``,
``pii_anon.moe``, ``pii_anon.fusion`` or ``pii_anon.policy``.

The check is AST-based (not substring) so it cannot false-positive on
docstrings, comments, or identifiers that merely mention a forbidden name.
This test is GREEN on introduction — a prior grep found zero violations —
and acts as a standing CI gate against future regressions.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pii_anon.eval_framework.rating as rating_pkg

FORBIDDEN_ROOTS: frozenset[str] = frozenset(
    {
        "pii_anon.swarm",
        "pii_anon.moe",
        "pii_anon.fusion",
        "pii_anon.policy",
    }
)


def _rating_module_paths() -> list[Path]:
    pkg_dir = Path(rating_pkg.__file__).parent
    return sorted(p for p in pkg_dir.glob("*.py"))


def _module_head_is_forbidden(module: str | None) -> bool:
    """True if a dotted module path is, or is a descendant of, a forbidden root."""
    if not module:
        return False
    for root in FORBIDDEN_ROOTS:
        if module == root or module.startswith(root + "."):
            return True
    return False


def _collect_imported_modules(tree: ast.AST) -> list[str]:
    """Return every absolute module path referenced by Import / ImportFrom.

    Relative imports (level > 0) cannot reach a sibling top-level package
    like ``pii_anon.swarm`` from within ``eval_framework.rating`` without an
    explicit ``pii_anon`` head, so we only inspect absolute references.
    """
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.append(node.module)
    return modules


def test_s3_05_rating_layer_has_no_forbidden_imports() -> None:
    """AST-walk every rating module; assert no Import/ImportFrom resolves to a
    forbidden top-level package. GREEN today; standing regression gate."""
    violations: list[str] = []
    for path in _rating_module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _collect_imported_modules(tree):
            if _module_head_is_forbidden(module):
                violations.append(f"{path.name}: imports {module}")

    assert not violations, (
        "rating layer must not import detection/orchestration packages: "
        + "; ".join(violations)
    )


def test_s3_05_at_least_one_rating_module_scanned() -> None:
    """Guard against a no-op pass if the glob ever silently matches nothing."""
    paths = _rating_module_paths()
    assert paths, "expected at least one *.py under eval_framework/rating/"
    assert any(p.name == "__init__.py" for p in paths)
