"""Import-boundary CI guard for the DC-09 ``attacks/`` package (S5-01).

Design trace: DC-09 attacks import-isolation invariant — the adversarial-attack
package must be import-isolated from the detection/orchestration layers. No
module under ``pii_anon/eval_framework/attacks/`` may import from
``pii_anon.swarm``, ``pii_anon.moe``, ``pii_anon.fusion`` or ``pii_anon.policy``.
The ``attacks/__init__`` docstring already promises this guard; S5-01 lands it as
a standing CI gate (the package boundary + the protocol family land together).

The check is AST-based (not substring) so it cannot false-positive on
docstrings, comments, or identifiers that merely mention a forbidden name. This
test is GREEN on introduction — the S5-04 substrate + the S5-01 reid body import
only stdlib + ``eval_framework``/``harness`` siblings — and acts as a standing CI
gate against future regressions (A10).

This module is a verbatim adaptation of ``tests/test_rating_import_boundary.py``
repointed at ``pii_anon.eval_framework.attacks``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pii_anon.eval_framework.attacks as attacks_pkg

FORBIDDEN_ROOTS: frozenset[str] = frozenset(
    {
        "pii_anon.swarm",
        "pii_anon.moe",
        "pii_anon.fusion",
        "pii_anon.policy",
    }
)


def _attacks_module_paths() -> list[Path]:
    pkg_dir = Path(attacks_pkg.__file__).parent
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

    Relative imports (level > 0) cannot reach a sibling top-level package like
    ``pii_anon.swarm`` from within ``eval_framework.attacks`` without an explicit
    ``pii_anon`` head, so we only inspect absolute references.
    """
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                modules.append(node.module)
    return modules


def test_ax_isolation_attacks_layer_has_no_forbidden_imports() -> None:
    """[AUDIT] A10 — AST-walk every attacks module; assert no Import/ImportFrom
    resolves to a forbidden top-level package (swarm/moe/fusion/policy). GREEN
    today; standing regression gate (the import-isolation invariant)."""
    violations: list[str] = []
    for path in _attacks_module_paths():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for module in _collect_imported_modules(tree):
            if _module_head_is_forbidden(module):
                violations.append(f"{path.name}: imports {module}")

    assert not violations, (
        "attacks layer must not import detection/orchestration packages: "
        + "; ".join(violations)
    )


def test_ax_isolation_at_least_one_attacks_module_scanned() -> None:
    """A10 — guard against a no-op pass if the glob ever silently matches
    nothing; ``reid.py`` must be among the scanned files."""
    paths = _attacks_module_paths()
    assert paths, "expected at least one *.py under eval_framework/attacks/"
    names = {p.name for p in paths}
    assert "__init__.py" in names
    assert "reid.py" in names
