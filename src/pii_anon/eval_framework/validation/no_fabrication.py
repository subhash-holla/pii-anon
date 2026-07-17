"""No-fabrication validators (public surface).

Each function mirrors, byte-for-byte in behaviour, the private validator of the
same stem in ``competitive_supremacy.py``. See that module's docstrings for the
adversarial history behind each guard (the S7-02 close rounds). The contract
test pins equivalence.
"""

from __future__ import annotations

import math
from typing import Any, TypeGuard


def is_finite_number(value: object) -> TypeGuard[int | float]:
    """Is ``value`` a real, finite number (an ``int``/``float`` that is not a bool)?

    Excludes ``bool`` (``isinstance(True, int)`` is ``True`` in Python, but a flag
    is not a measurement). Rejects ``NaN`` / ``±inf`` and ints wider than a C double
    (``math.isfinite`` raises ``OverflowError`` on ``10**400``) — fail CLOSED rather
    than crash on adversarial input.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def finite_unit_score(value: object) -> float | None:
    """Return ``value`` as a float IFF it is a real, finite, unit-interval score.

    Returns ``None`` for a bool, a non-finite value, or a value outside ``[0, 1]``,
    so a caller can treat the field as ABSENT rather than trust a fabricated value.
    """
    if not is_finite_number(value):
        return None
    v = float(value)
    if v < 0.0 or v > 1.0:
        return None
    return v


def is_nonblank_str(value: object) -> bool:
    """A provenance / identifier stamp is PRESENT only as a NON-BLANK STRING.

    A truthy-but-invalid value (whitespace-only string, int, bool, non-empty
    list/dict) is missing-equivalent.
    """
    return isinstance(value, str) and bool(value.strip())


def detected_entity_names(recall_map: dict[str, Any]) -> set[str]:
    """Entity types a system genuinely DETECTS — those whose per-entity score is a
    VALID score (finite, in ``[0, 1]``, not a bool) STRICTLY > 0.

    Routes each nested value through :func:`finite_unit_score`, closing the
    no-fabrication invariant on a nested field (a ``True`` / ``+inf`` / ``1e40`` /
    ``0.0`` / negative value never counts as a detected entity).
    """
    out: set[str] = set()
    for entity, raw in recall_map.items():
        score = finite_unit_score(raw)
        if score is not None and score > 0.0:
            out.add(entity)
    return out


def safe_repr(value: object, limit: int = 120) -> str:
    """A repr that can NEVER crash a detail string.

    Guards the CPython int→str digit limit (``repr(10**5000)`` raises) and any
    object whose ``__repr__`` itself raises. Both fail CLOSED into a typed
    placeholder; long honest reprs are truncated.
    """
    try:
        out = repr(value)
    except Exception:
        return f"<unrepresentable {type(value).__name__}>"
    if len(out) > limit:
        return out[:limit] + "…(truncated)"
    return out


def safe_names(items: Any) -> str:
    """A crash-free, deterministically-ordered list rendering for detail strings.

    Sorts and renders through :func:`safe_repr` so a huge-int member cannot crash
    the sort key or the element repr, while preserving the ``['a', 'b']`` shape.
    """
    return "[" + ", ".join(safe_repr(x) for x in sorted(items, key=safe_repr)) + "]"
