"""Agentic PII interception surface (S6-02 / DC-13).

A standalone library package — deliberately off the cycle-prone ``routing/``
path and never wired into the protected ``orchestrator.py`` — that intercepts
PII on all four agent channels (prompt / memory / tool-I/O / trace) under
least-privilege (AX-006) and guarantees no raw PII is persisted after masking
(FR-026).

Re-exports are **lazy** (PEP 562 ``__getattr__``) so importing this package
never forces the tokenization-store import graph; the public names resolve on
first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pii_anon.agentic.interception import (
        AgentChannel,
        ChannelMasker,
        ChannelResult,
        FourChannelGuard,
        InterceptionLedger,
        InterceptionRecord,
        NoRawPIIPersistError,
    )
    from pii_anon.agentic.leakage_sankey import (
        InjectionResistanceReport,
        LeakageSankey,
        SankeyEdge,
        build_leakage_sankey,
        score_injection_resistance,
    )

__all__ = [
    # interception (S6-02)
    "AgentChannel",
    "InterceptionRecord",
    "ChannelResult",
    "ChannelMasker",
    "FourChannelGuard",
    "InterceptionLedger",
    "NoRawPIIPersistError",
    # leakage_sankey (S6-05) — additive
    "SankeyEdge",
    "LeakageSankey",
    "build_leakage_sankey",
    "InjectionResistanceReport",
    "score_injection_resistance",
]

# Names resolved lazily from ``interception`` (S6-02).
_INTERCEPTION_NAMES = frozenset(
    {
        "AgentChannel",
        "InterceptionRecord",
        "ChannelResult",
        "ChannelMasker",
        "FourChannelGuard",
        "InterceptionLedger",
        "NoRawPIIPersistError",
    }
)
# Names resolved lazily from ``leakage_sankey`` (S6-05) — additive.
_LEAKAGE_SANKEY_NAMES = frozenset(
    {
        "SankeyEdge",
        "LeakageSankey",
        "build_leakage_sankey",
        "InjectionResistanceReport",
        "score_injection_resistance",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily resolve the public agentic names on first access (PEP 562)."""
    if name in _INTERCEPTION_NAMES:
        from pii_anon.agentic import interception

        return getattr(interception, name)
    if name in _LEAKAGE_SANKEY_NAMES:
        from pii_anon.agentic import leakage_sankey

        return getattr(leakage_sankey, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
