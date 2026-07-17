"""Tests for S4-CS-01: the Tier-R / Tier-C competitor registry (honesty-gated).

Implements / pins:
    FR-007 / FR-008 — canonical-run gate + provenance (the registry feeds the
              SDO ``CompetitiveSupremacyGate`` the honesty boundary it reports).
    RISK-6 (story §2a) — the Tier-R / Tier-C registry lives ENTIRELY in
              ``competitor_tiers.py``; ``evaluation/competitor_compare.py``'s
              ``_COMPETITOR_META.keys()`` drives the *run* set and must stay
              byte-identical, so the new tiers are metadata ONLY here.

Design trace: Program AMENDMENT "SOTA-Dominance Objective (SDO)" / DC-11
CanonicalRunGate. The registry is pure stdlib (frozen dataclasses) — no numpy,
no pydantic — so it imports in any environment.

Honesty invariants encoded here
-------------------------------
* Tier-R = {presidio, scrubadub, gliner, gliner2}; Tier-C = {openai-privacy-filter,
  azure-ai-language, aws-comprehend}.
* Every Tier-C competitor is UNRUN by default (real APIs/keys ⇒ Pass-2; never
  agent-simulated as real).
* gliner2 is UNRUN until its adapter actually runs (it is NOT in today's
  benchmark ``available_competitors``).
* ``waive(name, reason)`` requires a non-empty reason — a blank reason raises;
  a real reason flips the entry to WAIVED.
* ``apply_run_status`` is PURE — it returns a new registry mapping and never
  mutates the module-level registry.
"""

from __future__ import annotations

import pytest

from pii_anon.eval_framework.evaluation.competitor_tiers import (
    TIER_C_NAMES,
    TIER_R_NAMES,
    CompetitorTier,
    RunStatus,
    TierEntry,
    apply_run_status,
    default_registry,
    run_status_from_benchmark,
    unrun_tier_c,
    unrun_tier_r,
    waive,
)

# ---------------------------------------------------------------------------
# Tier membership (the registry contract)
# ---------------------------------------------------------------------------


def test_tier_r_membership_is_exactly_the_four_runnable_systems() -> None:
    """[CONTRACT-TEST] Tier-R = {presidio, scrubadub, gliner, gliner2}."""
    assert TIER_R_NAMES == frozenset(
        {"presidio", "scrubadub", "gliner", "gliner2"}
    )


def test_tier_c_membership_is_exactly_the_three_cited_apis() -> None:
    """[CONTRACT-TEST] Tier-C = {openai-privacy-filter, azure-ai-language,
    aws-comprehend} — the cited, must-run-before-claim APIs."""
    assert TIER_C_NAMES == frozenset(
        {"openai-privacy-filter", "azure-ai-language", "aws-comprehend"}
    )


def test_registry_covers_every_tier_r_and_tier_c_name() -> None:
    """[CONTRACT-TEST] default_registry has one entry per declared name and
    nothing extra (no silent membership drift)."""
    reg = default_registry()
    assert set(reg.keys()) == TIER_R_NAMES | TIER_C_NAMES


def test_every_entry_carries_a_citation_and_tier() -> None:
    """[AUDIT] Each entry names its tier and carries a non-empty citation so the
    gate output can cite the competitor it is being measured against."""
    reg = default_registry()
    for name, entry in reg.items():
        assert isinstance(entry, TierEntry)
        assert entry.name == name
        assert entry.tier in (CompetitorTier.R, CompetitorTier.C)
        assert entry.citation, f"{name} must carry a citation"


# ---------------------------------------------------------------------------
# Honesty boundary: Tier-C UNRUN by default; gliner2 UNRUN until adapter runs
# ---------------------------------------------------------------------------


def test_all_tier_c_are_unrun_by_default_honesty_boundary() -> None:
    """[AUDIT] Tier-C real-API adapters are Pass-2 — UNRUN by default; the gate
    never fabricates a Tier-C result."""
    reg = default_registry()
    for name in TIER_C_NAMES:
        assert reg[name].run_status is RunStatus.UNRUN


def test_gliner2_is_unrun_until_its_adapter_runs() -> None:
    """[AUDIT] gliner2 is a NEW Tier-R adapter not yet in the benchmark run —
    UNRUN by default (it is not in today's available_competitors)."""
    reg = default_registry()
    assert reg["gliner2"].tier is CompetitorTier.R
    assert reg["gliner2"].run_status is RunStatus.UNRUN


def test_unrun_tier_c_returns_the_blocking_set() -> None:
    """[UNIT-TEST] unrun_tier_c surfaces exactly the Tier-C names still UNRUN —
    the set that blocks CLAIM_GRADE until run or waived."""
    reg = default_registry()
    assert unrun_tier_c(reg) == TIER_C_NAMES


def test_unrun_tier_c_excludes_waived_and_run_entries() -> None:
    """[UNIT-TEST] A waived or run Tier-C name drops out of the blocking set."""
    reg = default_registry()
    reg2 = waive(reg, "openai-privacy-filter", "no API budget this cycle")
    blocking = unrun_tier_c(reg2)
    assert "openai-privacy-filter" not in blocking
    assert blocking == TIER_C_NAMES - {"openai-privacy-filter"}


def test_unrun_tier_r_is_all_tier_r_in_bare_default_registry() -> None:
    """[UNIT-TEST] §5: in the BARE default registry every Tier-R entry is UNRUN
    (run-status is only derived once a benchmark is applied), so unrun_tier_r ==
    the full Tier-R set. The interesting case (only gliner2 left) is the one
    after run-status is applied — see the next test."""
    reg = default_registry()
    assert unrun_tier_r(reg) == TIER_R_NAMES


def test_unrun_tier_r_excludes_run_and_waived_entries() -> None:
    """[UNIT-TEST] A RUN or WAIVED Tier-R name drops out of the Tier-R blocking
    set (symmetric to unrun_tier_c)."""
    reg = apply_run_status(
        default_registry(),
        {"presidio": RunStatus.RUN, "scrubadub": RunStatus.RUN, "gliner": RunStatus.RUN},
    )
    assert unrun_tier_r(reg) == frozenset({"gliner2"})  # only gliner2 left unrun
    reg2 = waive(reg, "gliner2", "adapter not yet wired; cited Tier-R")
    assert unrun_tier_r(reg2) == frozenset()  # waived ⇒ no longer blocking
    # Tier-C entries never appear in the Tier-R set.
    assert unrun_tier_r(reg2).isdisjoint(TIER_C_NAMES)


# ---------------------------------------------------------------------------
# waive(): reason is mandatory
# ---------------------------------------------------------------------------


def test_waive_with_empty_reason_raises() -> None:
    """[CONTRACT-TEST] waive(name, "") raises — a waiver MUST carry a documented
    reason recorded in the gate (no silent claim-grade unblocking)."""
    reg = default_registry()
    with pytest.raises(ValueError, match="reason"):
        waive(reg, "azure-ai-language", "")


def test_waive_with_blank_whitespace_reason_raises() -> None:
    """[CONTRACT-TEST] A whitespace-only reason is still empty — raises."""
    reg = default_registry()
    with pytest.raises(ValueError, match="reason"):
        waive(reg, "azure-ai-language", "   ")


def test_waive_with_reason_sets_waived_and_records_reason() -> None:
    """[UNIT-TEST] waive(name, reason) flips the entry to WAIVED and records the
    reason for the audit trail."""
    reg = default_registry()
    reg2 = waive(reg, "aws-comprehend", "vendor contract pending")
    assert reg2["aws-comprehend"].run_status is RunStatus.WAIVED
    assert reg2["aws-comprehend"].waiver_reason == "vendor contract pending"


def test_waive_is_pure_does_not_mutate_input_registry() -> None:
    """[PROPERTY-TEST] waive returns a NEW mapping; the input registry (and the
    module-level default) is never mutated."""
    reg = default_registry()
    before = reg["aws-comprehend"].run_status
    _ = waive(reg, "aws-comprehend", "vendor contract pending")
    assert reg["aws-comprehend"].run_status is before is RunStatus.UNRUN


def test_waive_unknown_name_raises() -> None:
    """[CONTRACT-TEST] Waiving a name not in the registry fails loud (a typo in a
    waiver must never silently no-op)."""
    reg = default_registry()
    with pytest.raises(KeyError):
        waive(reg, "not-a-real-competitor", "because")


# ---------------------------------------------------------------------------
# apply_run_status(): PURE derivation from the benchmark JSON
# ---------------------------------------------------------------------------


def test_apply_run_status_is_pure_no_global_mutation() -> None:
    """[PROPERTY-TEST] apply_run_status returns a fresh mapping and mutates
    neither its argument nor the module-level default registry."""
    reg = default_registry()
    statuses = {"gliner": RunStatus.RUN}
    out = apply_run_status(reg, statuses)
    # Input untouched.
    assert reg["gliner"].run_status is RunStatus.UNRUN
    # A second default_registry() call is pristine.
    assert default_registry()["gliner"].run_status is RunStatus.UNRUN
    # Output reflects the override.
    assert out["gliner"].run_status is RunStatus.RUN


def test_apply_run_status_only_touches_named_entries() -> None:
    """[UNIT-TEST] Entries not named in the status map keep their prior status."""
    reg = default_registry()
    out = apply_run_status(reg, {"gliner": RunStatus.RUN})
    assert out["presidio"].run_status is RunStatus.UNRUN
    assert out["gliner2"].run_status is RunStatus.UNRUN


def test_apply_run_status_unknown_name_raises() -> None:
    """[CONTRACT-TEST] A status for an unknown competitor fails loud."""
    reg = default_registry()
    with pytest.raises(KeyError):
        apply_run_status(reg, {"mystery-system": RunStatus.RUN})


# ---------------------------------------------------------------------------
# run_status_from_benchmark(): derive RUN/UNRUN from the benchmark JSON
# ---------------------------------------------------------------------------


def test_run_status_from_benchmark_marks_available_competitors_run() -> None:
    """[UNIT-TEST] A Tier-R competitor present in available_competitors (and
    available=True in its system record) is RUN; one that is absent is UNRUN.

    Derived from the benchmark JSON's own ``available_competitors`` + per-system
    ``available`` flag — NEVER fabricated (RISK-6)."""
    bench = {
        "available_competitors": ["presidio", "scrubadub", "gliner"],
        "systems": [
            {"system": "gliner", "available": True},
            {"system": "presidio", "available": True},
            {"system": "scrubadub", "available": True},
        ],
    }
    statuses = run_status_from_benchmark(bench)
    assert statuses["gliner"] is RunStatus.RUN
    assert statuses["presidio"] is RunStatus.RUN
    assert statuses["scrubadub"] is RunStatus.RUN
    # gliner2 + Tier-C never appear in a benchmark that did not run them.
    assert "gliner2" not in statuses
    assert "openai-privacy-filter" not in statuses


def test_run_status_from_benchmark_unavailable_flag_is_not_run() -> None:
    """[UNIT-TEST] A competitor listed but flagged available=False is NOT RUN."""
    bench = {
        "available_competitors": ["presidio"],
        "systems": [
            {"system": "presidio", "available": False},
        ],
    }
    statuses = run_status_from_benchmark(bench)
    assert statuses.get("presidio", RunStatus.UNRUN) is not RunStatus.RUN


def test_run_status_from_benchmark_applied_to_registry_is_consistent() -> None:
    """[INTEGRATION-TEST] Composing run_status_from_benchmark → apply_run_status
    marks the three runnable Tier-R systems RUN and leaves gliner2 + Tier-C
    UNRUN (the honesty boundary the gate then reports)."""
    bench = {
        "available_competitors": ["presidio", "scrubadub", "gliner"],
        "systems": [
            {"system": "gliner", "available": True},
            {"system": "presidio", "available": True},
            {"system": "scrubadub", "available": True},
        ],
    }
    reg = apply_run_status(default_registry(), run_status_from_benchmark(bench))
    assert reg["gliner"].run_status is RunStatus.RUN
    assert reg["gliner2"].run_status is RunStatus.UNRUN
    assert unrun_tier_c(reg) == TIER_C_NAMES
