"""S7-05 — docs discoverability [DOCS MUST] (D6 SME Docs MAJORs + FR-010 surface).

Implements the dev-log W2 [DOCS MUST] commitment: the program's API families
must be DISCOVERABLE from `docs/` — headline symbols in the api-reference or
index, a dedicated FR-010 anonymization-vs-pseudonymization document (with
the no-merge invariant and the vanilla-vs-swarm positioning), the
evaluate-your-pipeline guide covering the supremacy/canonical-run CLI, a
complete index with zero broken intra-docs links, and a top-level CLI help
that points at the surfaces. This module IS the teeth: future surfaces
cannot silently ship undocumented.

NEVER touches (user-WIP): README.md (repo root), docs/pii-rate-elo-value.md,
docs/benchmark-summary.md — both excluded via the explicit allowlist below.

Anchors A1–A7 per the story contract
(dev-assist-artifacts/04-development/02-stories/sprint-7/
S7-05-docs-discoverability.md).
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DOCS = _REPO / "docs"

# Generated or user-WIP docs the index is NOT required to link (and this
# story must never modify).
_INDEX_ALLOWLIST = {"README.md", "benchmark-summary.md", "pii-rate-elo-value.md"}

# User-WIP/generated docs excluded from the link-resolution sweep too: this
# story is forbidden from editing them, so their links cannot gate it.
# (docs/README.md itself IS link-checked.)
_LINK_CHECK_EXCLUDE = {"benchmark-summary.md", "pii-rate-elo-value.md"}

_LINK_RE = re.compile(r"\]\(([^)#]+?)(?:#[^)]*)?\)")


def _index_text() -> str:
    return (_DOCS / "README.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A1 — index completeness
# ---------------------------------------------------------------------------

def test_a1_index_links_every_non_generated_doc() -> None:
    """[UNIT-TEST] A1: every non-allowlisted docs/*.md appears as a link in
    the docs index — including the new FR-010 anon-vs-pseudo document."""
    index = _index_text()
    doc_files = sorted(
        p.name for p in _DOCS.glob("*.md") if p.name not in _INDEX_ALLOWLIST
    )
    missing = [name for name in doc_files if name not in index]
    assert missing == [], f"docs index does not link: {missing}"
    # The FR-010 headline doc must exist AND be indexed (explicit pin).
    assert "anonymization-vs-pseudonymization.md" in index


# ---------------------------------------------------------------------------
# A2 — intra-docs links resolve
# ---------------------------------------------------------------------------

def test_a2_all_relative_doc_links_resolve() -> None:
    """[UNIT-TEST] A2: every relative markdown link in every doc resolves to
    an existing file (zero broken links)."""
    broken: list[str] = []
    for doc in _DOCS.glob("*.md"):
        if doc.name in _LINK_CHECK_EXCLUDE:
            continue
        text = doc.read_text(encoding="utf-8")
        for target in _LINK_RE.findall(text):
            target = target.strip()
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (doc.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{doc.name} -> {target}")
    assert broken == [], f"broken intra-docs links: {broken}"


# ---------------------------------------------------------------------------
# A3 — the FR-010 anonymization-vs-pseudonymization document
# ---------------------------------------------------------------------------

def test_a3_anon_vs_pseudo_doc_names_both_families_and_no_merge() -> None:
    """[UNIT-TEST] A3: the FR-010 doc exists, names BOTH distinct scorer
    families, states the no-merge invariant, and carries the
    vanilla-vs-swarm positioning section."""
    doc = _DOCS / "anonymization-vs-pseudonymization.md"
    assert doc.exists(), "docs/anonymization-vs-pseudonymization.md missing"
    text = doc.read_text(encoding="utf-8")
    assert "AnonymizationScorer" in text
    assert "PseudonymizationIntegrityScorer" in text
    assert "no-merge" in text.lower()
    assert "vanilla" in text.lower() and "swarm" in text.lower()


# ---------------------------------------------------------------------------
# A4 — recall-floor doc references the LIVE mechanism (regression pin)
# ---------------------------------------------------------------------------

def test_a4_recall_floor_doc_references_live_mechanism() -> None:
    """[UNIT-TEST] A4: docs/recall-floor.md references the shipped
    SharedLayerProjector mechanism (the D6 SME divergence stays closed)."""
    text = (_DOCS / "recall-floor.md").read_text(encoding="utf-8")
    assert "SharedLayerProjector" in text
    assert "routing.shared_layer" in text or "routing/shared_layer" in text


# ---------------------------------------------------------------------------
# A5 — evaluate-your-pipeline covers the SDO CLI + the SDK group
# ---------------------------------------------------------------------------

def test_a5_evaluate_your_pipeline_covers_sdo_cli_and_sdk_group() -> None:
    """[UNIT-TEST] A5: the guide names the pii_anon.byo_pipelines group
    (S6-04) AND the supremacy + canonical-run CLI commands."""
    text = (_DOCS / "evaluate-your-pipeline.md").read_text(encoding="utf-8")
    assert "pii_anon.byo_pipelines" in text
    assert "supremacy" in text
    assert "canonical-run" in text


# ---------------------------------------------------------------------------
# A6 — headline program symbols are discoverable
# ---------------------------------------------------------------------------

_HEADLINE_SYMBOLS = [
    # S4-01 distinct deid families (FR-010)
    "AnonymizationScorer",
    "PseudonymizationIntegrityScorer",
    # S6-01 query-aware masking gate (FR-023)
    "QueryAwareMaskingGate",
    # S6-03 encrypted token store
    "EncryptedSQLiteTokenStore",
    # S6-04 BYO SDK (FR-001/002)
    "BYOPipelineRegistry",
    "evaluate_incumbent",
    # S7-01 native readers (FR-031)
    "NativeReaderRegistry",
    "reader_capabilities",
    # S7-03 fairness gate (FR-039/NFR-025)
    "evaluate_language_fairness",
]


def test_a6_headline_symbols_appear_in_api_reference_or_index() -> None:
    """[UNIT-TEST] A6: every curated headline symbol appears in
    docs/api-reference.md or the docs index — the discoverability teeth."""
    haystack = (
        (_DOCS / "api-reference.md").read_text(encoding="utf-8")
        + _index_text()
    )
    missing = [s for s in _HEADLINE_SYMBOLS if s not in haystack]
    assert missing == [], f"undocumented headline symbols: {missing}"


# ---------------------------------------------------------------------------
# A7 — CLI help points at the surfaces
# ---------------------------------------------------------------------------

def test_a7_cli_help_mentions_the_program_surfaces() -> None:
    """[UNIT-TEST] A7: `pii-anon --help` (module invocation) mentions the
    BYO scoring surface and the native readers (additive epilog only)."""
    result = subprocess.run(
        [sys.executable, "-m", "pii_anon.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=str(_REPO),
        env={"PYTHONPATH": str(_REPO / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-500:]
    help_text = result.stdout
    assert "BYO" in help_text
    assert "readers" in help_text
