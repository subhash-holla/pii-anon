"""sp7 STEP 1 — FR-036 stream/batch/offline scrub-decision parity.

The Phase-0 streaming audit found GAP-1, a silent PII LEAK: a multi-token
entity straddling a segment boundary is detected by NEITHER adjacent segment
when the overlap is too small (batch detects "415 555 1234"; segmented at
overlap 0 -> nothing). The default overlap (128) hides this for realistic
entities, but FR-036 requires a GUARANTEE, not a default-dependent accident.

Fix under test: the Segmenter enforces a minimum-safe overlap floor so any
entity up to that many tokens is wholly contained in at least one window
regardless of the caller's overlap request. Over-detection + dedup is the
leak-SAFE direction; the guarantee is independent of config.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.segmentation.segmenter import Segmenter

_ENGINE = RegexEngineAdapter(enabled=True)


def _batch_spans(text: str) -> set[tuple[str, int, int]]:
    return {
        (str(f.entity_type), f.span_start, f.span_end)
        for f in _ENGINE.detect({"text": text}, {"language": "en"})
    }


def _segmented_spans(text: str, *, max_tokens: int, overlap_tokens: int) -> set[tuple[str, int, int]]:
    """Detect per segment, re-base offsets to the full text, dedupe — the
    reconciler's global-coordinate contract."""
    out: set[tuple[str, int, int]] = set()
    for seg in Segmenter().segment(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
        for f in _ENGINE.detect({"text": seg.text}, {"language": "en"}):
            out.add(
                (str(f.entity_type), seg.start_char + f.span_start, seg.start_char + f.span_end)
            )
    return out


# A corpus of multi-token entities that straddle boundaries at small windows.
_TEXTS = [
    "Please call the office at 415 555 1234 tomorrow morning to confirm today.",
    "The patient John Michael Smith arrived at the clinic on the third of May.",
    "Wire the funds to IBAN GB29 NWBK 6016 1331 9268 19 before the end of day.",
    "Her address is 742 Evergreen Terrace Springfield and the meeting is at two.",
]


# Window sizes large enough to hold an entity + its detection context (the
# realistic regime; a window smaller than entity+context is GAP-2 context
# starvation, a separate concern from the GAP-1 boundary-split leak). The
# overlap floor makes the step small so the straddling entity is always
# re-covered WITH context.
@pytest.mark.parametrize("text", _TEXTS)
@pytest.mark.parametrize("overlap_tokens", [0, 1, 2, 5])
def test_no_boundary_leak_at_small_overlap(text: str, overlap_tokens: int) -> None:
    """GAP-1: every batch-detected span must survive segmentation at ANY
    overlap — a straddling entity that vanishes is a production PII leak."""
    batch = _batch_spans(text)
    seg = _segmented_spans(text, max_tokens=10, overlap_tokens=overlap_tokens)
    missing = batch - seg
    assert not missing, (
        f"boundary LEAK at overlap={overlap_tokens}: batch spans absent from "
        f"segmented output: {[(t, text[s:e]) for t, s, e in missing]}"
    )


@pytest.mark.parametrize("text", _TEXTS)
@pytest.mark.parametrize("max_tokens,overlap_tokens", [(10, 0), (12, 2), (16, 4)])
def test_segmented_superset_of_batch(text: str, max_tokens: int, overlap_tokens: int) -> None:
    """Segmented detection must be a SUPERSET of batch (parity in the
    leak-safe direction: it may over-detect in overlap regions, never
    under-detect)."""
    batch = _batch_spans(text)
    seg = _segmented_spans(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
    assert batch <= seg, f"under-detection: {batch - seg}"


def test_segmenter_never_produces_zero_step() -> None:
    """A config with overlap >= max_tokens must not hang or skip text — the
    segmenter must still advance and cover every token."""
    text = " ".join(f"tok{i}" for i in range(50))
    segments = Segmenter().segment(text, max_tokens=3, overlap_tokens=10)
    covered = set()
    for s in segments:
        covered.update(range(s.start_token, s.end_token))
    assert covered == set(range(50)), "segmentation left tokens uncovered"
