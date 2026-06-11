"""Tests for the SP1 per-change detection measurement harness.

The harness scores core-only detection on the DATA corpus with the same
overlap-match + canonical-label semantics the eval uses. Anchors are EXACT
(standing program lesson: never bound-anchor representative metrics).
"""

from __future__ import annotations

from scripts.sp1_detection_delta import score_records


class _Rec:
    """Minimal BenchmarkRecord stand-in (text + labels + record_id)."""

    def __init__(self, text: str, labels: list[dict]) -> None:
        self.record_id = "r1"
        self.text = text
        self.labels = labels
        self.language = "en"


def test_exact_counts_tp_fp_fn() -> None:
    rec = _Rec(
        "Mail bob@x.com ref 12345",
        [
            {"entity_type": "EMAIL_ADDRESS", "start": 5, "end": 14},
            {"entity_type": "EMPLOYEE_ID", "start": 19, "end": 24},
        ],
    )
    # Detector finds the email (TP) and a bogus PHONE (FP); misses EMPLOYEE_ID (FN).
    found = [("r1", "EMAIL_ADDRESS", 5, 14), ("r1", "PHONE_NUMBER", 19, 24)]
    stats = score_records([(rec, found)])
    assert stats.per_entity["EMAIL_ADDRESS"] == (1, 0, 0)
    assert stats.per_entity["PHONE_NUMBER"] == (0, 1, 0)
    assert stats.per_entity["EMPLOYEE_ID"] == (0, 0, 1)
    assert stats.micro_precision == 0.5
    assert stats.micro_recall == 0.5
    # F2 = 5PR/(4P+R) = 5*0.25/(2.0+0.5) = 0.5
    assert stats.micro_f2 == 0.5


def test_overlap_counts_as_match_and_benchmark_ignore_skipped() -> None:
    rec = _Rec(
        "DOB: 1981-02-20 at 10:00",
        [
            {"entity_type": "DATE_OF_BIRTH", "start": 5, "end": 15},
            {"entity_type": "_BENCHMARK_IGNORE", "start": 19, "end": 24},
        ],
    )
    # Partial overlap (5..12 vs truth 5..15) still a TP; nothing scored for ignored.
    found = [("r1", "DATE_OF_BIRTH", 5, 12)]
    stats = score_records([(rec, found)])
    assert stats.per_entity["DATE_OF_BIRTH"] == (1, 0, 0)
    assert "_BENCHMARK_IGNORE" not in stats.per_entity
