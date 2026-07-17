"""Tests for the SP1 per-change detection measurement harness.

The harness scores core-only detection on the DATA corpus with the same
canonical-label semantics the eval uses (truth AND detections normalized via
``_normalize_entity_type``; FP-universe rule; ``_BENCHMARK_IGNORE`` truth
excluded into ``unmatchable_truth_spans``). Anchors are EXACT (standing
program lesson: never bound-anchor representative metrics).

ACCEPTED divergences pinned here (do NOT change): any-character-overlap
matching (canonical: IoU >= 0.5) and non-exclusive matching (canonical:
greedy 1:1).
"""

from __future__ import annotations

from scripts.sp1_detection_delta import ScoreStats, build_payload, score_records


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
    # Detector finds the email (TP) and an EMPLOYEE_ID at the wrong location
    # (FP — the type IS in the truth universe); the real EMPLOYEE_ID is missed
    # (FN).
    found = [("r1", "EMAIL_ADDRESS", 5, 14), ("r1", "EMPLOYEE_ID", 0, 3)]
    stats = score_records([(rec, found)])
    assert stats.per_entity["EMAIL_ADDRESS"] == (1, 0, 0)
    assert stats.per_entity["EMPLOYEE_ID"] == (0, 1, 1)
    assert stats.micro_precision == 0.5
    assert stats.micro_recall == 0.5
    # F2 = 5PR/(4P+R) = 5*0.25/(2.0+0.5) = 0.5
    assert stats.micro_f2 == 0.5
    assert stats.unmatchable_truth_spans == 0


def test_fp_universe_rule_drops_truthless_type() -> None:
    # Canonical drops predictions whose canonical type has no truth anywhere
    # in the run before FP counting (competitor_compare.py:1802-1809). A
    # PHONE_NUMBER detection with no PHONE_NUMBER truth in the sample is
    # dropped — NOT a false positive.
    rec = _Rec(
        "Mail bob@x.com ref 12345",
        [{"entity_type": "EMAIL_ADDRESS", "start": 5, "end": 14}],
    )
    found = [("r1", "EMAIL_ADDRESS", 5, 14), ("r1", "PHONE_NUMBER", 19, 24)]
    stats = score_records([(rec, found)])
    assert "PHONE_NUMBER" not in stats.per_entity
    assert stats.per_entity["EMAIL_ADDRESS"] == (1, 0, 0)
    assert stats.micro_precision == 1.0
    assert stats.micro_recall == 1.0
    assert stats.micro_f2 == 1.0


def test_canonical_truth_normalization_iban_is_bank_account() -> None:
    # THE critical fix: corpus truth type IBAN normalizes canonically to
    # BANK_ACCOUNT, so a BANK_ACCOUNT detection on the same span is ONE true
    # positive — not the 1 FP + 1 FN double-penalty of raw-label comparison.
    rec = _Rec(
        "Pay to DE89370400440532013000 today",
        [{"entity_type": "IBAN", "start": 7, "end": 29}],
    )
    found = [("r1", "BANK_ACCOUNT", 7, 29)]
    stats = score_records([(rec, found)])
    assert stats.per_entity["BANK_ACCOUNT"] == (1, 0, 0)
    assert "IBAN" not in stats.per_entity
    assert stats.micro_precision == 1.0
    assert stats.micro_recall == 1.0


def test_overlap_counts_as_match_and_literal_ignore_is_unmatchable() -> None:
    rec = _Rec(
        "DOB: 1981-02-20 at 10:00",
        [
            {"entity_type": "DATE_OF_BIRTH", "start": 5, "end": 15},
            {"entity_type": "_BENCHMARK_IGNORE", "start": 19, "end": 24},
        ],
    )
    # Partial overlap (5..12 vs truth 5..15) still a TP (ACCEPTED divergence:
    # any-character overlap, not IoU >= 0.5); the literal ignore sentinel goes
    # to unmatchable_truth_spans, never per_entity.
    found = [("r1", "DATE_OF_BIRTH", 5, 12)]
    stats = score_records([(rec, found)])
    assert stats.per_entity["DATE_OF_BIRTH"] == (1, 0, 0)
    assert "_BENCHMARK_IGNORE" not in stats.per_entity
    assert stats.unmatchable_truth_spans == 1


def test_raw_ignore_mapped_truth_counts_as_unmatchable() -> None:
    # API_KEY is a RAW corpus truth type that maps to _BENCHMARK_IGNORE in
    # the canonical map (competitor_compare.py). It must land in
    # unmatchable_truth_spans — excluded from per_entity AND from the micro
    # recall denominator.
    rec = _Rec(
        "key sk-123456789A mail bob@x.com",
        [
            {"entity_type": "API_KEY", "start": 4, "end": 17},
            {"entity_type": "EMAIL_ADDRESS", "start": 23, "end": 32},
        ],
    )
    found = [("r1", "EMAIL_ADDRESS", 23, 32)]
    stats = score_records([(rec, found)])
    assert stats.unmatchable_truth_spans == 1
    assert "API_KEY" not in stats.per_entity
    assert "_BENCHMARK_IGNORE" not in stats.per_entity
    assert stats.per_entity["EMAIL_ADDRESS"] == (1, 0, 0)
    assert stats.micro_recall == 1.0


def test_adjacent_spans_do_not_overlap() -> None:
    # Touching spans share no character: detection [0,5) vs truth [5,15) is
    # 1 FP + 1 FN, never a match.
    rec = _Rec(
        "0123456789ABCDE",
        [{"entity_type": "X", "start": 5, "end": 15}],
    )
    found = [("r1", "X", 0, 5)]
    stats = score_records([(rec, found)])
    assert stats.per_entity["X"] == (0, 1, 1)


def test_payload_contract_round_trip() -> None:
    # build_payload is the exact dict main() writes to disk — pin the
    # contract: key set, micro keys, and per_entity values as 3-int lists.
    stats = ScoreStats(
        per_entity={"EMAIL_ADDRESS": (3, 1, 2), "PERSON_NAME": (5, 0, 4)},
        micro_precision=0.8,
        micro_recall=0.571429,
        micro_f2=0.606061,
        latency_p50_ms=0.2382,
        n_records=7,
        unmatchable_truth_spans=4,
    )
    payload = build_payload(stats, seed=8314)
    assert set(payload) == {
        "n",
        "seed",
        "micro",
        "latency_p50_ms",
        "per_entity",
        "unmatchable_truth_spans",
    }
    assert payload["n"] == 7
    assert payload["seed"] == 8314
    assert payload["unmatchable_truth_spans"] == 4
    assert set(payload["micro"]) == {"precision", "recall", "f2"}
    assert sorted(payload["per_entity"]) == ["EMAIL_ADDRESS", "PERSON_NAME"]
    for value in payload["per_entity"].values():
        assert isinstance(value, list)
        assert len(value) == 3
        assert all(isinstance(i, int) for i in value)
    assert payload["per_entity"]["EMAIL_ADDRESS"] == [3, 1, 2]
