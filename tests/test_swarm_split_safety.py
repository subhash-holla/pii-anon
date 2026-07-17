"""Split-safety for swarm training (roadmap lever #2, 2026-07-17).

The deployed 2026-04 DS artifact was trained with the FULL corpus loaded
unsplit — 100% of the current test split sat inside its training pool, which
makes any trained artifact indefensible. These tests pin the two repairs:

1. ``load_pii_anon_data`` requests the TRAIN split by default (leak-safe
   default; eval paths never use this loader).
2. ``assert_train_test_disjoint`` fails loud on any train∩test id overlap, so
   a stale/ignoring loader can never silently reintroduce the leak.
"""
from __future__ import annotations

from unittest import mock

import pytest

from pii_anon.swarm_datasets import (
    TrainingRecord,
    assert_train_test_disjoint,
    load_pii_anon_data,
)


def _rec(record_id: str) -> TrainingRecord:
    return TrainingRecord(record_id=record_id, text="x", labels=[], language="en", source_dataset="t")


def test_disjoint_guard_passes_on_disjoint_sets() -> None:
    train = [_rec("a"), _rec("b")]
    assert_train_test_disjoint(train, {"c", "d"})  # must not raise


def test_disjoint_guard_fails_loud_on_overlap() -> None:
    train = [_rec("a"), _rec("b"), _rec("c")]
    with pytest.raises(ValueError, match=r"train.test|leak"):
        assert_train_test_disjoint(train, {"b", "z"})


def test_disjoint_guard_reports_overlap_size() -> None:
    train = [_rec(f"id-{i}") for i in range(10)]
    with pytest.raises(ValueError, match=r"3"):
        assert_train_test_disjoint(train, {"id-1", "id-2", "id-3"})


def test_load_pii_anon_data_requests_train_split_by_default() -> None:
    fake_pkg = mock.MagicMock()
    fake_pkg.load_dataset.return_value = []
    with mock.patch.dict("sys.modules", {"pii_anon_datasets": fake_pkg}):
        load_pii_anon_data()
    assert fake_pkg.load_dataset.call_args.kwargs.get("split") == "train"


def test_load_pii_anon_data_split_is_overridable() -> None:
    fake_pkg = mock.MagicMock()
    fake_pkg.load_dataset.return_value = []
    with mock.patch.dict("sys.modules", {"pii_anon_datasets": fake_pkg}):
        load_pii_anon_data(split="dev")
    assert fake_pkg.load_dataset.call_args.kwargs.get("split") == "dev"
