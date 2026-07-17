"""Record-level paired-outcome assembly — S3-04 (FR-004).

Pins :func:`assemble_paired_set` (temp-local ``# SWITCH-POINT(S6)`` builder) and
the :class:`PairedComparisonSet` it produces:

* per-record F1 → ``{(sys_i, sys_j): (wins_i, wins_j, ties, n)}`` with a tie-band
  ε (``|f1_i − f1_j| ≤ ε`` ⇒ tie);
* the tie-band ε boundary is exact (``== ε`` is a tie, ``ε + δ`` is a win);
* **ragged inputs are rejected LOUDLY** — this DELIBERATELY DIVERGES from the
  silent ``min_len`` truncation at ``competitor_compare.py:2871`` (a reviewer
  must not "fix" the loud raise back into a silent truncation);
* ``< 2`` systems / ``0`` records are rejected;
* determinism (AX-002);
* the two adapter views — ``to_paired_counts()`` (the **integer** 3-tuple shape
  S3-02 ``fit_paired`` consumes, ties EXCLUDED so totals stay integral) and
  ``to_paired_counts_with_ties()`` (the 4-tuple the Davidson path consumes).

These primitives are pure-stdlib + numpy and take a plain
``dict[str, list[float]]`` (NOT an ``ExternalEvaluationResult``) so the rating
layer stays import-isolated from ``evaluation/`` (S3-05 boundary).

Test-type tags: ``[UNIT-TEST]`` ``[CONTRACT-TEST]`` ``[PROPERTY-TEST]``.
"""

from __future__ import annotations

import math

import pytest

from pii_anon.eval_framework.rating.paired_set import (
    PairedComparisonSet,
    assemble_paired_set,
)


# ---------------------------------------------------------------------------
# fr_004: assembly from per_record_f1 → (wins_i, wins_j, ties, n)
# ---------------------------------------------------------------------------


def test_fr_004_assemble_counts_wins_losses_ties_per_pair() -> None:
    """[UNIT-TEST] Given per-record F1 for 2 systems, when assembled, then each
    pair carries (wins_i, wins_j, ties, n) summing to n, with the tie band
    classifying near-equal records as ties."""
    per_record = {
        "sys_a": [0.9, 0.9, 0.5, 0.2, 0.7],
        "sys_b": [0.1, 0.9, 0.5, 0.8, 0.7],
    }
    # tie band wide enough that exact-equal records (idx 1,2,4) count as ties.
    pset = assemble_paired_set(per_record, tie_eps=1e-9)
    counts = pset.to_paired_counts_with_ties()
    assert set(counts) == {("sys_a", "sys_b")}
    wins_i, wins_j, ties, n = counts[("sys_a", "sys_b")]
    # idx0: a>b win_i; idx1: tie; idx2: tie; idx3: b>a win_j; idx4: tie.
    assert (wins_i, wins_j, ties, n) == (1, 1, 3, 5)
    assert wins_i + wins_j + ties == n


def test_fr_004_pair_keys_are_sorted_and_all_pairs_present() -> None:
    """[CONTRACT-TEST] K systems → C(K,2) pairs; keys are (sys_i, sys_j) with
    sys_i < sys_j (sorted) so iteration / column order is deterministic."""
    per_record = {
        "z_sys": [0.4, 0.4, 0.4],
        "a_sys": [0.9, 0.9, 0.9],
        "m_sys": [0.6, 0.6, 0.6],
    }
    pset = assemble_paired_set(per_record, tie_eps=1e-9)
    counts = pset.to_paired_counts_with_ties()
    assert set(counts) == {
        ("a_sys", "m_sys"),
        ("a_sys", "z_sys"),
        ("m_sys", "z_sys"),
    }
    for (i, j) in counts:
        assert i < j, "pair keys must be sorted (sys_i < sys_j)"


def test_fr_004_n_equals_record_count_for_every_pair() -> None:
    """[UNIT-TEST] Every pair's total n equals the (shared) record count."""
    per_record = {
        "a": [0.1, 0.2, 0.3, 0.4],
        "b": [0.5, 0.5, 0.5, 0.5],
        "c": [0.9, 0.8, 0.7, 0.6],
    }
    pset = assemble_paired_set(per_record, tie_eps=1e-9)
    for _pair, (wi, wj, ties, n) in pset.to_paired_counts_with_ties().items():
        assert n == 4
        assert wi + wj + ties == 4


# ---------------------------------------------------------------------------
# fr_004: tie-band ε boundary is EXACT
# ---------------------------------------------------------------------------


def test_fr_004_tie_band_boundary_equal_to_eps_is_a_tie() -> None:
    """[UNIT-TEST] A gap EXACTLY equal to ε classifies as a tie (|Δ| ≤ ε).

    Uses an exactly-representable gap (``0.0`` vs ``eps``) so the boundary test
    is not at the mercy of binary float rounding (``0.55 − 0.50`` is NOT exactly
    ``0.05``)."""
    eps = 0.05
    per_record = {
        "a": [0.0],
        "b": [eps],  # gap == eps exactly (0.0 + eps − 0.0 == eps)
    }
    counts = assemble_paired_set(per_record, tie_eps=eps).to_paired_counts_with_ties()
    wi, wj, ties, n = counts[("a", "b")]
    assert (wi, wj, ties, n) == (0, 0, 1, 1), "gap == eps must be a tie"


def test_fr_004_tie_band_boundary_just_over_eps_is_a_win() -> None:
    """[UNIT-TEST] A gap just over ε is a decisive win for the larger F1, not a
    tie — the boundary is inclusive on ties only."""
    eps = 0.05
    per_record = {
        "a": [0.0],
        "b": [eps * 2.0],  # gap = 0.10 > eps, exactly representable enough
    }
    counts = assemble_paired_set(per_record, tie_eps=eps).to_paired_counts_with_ties()
    wi, wj, ties, n = counts[("a", "b")]
    # b is larger → win for b (the j side, since 'a' < 'b' sorted).
    assert (wi, wj, ties, n) == (0, 1, 0, 1), "gap > eps must be a decisive win"


def test_fr_004_default_tie_eps_is_tiny() -> None:
    """[UNIT-TEST] The default ε is a tiny float-equality band (1e-9) so only
    numerically-identical F1 counts as a tie by default."""
    per_record = {"a": [0.5, 0.5], "b": [0.5, 0.5000001]}
    counts = assemble_paired_set(per_record).to_paired_counts_with_ties()
    wi, wj, ties, n = counts[("a", "b")]
    # idx0 exactly equal → tie; idx1 differs by 1e-7 > 1e-9 → decisive.
    assert ties == 1
    assert wi + wj == 1


# ---------------------------------------------------------------------------
# fr_004: ragged inputs rejected LOUDLY (the deliberate divergence)
# ---------------------------------------------------------------------------


def test_fr_004_ragged_inputs_raise_naming_systems_and_lengths() -> None:
    """[CONTRACT-TEST] Mismatched per-system list lengths raise ValueError naming
    the systems and lengths.

    This DELIBERATELY DIVERGES from the silent ``min_len`` truncation at
    ``competitor_compare.py:2871`` (which would silently align two ragged lists
    to the shorter). Record-level paired outcomes must be index-aligned over the
    SAME records, so ragged input is a data error, not something to truncate.
    """
    per_record = {
        "sys_a": [0.9, 0.8, 0.7],
        "sys_b": [0.1, 0.2],  # one shorter
    }
    with pytest.raises(ValueError) as exc:
        assemble_paired_set(per_record)
    msg = str(exc.value)
    assert "sys_a" in msg and "sys_b" in msg, f"must name the systems: {msg!r}"
    assert "3" in msg and "2" in msg, f"must name the lengths: {msg!r}"


def test_fr_004_ragged_does_not_truncate_to_min_len() -> None:
    """[CONTRACT-TEST] Assert the builder does NOT silently truncate ragged input
    to the shorter length (the anti-competitor_compare regression).

    If it truncated, a 5-vs-3 input would yield n=3 with no error. We assert it
    RAISES instead, so the truncation behaviour cannot creep back in.
    """
    per_record = {
        "sys_a": [0.9, 0.8, 0.7, 0.6, 0.5],  # len 5
        "sys_b": [0.1, 0.2, 0.3],  # len 3
    }
    with pytest.raises(ValueError):
        # A silent-truncation impl would NOT raise (it would emit n=3); the raise
        # is the teeth of the divergence.
        assemble_paired_set(per_record)


# ---------------------------------------------------------------------------
# fr_004: degenerate inputs rejected loudly
# ---------------------------------------------------------------------------


def test_fr_004_fewer_than_two_systems_rejected() -> None:
    """[CONTRACT-TEST] < 2 systems cannot form a pair → ValueError."""
    with pytest.raises(ValueError) as exc:
        assemble_paired_set({"only_one": [0.5, 0.6]})
    assert "2" in str(exc.value) or "two" in str(exc.value).lower()


def test_fr_004_zero_records_rejected() -> None:
    """[CONTRACT-TEST] 0 records (empty per-system lists) → ValueError; you
    cannot assemble paired outcomes from no records."""
    with pytest.raises(ValueError) as exc:
        assemble_paired_set({"a": [], "b": []})
    assert "record" in str(exc.value).lower() or "empty" in str(exc.value).lower()


def test_fr_004_empty_mapping_rejected() -> None:
    """[CONTRACT-TEST] An empty mapping → ValueError (no systems)."""
    with pytest.raises(ValueError):
        assemble_paired_set({})


# ---------------------------------------------------------------------------
# ax_002: determinism
# ---------------------------------------------------------------------------


def test_ax_002_assemble_is_deterministic() -> None:
    """[UNIT-TEST] Identical input → identical counts across calls (AX-002)."""
    per_record = {
        "a": [0.1, 0.9, 0.5, 0.5, 0.3],
        "b": [0.9, 0.1, 0.5, 0.6, 0.3],
        "c": [0.5, 0.5, 0.5, 0.5, 0.5],
    }
    c1 = assemble_paired_set(per_record, tie_eps=1e-3).to_paired_counts_with_ties()
    c2 = assemble_paired_set(dict(per_record), tie_eps=1e-3).to_paired_counts_with_ties()
    assert c1 == c2


# ---------------------------------------------------------------------------
# fr_004: the TWO adapter views — 3-tuple (S3-02 integer) and 4-tuple (Davidson)
# ---------------------------------------------------------------------------


def test_fr_004_to_paired_counts_excludes_ties_and_is_integral() -> None:
    """[CONTRACT-TEST] ``to_paired_counts()`` returns the S3-02 3-tuple shape
    ``(wins_i, wins_j, n)`` with ties EXCLUDED (n' = wins_i + wins_j) so totals
    stay INTEGER — never half-split ties (bayes_bt ``_normalize_comparisons``
    does ``int(round(...))`` and half-integers would corrupt counts)."""
    per_record = {
        "a": [0.9, 0.5, 0.5, 0.5, 0.1],  # vs b: 1 win_a, 3 ties, 1 win_b
        "b": [0.1, 0.5, 0.5, 0.5, 0.9],
    }
    pset = assemble_paired_set(per_record, tie_eps=1e-9)
    three = pset.to_paired_counts()
    wins_i, wins_j, n = three[("a", "b")]
    # ties EXCLUDED → n' = 1 + 1 = 2 (NOT 5), and all values are integers.
    assert (wins_i, wins_j, n) == (1, 1, 2)
    assert isinstance(wins_i, int) and isinstance(wins_j, int) and isinstance(n, int)
    # No half-integers anywhere (would corrupt _normalize_comparisons rounding).
    assert float(wins_i).is_integer() and float(wins_j).is_integer()


def test_fr_004_to_paired_counts_with_ties_keeps_full_n() -> None:
    """[CONTRACT-TEST] ``to_paired_counts_with_ties()`` keeps the FULL n
    (= wins_i + wins_j + ties) — the 4-tuple the Davidson likelihood needs."""
    per_record = {
        "a": [0.9, 0.5, 0.5, 0.5, 0.1],
        "b": [0.1, 0.5, 0.5, 0.5, 0.9],
    }
    pset = assemble_paired_set(per_record, tie_eps=1e-9)
    four = pset.to_paired_counts_with_ties()
    wins_i, wins_j, ties, n = four[("a", "b")]
    assert (wins_i, wins_j, ties, n) == (1, 1, 3, 5)
    assert wins_i + wins_j + ties == n


def test_fr_004_two_adapters_agree_on_decisive_counts() -> None:
    """[CONTRACT-TEST] The two views must agree on the decisive counts: the
    3-tuple's (wins_i, wins_j) equal the 4-tuple's, and the 3-tuple's n equals
    the 4-tuple's (n − ties). The only difference is ties handling."""
    per_record = {
        "a": [0.1, 0.9, 0.5, 0.2, 0.8, 0.5],
        "b": [0.9, 0.1, 0.5, 0.8, 0.2, 0.5],
        "c": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    }
    pset = assemble_paired_set(per_record, tie_eps=1e-3)
    three = pset.to_paired_counts()
    four = pset.to_paired_counts_with_ties()
    assert set(three) == set(four)
    for pair in three:
        wi3, wj3, n3 = three[pair]
        wi4, wj4, ties4, n4 = four[pair]
        assert (wi3, wj3) == (wi4, wj4)
        assert n3 == n4 - ties4


def test_fr_004_paired_set_is_a_dataclass_carrying_systems_and_records() -> None:
    """[UNIT-TEST] PairedComparisonSet exposes the system list (sorted) and the
    record count, so downstream code can size the Davidson model."""
    per_record = {"b": [0.1, 0.2], "a": [0.9, 0.8]}
    pset = assemble_paired_set(per_record, tie_eps=1e-9)
    assert isinstance(pset, PairedComparisonSet)
    assert pset.systems == ["a", "b"], "systems sorted"
    assert pset.n_records == 2


def test_fr_004_all_ties_design_has_zero_decisive_counts() -> None:
    """[UNIT-TEST] A design where every record ties → wins_i = wins_j = 0, ties =
    n; to_paired_counts() then has n' = 0 (no decisive records)."""
    per_record = {"a": [0.5, 0.5, 0.5], "b": [0.5, 0.5, 0.5]}
    pset = assemble_paired_set(per_record, tie_eps=1e-9)
    wi, wj, ties, n = pset.to_paired_counts_with_ties()[("a", "b")]
    assert (wi, wj, ties, n) == (0, 0, 3, 3)
    wi3, wj3, n3 = pset.to_paired_counts()[("a", "b")]
    assert (wi3, wj3, n3) == (0, 0, 0)


def test_fr_004_negative_tie_eps_rejected() -> None:
    """[CONTRACT-TEST] A negative tie band is meaningless → ValueError."""
    with pytest.raises(ValueError):
        assemble_paired_set({"a": [0.5], "b": [0.5]}, tie_eps=-0.1)


def test_fr_004_nan_f1_rejected_loud() -> None:
    """[CONTRACT-TEST] A NaN F1 cannot be classified (NaN comparisons are
    silently False) → reject loudly rather than silently mis-bucket it."""
    per_record = {"a": [0.5, math.nan], "b": [0.5, 0.5]}
    with pytest.raises(ValueError):
        assemble_paired_set(per_record)
