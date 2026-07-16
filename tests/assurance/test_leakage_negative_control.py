"""Negative-control FPR characterization for the residual-leakage digit path.

Spec §6.2 (docs/superpowers/specs/2026-06-17-pii-assurance-report-design.md):
"Min-specificity guard ... characterize its FPR on a synthetic negative control;
cap substring-search complexity."

The digit-normalized path (`_surface_leaked`) concatenates ALL digits of a
transformed output into one blob and asks whether a victim's >=7-digit PII digit
string survives as a substring. This deliberately catches a separator-REFORMATTED
leak — "123-45-6789" -> "123 45 6789" -> "123.45.6789" all collapse to the same
digit blob — at the cost of a rare false positive when unrelated small numbers in a
clean output happen to concatenate into a blob that CONTAINS a victim's digit string
(e.g. "1 2 3 4 5 6 7 8 9" -> "123456789" == SSN 123-45-6789).

That failure is in the SAFE direction: leakage is OVER-reported, never hidden, and
the dimension is always labelled a LOWER BOUND. These tests (a) pin the realistic
digit-path FPR below a committed bound on a large synthetic clean corpus, and (b) pin
the one contrived trigger as a documented, accepted limitation.

Why the blob is NOT reset at non-digit boundaries
-------------------------------------------------
Matching per digit-run instead of the global concatenation would eliminate the
cross-token coincidence, but it would ALSO drop the space/separator-reformatted-SSN
catch that is the entire purpose of the digit path: "123 45 6789" -> runs
"123","45","6789", none of which contains "123456789". A reformatted SSN and nine
stray single digits are indistinguishable at the digit level, so resetting the blob
converts a safe over-report into an UNSAFE under-report (a hidden leak), violating the
assessor's fail-closed contract. The space-separated catch is already a committed
guarantee (tests/assurance/test_close_fixes_r4.py::test_short_digit_run_not_normalized_leak),
so the global concatenation is retained by design and the rare FP is documented here.

Substring-search complexity is bounded: the needle is a fixed victim digit string
(>= 7 digits), the haystack is `_digits(out)` bounded by output length, and Python's
`str.__contains__` is linear in haystack length — no quadratic blow-up.
"""

from __future__ import annotations

import random

from pii_anon.assurance.assessors.leakage import _digits, _norm_text, _surface_leaked

# Committed bound: realistic digit-blob FPR on clean redactions is ~0 (measured ~0%
# over 200k pairs). 1e-4 leaves headroom for the rare random coincidence without ever
# admitting a systematic false-leak regime.
_COMMITTED_FPR_BOUND = 1e-4

# Clean redacted-output templates carrying the kinds of unrelated small numbers a real
# transformed output legitimately contains (dates, prices, counts, steps, versions,
# durations, ratings, coordinates). None contains any victim PII or the contrived
# sequential trigger.
_CLEAN_TEMPLATES = (
    "Meeting at {a} PM on day {b}; {c} attendees confirmed.",
    "Order #{a} shipped: {b} items, ${c}.{d} total.",
    "Step {a}: open. Step {b}: review. Step {c}: submit.",
    "Call lasted {a} minutes with {b} follow-ups over {c} days.",
    "Version {a}.{b}.{c} released; {d} bug fixes this cycle.",
    "Rated {a} of {b} by {c} reviewers in quarter {d}.",
    "Section {a}.{b}, paragraph {c}, line {d}.",
    "Batch {a} of {b}: {c} units at {d} percent yield.",
    "Page {a}, figure {b}, table {c}, appendix {d}.",
    "Temperature {a}C, humidity {b} percent, {c} sensors online.",
)


def _leaked(surface: str, out: str) -> bool:
    """`_surface_leaked` with the per-output blobs the assessor would pass it."""
    return _surface_leaked(surface, out, _digits(out), _norm_text(out))


def _clean_corpus(n: int, rng: random.Random) -> list[str]:
    rows = []
    for _ in range(n):
        tmpl = rng.choice(_CLEAN_TEMPLATES)
        # small unrelated numbers (0-999) — the realistic regime for clean outputs
        rows.append(tmpl.format(**{k: rng.randint(0, 999) for k in "abcd"}))
    return rows


def _victim_ssns(n: int, rng: random.Random) -> list[str]:
    """Distinct canonical 3-2-4 SSN surfaces (9 distinctive digits each)."""
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        d = "".join(str(rng.randint(0, 9)) for _ in range(9))
        if d in seen:
            continue
        seen.add(d)
        out.append(f"{d[:3]}-{d[3:5]}-{d[5:]}")
    return out


def test_digit_path_fpr_below_committed_bound() -> None:
    """Over a large synthetic clean corpus the digit path's FPR stays below the bound."""
    rng = random.Random(20260617)
    outputs = _clean_corpus(2000, rng)
    victims = _victim_ssns(100, rng)

    # the negative control must be genuinely negative: no victim digit string and no
    # contrived sequential trigger may be planted in the clean corpus by construction.
    victim_digits = {_digits(v) for v in victims}
    assert not any("1 2 3 4 5 6 7 8 9" in o for o in outputs)
    assert not any(vd in _digits(o) for o in outputs for vd in victim_digits)

    pairs = false_positives = 0
    for out in outputs:
        od, on = _digits(out), _norm_text(out)
        for v in victims:
            pairs += 1
            if _surface_leaked(v, out, od, on):
                false_positives += 1

    assert pairs == 200_000  # 2000 clean outputs x 100 victims
    fpr = false_positives / pairs
    assert fpr <= _COMMITTED_FPR_BOUND, (
        f"digit-path FPR {fpr:.2e} ({false_positives}/{pairs}) exceeds committed bound "
        f"{_COMMITTED_FPR_BOUND:.0e}"
    )


def test_contrived_sequential_trigger_is_a_documented_limitation() -> None:
    """DOCUMENTED LIMITATION: nine ascending stray digits forge an SSN digit blob.

    "1 2 3 4 5 6 7 8 9" -> "123456789" is indistinguishable from a separator-
    reformatted SSN at the digit level. This over-reports (the SAFE direction, always
    labelled a LOWER BOUND) and is accepted rather than fixed by blob-resetting, which
    would drop real reformatted-SSN leaks (see module docstring).
    """
    out = "Steps: 1 2 3 4 5 6 7 8 9 then submit."
    assert _digits(out) == "123456789"
    assert _leaked("123-45-6789", out) is True  # the known false positive, by design


def test_separator_reformatted_ssn_is_still_caught() -> None:
    """The reason the global concatenation is retained: it catches reformatted leaks.

    A blob-reset "fix" for the cross-token FP would silently drop every one of these
    (an UNSAFE under-report), so they are pinned here as the design's load-bearing
    guarantee. Each is caught ONLY by the digit path (verbatim and normalized both miss).
    """
    ssn = "123-45-6789"
    for reformatted in ("123 45 6789", "123.45.6789", "123/45/6789", "123456789"):
        assert ssn not in reformatted  # verbatim path misses
        assert _norm_text(ssn) not in _norm_text(reformatted)  # normalized path misses
        assert _leaked(ssn, reformatted) is True  # digit path catches it


def test_short_unrelated_runs_do_not_false_leak() -> None:
    """A clean output of several short unrelated numbers does not forge a >=7-digit hit
    unless their concatenation genuinely contains the victim string."""
    out = "Order 12, line 34, qty 5, page 6 of 78."  # digit blob: "12345678"
    assert _digits(out) == "12345678"
    # victims whose 7-9 digit strings are NOT substrings of the blob must not leak.
    assert _leaked("555-66-7777", out) is False
    assert _leaked("(202) 555-0143", out) is False
