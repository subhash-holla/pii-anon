"""sp3 v2.2.0 re-baseline tranche — recover recall lost to substrate drift +
add the 3 GDPR Article-9 special-category types.

Grounded in a root-cause pass over the pii-anon-eval-data v2.2.0 DEV split
(2026-07-10). Two families:

1. **Value-class recovery** (PL-2): the v2.2.0 corpus obfuscates several
   secret-like values as base64 / short alphanumerics / zero-width-embedded
   strings behind their SPECIFIC field label ("CVV: MzIx", "PIN: ODQzNw==",
   "Policy: P0L-2694750", "Authentication Token: Bearer ..."). The legacy
   digit-only value classes (``_CVV=(\\d{3,4})``, ``_PIN=(\\d{4,6})``) could
   not reach them, so recall on these census-external types collapsed. The
   fixes gate on the same specific label (leak-safe, additive) and widen the
   value class to admit base64 / ``_ZW`` / OCR-style ``P0L``.

2. **Article-9 coverage** (PL-1): SEXUAL_ORIENTATION / TRADE_UNION_MEMBERSHIP /
   GENETIC_DATA (taxonomy 63->66). Detection keys on a SPECIFIC field label or
   on intrinsic value structure (orientation lexicon, gene symbols, dbSNP
   rs-IDs) — never the universal "Record shows X" generator filler (the
   eval-integrity line drawn in patterns.py ~line 1005 and the sp2
   remediation). These three earn EXTERNAL credit via the DATA harness
   LABEL_MAP; internally they are census-unreachable (documented in
   tests/test_pattern_label_alignment.py).

Every fixture below uses a SYNTHETIC value in the corpus's format (no gold
string is copied), so this file is a format contract, not eval-data coupling.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter

_ENGINE = RegexEngineAdapter(enabled=True)
_ZWSP = "​"


def _spans(text: str, entity_type: str) -> list[str]:
    return sorted(
        text[f.span_start : f.span_end]
        for f in _ENGINE.detect({"text": text}, {"language": "en"})
        if f.entity_type == entity_type
        and isinstance(f.span_start, int)
        and isinstance(f.span_end, int)
    )


@pytest.mark.parametrize(
    ("entity_type", "text", "expected"),
    [
        # ── PL-2: value-class recovery (base64 / alnum / zero-width) ──────
        # CVV: base64 + short-alpha forms behind the CVV label.
        ("CVV", "Record for Steven Perez, CVV: MzIx, contact steve@x.com", "MzIx"),
        ("CVV", "Payment failed. CVV: 2gO recorded.", "2gO"),
        ("CVV", "card 5015 9369 6510 5171 cvv: 986 keeps failing", "986"),
        # PIN: base64 + alnum forms behind the PIN label.
        ("PIN", "Record for Sandra Allen, PIN: ODQzNw==, contact s@x.com", "ODQzNw=="),
        ("PIN", "Kevin Lee, PIN: 348A, contact kevin@x.com", "348A"),
        ("PIN", f"ATM PIN: 3{_ZWSP}{_ZWSP}338 was used", f"3{_ZWSP}{_ZWSP}338"),
        # PASSWORD: code-config quoted form (pass := "...") + JSON.
        ("PASSWORD", 'config { pass := "SJyN$Pyc9lU" }', "SJyN$Pyc9lU"),
        ("PASSWORD", '{ "password": "bDVksBHZPmf##" }', "bDVksBHZPmf##"),
        # INSURANCE_POLICY_NUMBER: OCR P0L, zero-width, base64, alnum suffix.
        ("INSURANCE_POLICY_NUMBER", "Claimant: X / Policy: P0L-2694750 / DOB: 1959-05-01", "P0L-2694750"),
        ("INSURANCE_POLICY_NUMBER", f"Kevin, Insurance Policy Number: PO{_ZWSP}L-53091{_ZWSP}54, contact k@x.com", f"PO{_ZWSP}L-53091{_ZWSP}54"),
        ("INSURANCE_POLICY_NUMBER", "Insurance Policy Number: POL-48BS84B on file", "POL-48BS84B"),
        # AUTHENTICATION_TOKEN: Bearer / base64 / truncated-JWT placeholder.
        ("AUTHENTICATION_TOKEN", "Record for X, Authentication Token: Bearer daeb3e61Zc123d68A6TZ2f0cb2db6dcc5eZ3fI88, contact x@x.com", "Bearer daeb3e61Zc123d68A6TZ2f0cb2db6dcc5eZ3fI88"),
        ("AUTHENTICATION_TOKEN", "auth eyJhbGc... presented", "eyJhbGc..."),
        # Truncated-JWT placeholder followed by a sentence period: the span
        # must NOT swallow the 4th dot (was an over-capture FP + missed span).
        ("AUTHENTICATION_TOKEN", "Record shows eyJhbGc.... Next line.", "eyJhbGc..."),
        ("AUTHENTICATION_TOKEN", "Authentication Token: QmVhcmVyIDFkY2E5ZDk2NzVhZQ== issued", "QmVhcmVyIDFkY2E5ZDk2NzVhZQ=="),
        # Adversarial Bearer forms: B->8 OCR and zero-width-embedded keyword.
        ("AUTHENTICATION_TOKEN", "auth 8earer cI4E796e9A84Sggf7c8O56aSc48g2O802174SS9d done", "8earer cI4E796e9A84Sggf7c8O56aSc48g2O802174SS9d"),
        ("AUTHENTICATION_TOKEN", f"token Bea{_ZWSP}rer 7b6fa800713075e65af8bcb943c6c613d6c3c276 ok", f"Bea{_ZWSP}rer 7b6fa800713075e65af8bcb943c6c613d6c3c276"),

        # ── PL-1: Article-9 special-category types ───────────────────────
        # SEXUAL_ORIENTATION — label-gated closed lexicon (leak-safe).
        ("SEXUAL_ORIENTATION", "Record for Sandra Martin, Sexual Orientation: gay, contact s@x.com", "gay"),
        ("SEXUAL_ORIENTATION", "Patient intake — Sexual Orientation: pansexual", "pansexual"),
        ("SEXUAL_ORIENTATION", "HR form. Sexual Orientation: heterosexual.", "heterosexual"),
        # TRADE_UNION_MEMBERSHIP — label + value capture (proper-noun unions).
        ("TRADE_UNION_MEMBERSHIP", "Record for Betty Sanchez, Trade Union Membership: Verdi, contact b@x.com", "Verdi"),
        ("TRADE_UNION_MEMBERSHIP", "Thomas, Trade Union Membership: Teamsters Local 25, contact t@x.com", "Teamsters Local 25"),
        ("TRADE_UNION_MEMBERSHIP", "Mary, Trade Union Membership: NUT member, contact m@x.com", "NUT member"),
        # GENETIC_DATA — label + value capture AND intrinsic (gene / rs-ID).
        ("GENETIC_DATA", "Record for William, Genetic Data: CFTR ΔF508 homozygous, contact w@x.com", "CFTR ΔF508 homozygous"),
        ("GENETIC_DATA", "Lab note: BRCA1 c.68_69delAG pathogenic variant found", "BRCA1 c.68_69delAG pathogenic variant"),
        ("GENETIC_DATA", "Genetic Data: rs53576 GG genotype on file", "rs53576 GG genotype"),
    ],
)
def test_sp3_gold_shape_detected_with_exact_extent(
    entity_type: str, text: str, expected: str
) -> None:
    got = _spans(text, entity_type)
    assert expected in got, (
        f"{entity_type}: expected exact-extent span {expected!r} not detected "
        f"in {text!r}; got {got!r}"
    )


@pytest.mark.parametrize(
    ("entity_type", "text"),
    [
        # Leak-direction / FP guards: the universal "Record shows X" generator
        # filler must NOT anchor any Art-9 detection (eval-integrity axiom),
        # and bare orientation words in ordinary prose must not fire.
        ("SEXUAL_ORIENTATION", "The team had a gay old time at the party."),
        ("SEXUAL_ORIENTATION", "Record shows straight to the point analysis."),
        ("TRADE_UNION_MEMBERSHIP", "The peanut allergy panel and CGT were normal."),
        ("GENETIC_DATA", "The genetic algorithm converged after 500 epochs."),
    ],
)
def test_sp3_no_false_positive_on_prose_or_filler(entity_type: str, text: str) -> None:
    got = _spans(text, entity_type)
    assert got == [], f"{entity_type}: expected no detection in prose {text!r}, got {got!r}"
