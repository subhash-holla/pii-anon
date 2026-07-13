"""sp7 panel (robustness) — deny-list canonical folding.

Membership was ``matched_text.lower() in deny_set``, so a denied entry like
"new york" failed to match "New  York" (double space) or "New York." (trailing
period) — the FP slipped back in. Canonical folding (lowercase +
whitespace-collapse + strip SURROUNDING punctuation) is applied symmetrically
to entries and to the matched text. It deliberately does NOT fold unicode
confusables: a homoglyph variant ("Jоhn Doe", Cyrillic о) is suspicious input
that must stay MASKABLE, never denied — a leak-direction safeguard.
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex.deny_list import DenyListManager, canonical_deny_form


class TestCanonicalDenyForm:
    @pytest.mark.parametrize(
        "raw,canon",
        [
            ("New  York", "new york"),
            ("New York.", "new york"),
            ("  New\tYork  ", "new york"),
            ('"New York"', "new york"),
            ("st. petersburg", "st. petersburg"),  # interior punct preserved
        ],
    )
    def test_canonical_form(self, raw: str, canon: str) -> None:
        assert canonical_deny_form(raw) == canon


class TestFoldedMembership:
    def _mgr(self) -> DenyListManager:
        return DenyListManager(
            deny_config={"enabled": True, "lists": {"PERSON_NAME": ["New York", "John Doe"]}}
        )

    @pytest.mark.parametrize(
        "text",
        ["New York", "New  York", "New York.", "  new york  ", "NEW YORK"],
    )
    def test_whitespace_and_punct_variants_denied(self, text: str) -> None:
        assert self._mgr().is_denied("PERSON_NAME", text)

    def test_homoglyph_variant_not_denied(self) -> None:
        # Cyrillic 'о' in "New Yоrk" — a confusable is suspicious, must NOT be
        # denied (stays maskable). Leak-direction safeguard.
        assert not self._mgr().is_denied("PERSON_NAME", "New Yоrk")

    def test_real_name_still_maskable(self) -> None:
        # a name NOT on the deny list is unaffected.
        assert not self._mgr().is_denied("PERSON_NAME", "Maria Garcia")

    def test_empty_canonical_not_denied(self) -> None:
        assert not self._mgr().is_denied("PERSON_NAME", "...")
