"""sp7 panel #2 — value-consistent (coreference) masking, production wire-in.

The primary transform masks each DETECTED span. A distinctive value detected at
ONE mention but repeated verbatim elsewhere (a mention the detector missed, or
the same value in another field) otherwise survives in the output and is
trivially reconstructable. The pass-2 sweep redacts every remaining verbatim
occurrence of a detected sticky-type value with the SAME replacement.

Direction: ADDITIVE on the masking path — it can only redact MORE, with a
replacement already deemed safe for that value; it never narrows a pattern,
drops a finding, or touches detection. So detection metrics are byte-identical
and the coverage invariant coverage(on) ⊇ coverage(off) holds.
"""
from __future__ import annotations

import re

from pii_anon.config import CoreConfig
from pii_anon.orchestrator import AsyncPIIOrchestrator, PIIOrchestrator
from pii_anon.transforms.value_consistency import (
    VALUE_CONSISTENT_MIN_LEN,
    VALUE_CONSISTENT_TYPES,
    build_surfaces,
    sweep_residual_occurrences,
)
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan

_PROFILE = ProcessingProfileSpec(profile_id="p1", mode="weighted_consensus")
_SEG = SegmentationPlan(enabled=False)


def _audit(
    etype: str, start: int, end: int, mention: str, repl: str, field: str, cluster: str = "c1"
) -> dict:
    return {
        "entity_type": etype,
        "span": {"start": start, "end": end},
        "mention_text": mention,
        "canonical_text": mention,
        "replacement": repl,
        "field_path": field,
        "cluster_id": cluster,
    }


class TestBuildSurfaces:
    def test_sticky_types_only(self) -> None:
        audit = [
            _audit("PERSON_NAME", 0, 9, "Jane Roe", "<N>", "a"),
            _audit("DATE_TIME", 0, 10, "2021-01-01", "<D>", "a"),  # not sticky
        ]
        surfaces = build_surfaces(audit)
        assert [s[0] for s in surfaces] == ["Jane Roe"]

    def test_min_len_bounds_over_masking(self) -> None:
        audit = [
            _audit("PERSON_NAME", 0, 3, "Bob", "<N>", "a"),  # len 3 < 4, excluded
            _audit("PERSON_NAME", 0, 4, "Bram", "<N>", "a"),  # len 4, kept
        ]
        assert [s[0] for s in build_surfaces(audit)] == ["Bram"]

    def test_dedup_and_longest_first(self) -> None:
        audit = [
            _audit("ORGANIZATION", 0, 4, "Acme", "<O1>", "a"),
            _audit("ORGANIZATION", 0, 12, "Acme Holdings", "<O2>", "b"),
            _audit("ORGANIZATION", 0, 4, "Acme", "<O1>", "c"),  # dup surface
        ]
        surfaces = build_surfaces(audit)
        assert [s[0] for s in surfaces] == ["Acme Holdings", "Acme"]  # longest first, deduped

    def test_both_mention_and_canonical_harvested(self) -> None:
        audit = [{
            "entity_type": "PERSON_NAME",
            "span": {"start": 0, "end": 8},
            "mention_text": "J. Smith",
            "canonical_text": "John Smith",
            "replacement": "<N>",
            "field_path": "a",
        }]
        got = {s[0] for s in build_surfaces(audit)}
        assert got == {"J. Smith", "John Smith"}

    def test_every_sticky_type_is_a_real_type_name(self) -> None:
        # guard: the shared allowlist should be plausible entity types (upper snake)
        assert all(re.fullmatch(r"[A-Z_]+", t) for t in VALUE_CONSISTENT_TYPES)
        assert VALUE_CONSISTENT_MIN_LEN >= 4


class TestSweepResidualOccurrences:
    def test_skips_covered_first_masks_residual(self) -> None:
        text = "later Bram and Bram again"
        # pass-1 covered the first "Bram" (6..10); second is residual (15..19)
        residuals = sweep_residual_occurrences(
            text, [("Bram", "<N>", "PERSON_NAME", "c1")], covered=[(6, 10)]
        )
        assert residuals == [(15, 19, "<N>", "PERSON_NAME", "c1")]

    def test_no_residual_when_all_covered(self) -> None:
        text = "Bram"
        surfaces = [("Bram", "<N>", "PERSON_NAME", "c1")]
        assert sweep_residual_occurrences(text, surfaces, [(0, 4)]) == []

    def test_multiple_residuals_all_captured(self) -> None:
        text = "Acme Acme Acme"
        residuals = sweep_residual_occurrences(
            text, [("Acme", "<O>", "ORGANIZATION", "c2")], covered=[(0, 4)]
        )
        assert residuals == [
            (5, 9, "<O>", "ORGANIZATION", "c2"),
            (10, 14, "<O>", "ORGANIZATION", "c2"),
        ]

    def test_residuals_never_overlap_each_other(self) -> None:
        # a surface that is a prefix of itself must not double-mask overlapping ranges
        text = "aaaa"  # surface "aaa" occurs at 0 and 1 (overlapping)
        residuals = sweep_residual_occurrences(text, [("aaa", "<X>", "PERSON_NAME", "c1")], covered=[])
        # only the first non-overlapping hit is taken
        assert residuals == [(0, 3, "<X>", "PERSON_NAME", "c1")]


class TestApplyValueConsistentMaskingDirect:
    """Unit-test the orchestrator hook directly with a synthetic pass-1 audit,
    so behavior is independent of detector quirks."""

    def _orch(self) -> AsyncPIIOrchestrator:
        return AsyncPIIOrchestrator(token_key="k")

    def test_residual_masked_with_same_replacement(self) -> None:
        payload = {"a": "Alastair Penn signed", "b": "later Alastair Penn and Alastair Penn"}
        transformed = {"a": "<N> signed", "b": "later <N> and Alastair Penn"}
        link_audit = [
            _audit("PERSON_NAME", 0, 13, "Alastair Penn", "<N>", "a"),
            _audit("PERSON_NAME", 6, 19, "Alastair Penn", "<N>", "b"),
        ]
        self._orch()._apply_value_consistent_masking(payload, transformed, link_audit)
        # field b's second (residual) mention is now masked with the SAME token
        assert transformed["b"] == "later <N> and <N>"
        assert "Alastair Penn" not in transformed["b"]
        # field a had no residual → untouched
        assert transformed["a"] == "<N> signed"

    def test_no_residual_is_byte_identical(self) -> None:
        payload = {"a": "Alastair Penn only once"}
        transformed = {"a": "<N> only once"}
        link_audit = [_audit("PERSON_NAME", 0, 13, "Alastair Penn", "<N>", "a")]
        before_audit = list(link_audit)
        self._orch()._apply_value_consistent_masking(payload, transformed, link_audit)
        assert transformed == {"a": "<N> only once"}
        assert link_audit == before_audit  # no pass-2 entries appended

    def test_pass2_audit_entry_recorded(self) -> None:
        payload = {"b": "Alastair Penn and Alastair Penn"}
        transformed = {"b": "<N> and Alastair Penn"}
        link_audit = [_audit("PERSON_NAME", 0, 13, "Alastair Penn", "<N>", "b", cluster="clX")]
        self._orch()._apply_value_consistent_masking(payload, transformed, link_audit)
        pass2 = [e for e in link_audit if e.get("rule") == "value-consistent-coreference"]
        assert len(pass2) == 1
        assert pass2[0]["field_path"] == "b"
        assert pass2[0]["replacement"] == "<N>"
        assert pass2[0]["entity_type"] == "PERSON_NAME"
        # the residual is attributed to the SAME identity cluster as its source
        assert pass2[0]["cluster_id"] == "clX"

    def test_non_sticky_type_not_swept(self) -> None:
        # a DATE repeated verbatim is NOT redacted (bounded over-masking)
        payload = {"b": "on 2021-01-01 and again 2021-01-01"}
        transformed = {"b": "on <D> and again 2021-01-01"}
        link_audit = [_audit("DATE_TIME", 3, 13, "2021-01-01", "<D>", "b")]
        self._orch()._apply_value_consistent_masking(payload, transformed, link_audit)
        assert transformed["b"] == "on <D> and again 2021-01-01"  # unchanged


class TestEndToEnd:
    def _pair(self) -> tuple[PIIOrchestrator, PIIOrchestrator]:
        on = PIIOrchestrator(token_key="k")
        cfg = CoreConfig.default()
        cfg.transform.value_consistent_masking = False
        off = PIIOrchestrator(token_key="k", config=cfg)
        return on, off

    def _run(self, orch: PIIOrchestrator, payload: dict) -> dict:
        return orch.run(dict(payload), profile=_PROFILE, segmentation=_SEG, scope="s", token_version=1)

    def test_repeated_name_all_occurrences_masked(self) -> None:
        # "Bram" is bridged from the labeled cue at mention 1; the prose repeats
        # ("Bram X and Bram X") are the reconstruction leak the sweep closes.
        pay = {"t": "patient name: Bram X reviewed. Bram X and Bram X returned."}
        on, off = self._pair()
        out_on = self._run(on, pay)["transformed_payload"]["t"]
        out_off = self._run(off, pay)["transformed_payload"]["t"]
        # OFF leaks the repeated verbatim name; ON does not.
        assert "Bram" in out_off
        assert "Bram" not in out_on
        # all masked mentions use the SAME token (value consistency)
        tokens = set(re.findall(r"<PERSON_NAME:v1:tok_[A-Za-z0-9_-]+>", out_on))
        assert len(tokens) == 1

    def test_detection_findings_byte_identical(self) -> None:
        pay = {"t": "patient name: Bram X reviewed. Bram X and Bram X returned."}
        on, off = self._pair()
        f_on = self._run(on, pay)["ensemble_findings"]
        f_off = self._run(off, pay)["ensemble_findings"]
        # the sweep never touches detection → findings identical on/off
        assert f_on == f_off

    def test_coverage_superset_invariant(self) -> None:
        # every original substring absent from the OFF output is also absent
        # from the ON output (ON redacts a superset — never less).
        pay = {"t": "account holder: Vexa 9. The Vexa 9 record; Vexa 9 archived."}
        on, off = self._pair()
        out_on = self._run(on, pay)["transformed_payload"]["t"]
        out_off = self._run(off, pay)["transformed_payload"]["t"]
        if "Vexa" not in out_off:  # if off already scrubbed it, on must too
            assert "Vexa" not in out_on
        # ON must never REINTRODUCE a value off had masked
        assert out_on.count("Vexa") <= out_off.count("Vexa")

    def test_no_coreference_payload_is_byte_identical(self) -> None:
        # a payload with no repeated value → on and off produce identical output
        pay = {"name": "John Smith", "ssn": "123-45-6789", "note": "no repeats here"}
        on, off = self._pair()
        assert (
            self._run(on, pay)["transformed_payload"]
            == self._run(off, pay)["transformed_payload"]
        )
