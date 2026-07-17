"""sp7 panel #1 — CUSTOMER_ID labeled-field type (genuine new masking coverage).

The improvement-panel found CUSTOMER_ID at 0% recall — a quasi-identifier
nemotron labels ~8,830 times that pii-anon never detected/masked. It rides the
existing A2 labeled-field ':'/'=' bridge (alnum_id kind), so it is strictly
ADDITIVE and leak-safe by construction, and it improves the PRODUCTION masking
path (not just a benchmark number).
"""
from __future__ import annotations

import pytest

from pii_anon.engines.regex_adapter import RegexEngineAdapter
from pii_anon.orchestrator import SUPPORTED_ENTITY_TYPES

_PROD = RegexEngineAdapter(enabled=True)


def _spans(text: str, etype: str) -> list[str]:
    return [
        text[f.span_start:f.span_end]
        for f in _PROD.detect({"text": text}, {"language": "en"})
        if str(f.entity_type) == etype and f.span_start is not None
    ]


class TestCustomerId:
    @pytest.mark.parametrize(
        "text,value",
        [
            ("Customer ID: CUST-88213 on file.", "CUST-88213"),
            ("Customer Number: 4471-A recorded.", "4471-A"),
            ("Client ID = CLT00934 for billing.", "CLT00934"),
            ("Account Holder ID: AH-55217", "AH-55217"),
        ],
    )
    def test_customer_id_detected(self, text: str, value: str) -> None:
        assert value in _spans(text, "CUSTOMER_ID"), f"{value!r} not in {_spans(text, 'CUSTOMER_ID')!r}"

    def test_customer_id_is_supported(self) -> None:
        # must be orchestrator-maskable, else the extracted value would be
        # dropped and leak (the sp6 inversion class).
        assert "CUSTOMER_ID" in SUPPORTED_ENTITY_TYPES

    def test_no_value_no_emit(self) -> None:
        # the alnum_id gate: a non-conforming value must not emit.
        assert _spans("Customer ID: pending assignment", "CUSTOMER_ID") == []

    def test_additive_masked_on_production_path(self) -> None:
        assert "CUST-88213" in _spans("Customer ID: CUST-88213 today.", "CUSTOMER_ID")
