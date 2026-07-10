#!/usr/bin/env python3
"""pii-anon-swarm detection adapter — the sister library's swarm fusion pipeline.

The first-party SWARM detector from the sibling ``pii-anon-code`` library: an engine pool
(regex always; GLiNER/Presidio/Stanza when importable) merged through the four-layer swarm
fusion strategy with the recall-floor projection. Wraps the library's public BYO predictor
seam (``first_party_predictor("pii_anon_swarm")``), which emits NATIVE pii-anon labels;
the canonical-63 projection is the SAME ``LABEL_MAP`` as the vanilla adapter (one
projection, two detectors — drift between maps would silently skew the comparison).
Model-backed and non-deterministic when GLiNER joins the pool. Loads LAZILY in ``build()``
(NFR-050). Install: ``pip install -e ../pii-anon-code``.
"""

from __future__ import annotations

import importlib.util

from baselines.pii_anon_baseline import LABEL_MAP
from pii_anon_datasets.baselines.contract import AdapterSpan, coverage_of


class _PiiAnonSwarmAdapter:
    name = "pii_anon_swarm"
    model_id = "pii-anon swarm (4-layer fusion: regex + NER pool)"
    label_map = LABEL_MAP
    deterministic = False  # GLiNER in the pool; regex-only fallback IS deterministic

    def available(self) -> bool:
        return importlib.util.find_spec("pii_anon") is not None

    def map_label(self, native: str) -> str | None:
        return LABEL_MAP.get((native or "").strip().upper())

    def build(self) -> object:
        try:
            from pii_anon.eval_framework.byo_pipeline import first_party_predictor
        except ImportError as e:  # pragma: no cover - exercised only on a lib-less checkout
            raise RuntimeError("pii-anon not installed — pip install -e ../pii-anon-code") from e
        return first_party_predictor("pii_anon_swarm")

    def detect(self, text: str, model: object) -> list[AdapterSpan]:
        if not text:
            return []
        out: list[AdapterSpan] = []
        for native, start, end in model(text):  # type: ignore[operator]
            et = self.map_label(str(native))
            if et is not None:
                s, e = int(start), int(end)
                out.append(AdapterSpan(s, e, et, text[s:e]))
        return out

    def coverage(self) -> int:
        return coverage_of(LABEL_MAP)


ADAPTER = _PiiAnonSwarmAdapter()
