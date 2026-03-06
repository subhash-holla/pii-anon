from __future__ import annotations

from typing import Any


class StreamSynthesizer:
    def run(self, events: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
        if mode == "split":
            out: list[dict[str, Any]] = []
            for e in events:
                text = str(e.get("text", ""))
                mid = max(1, len(text) // 2)
                out.append({**e, "text": text[:mid], "disturbance": "split-a"})
                out.append({**e, "text": text[mid:], "disturbance": "split-b"})
            return out
        if mode == "reorder":
            return list(reversed(events))
        if mode == "duplicate":
            out = []
            for e in events:
                out.extend([e, {**e, "disturbance": "duplicate"}])
            return out
        if mode == "truncate":
            return [{**e, "text": str(e.get("text", ""))[: max(1, len(str(e.get("text", ""))) // 2)], "disturbance": "truncate"} for e in events]
        return events
