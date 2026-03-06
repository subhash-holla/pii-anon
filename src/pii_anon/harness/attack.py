from __future__ import annotations

from typing import Any


def reconstruction_attack_score(masked_texts: list[str], known_tokens: list[str]) -> dict[str, Any]:
    recovered = 0
    for token in known_tokens:
        if any(token in text for text in masked_texts):
            recovered += 1
    score = recovered / max(1, len(known_tokens))
    return {"attack": "reconstruction_rank_style", "success_rate": score, "recovered": recovered, "total": len(known_tokens)}
