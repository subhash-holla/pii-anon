from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class Segment:
    segment_id: str
    start_token: int
    end_token: int
    start_char: int
    end_char: int
    text: str


class Segmenter:
    def segment(self, text: str, *, max_tokens: int, overlap_tokens: int) -> list[Segment]:
        token_matches = list(re.finditer(r"\S+", text))
        token_count = len(token_matches)
        if token_count <= max_tokens:
            return [
                Segment(
                    segment_id="seg-0",
                    start_token=0,
                    end_token=token_count,
                    start_char=0,
                    end_char=len(text),
                    text=text,
                )
            ]

        segments: list[Segment] = []
        start = 0
        index = 0
        step = max(1, max_tokens - overlap_tokens)
        while start < token_count:
            end = min(token_count, start + max_tokens)
            start_char = token_matches[start].start()
            end_char = token_matches[end - 1].end()
            chunk = text[start_char:end_char]
            segments.append(
                Segment(
                    segment_id=f"seg-{index}",
                    start_token=start,
                    end_token=end,
                    start_char=start_char,
                    end_char=end_char,
                    text=chunk,
                )
            )
            if end >= token_count:
                break
            start += step
            index += 1
        return segments
