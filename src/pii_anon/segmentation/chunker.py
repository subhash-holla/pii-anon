"""Memory-efficient streaming chunker for very large text (1M+ tokens).

The standard :class:`Segmenter` materializes all token positions upfront
which is fine for typical documents but consumes O(n) memory for n tokens.
:class:`StreamingChunker` uses a sliding-window scan that only keeps
O(max_tokens) boundary positions in memory at any time, yielding segments
lazily via an iterator.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator

from .segmenter import Segment

# Default threshold (in tokens) above which the streaming path is preferred.
STREAMING_THRESHOLD_TOKENS = 100_000


class StreamingChunker:
    """Yield text segments lazily without holding all token positions in memory.

    The algorithm scans the text incrementally, tracking word-boundary
    positions in a bounded deque.  When the deque reaches *max_tokens*
    entries a :class:`Segment` is yielded and the window slides forward
    by ``max_tokens - overlap_tokens`` positions.

    Parameters
    ----------
    max_tokens:
        Maximum number of whitespace-delimited tokens per segment.
    overlap_tokens:
        Number of tokens shared between consecutive segments for
        boundary reconciliation.
    """

    _TOKEN_RE = re.compile(r"\S+")

    def __init__(
        self,
        *,
        max_tokens: int = 4096,
        overlap_tokens: int = 128,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str) -> Iterator[Segment]:
        """Yield :class:`Segment` objects covering the full *text*."""
        # Buffer: each entry is (token_start_char, token_end_char)
        buffer: deque[tuple[int, int]] = deque()
        global_token_idx = 0  # token counter across the whole text
        segment_idx = 0
        segment_start_token = 0

        for match in self._TOKEN_RE.finditer(text):
            buffer.append((match.start(), match.end()))
            global_token_idx += 1

            if len(buffer) == self.max_tokens:
                yield self._emit_segment(
                    text, buffer, segment_idx, segment_start_token
                )
                segment_idx += 1
                # Slide: keep only the last overlap_tokens entries
                advance = self.max_tokens - self.overlap_tokens
                for _ in range(advance):
                    buffer.popleft()
                segment_start_token += advance

        # Emit any remaining tokens in the buffer
        if buffer:
            yield self._emit_segment(
                text, buffer, segment_idx, segment_start_token
            )

    def _emit_segment(
        self,
        text: str,
        buffer: deque[tuple[int, int]],
        segment_idx: int,
        start_token: int,
    ) -> Segment:
        start_char = buffer[0][0]
        end_char = buffer[-1][1]
        return Segment(
            segment_id=f"seg-{segment_idx}",
            start_token=start_token,
            end_token=start_token + len(buffer),
            start_char=start_char,
            end_char=end_char,
            text=text[start_char:end_char],
        )


def estimate_token_count(text: str) -> int:
    """Fast approximation of whitespace-delimited token count.

    Uses byte-length heuristic (avg ~5 chars/token for English) to avoid
    a full regex scan.  For more accurate counts the caller should use
    ``len(re.findall(r'\\S+', text))``.
    """
    if not text:
        return 0
    # Rough estimate: count spaces + 1
    # This is ~10x faster than regex on large strings
    return text.count(" ") + text.count("\n") + text.count("\t") + 1
