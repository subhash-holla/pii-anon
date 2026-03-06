"""Tests for pii_anon.segmentation.chunker — streaming chunker for large text."""

from __future__ import annotations

import pytest

from pii_anon.segmentation.chunker import StreamingChunker, estimate_token_count


class TestStreamingChunkerInit:
    def test_valid_defaults(self) -> None:
        chunker = StreamingChunker()
        assert chunker.max_tokens == 4096
        assert chunker.overlap_tokens == 128

    def test_custom_params(self) -> None:
        chunker = StreamingChunker(max_tokens=100, overlap_tokens=10)
        assert chunker.max_tokens == 100
        assert chunker.overlap_tokens == 10

    def test_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            StreamingChunker(max_tokens=0)

    def test_invalid_overlap_negative(self) -> None:
        with pytest.raises(ValueError, match="overlap_tokens must be >= 0"):
            StreamingChunker(overlap_tokens=-1)

    def test_overlap_gte_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="overlap_tokens must be < max_tokens"):
            StreamingChunker(max_tokens=10, overlap_tokens=10)


class TestStreamingChunkerChunk:
    def test_empty_text(self) -> None:
        chunker = StreamingChunker(max_tokens=5, overlap_tokens=1)
        segments = list(chunker.chunk(""))
        assert len(segments) == 0

    def test_text_within_single_chunk(self) -> None:
        chunker = StreamingChunker(max_tokens=10, overlap_tokens=2)
        segments = list(chunker.chunk("one two three"))
        assert len(segments) == 1
        assert segments[0].text == "one two three"
        assert segments[0].start_token == 0
        assert segments[0].end_token == 3
        assert segments[0].segment_id == "seg-0"

    def test_exact_max_tokens(self) -> None:
        chunker = StreamingChunker(max_tokens=3, overlap_tokens=0)
        text = "a b c"
        segments = list(chunker.chunk(text))
        assert len(segments) == 1
        assert segments[0].text == "a b c"

    def test_two_chunks_no_overlap(self) -> None:
        chunker = StreamingChunker(max_tokens=3, overlap_tokens=0)
        text = "a b c d e"
        segments = list(chunker.chunk(text))
        assert len(segments) == 2
        assert segments[0].text == "a b c"
        assert segments[1].text == "d e"
        assert segments[0].start_token == 0
        assert segments[0].end_token == 3
        assert segments[1].start_token == 3

    def test_chunks_with_overlap(self) -> None:
        chunker = StreamingChunker(max_tokens=4, overlap_tokens=2)
        text = "a b c d e f"
        segments = list(chunker.chunk(text))
        # max_tokens=4, overlap=2 → step=2
        # Chunk 0: tokens 0-3 ("a b c d")
        # Chunk 1: tokens 2-5 ("c d e f")
        # Chunk 2: tokens 4-5 ("e f") — residual
        assert len(segments) == 3
        assert segments[0].text == "a b c d"
        assert segments[1].text == "c d e f"
        assert segments[2].text == "e f"

    def test_overlap_content_matches(self) -> None:
        chunker = StreamingChunker(max_tokens=4, overlap_tokens=1)
        text = "w1 w2 w3 w4 w5 w6 w7 w8"
        segments = list(chunker.chunk(text))
        # 8 tokens, step=3: [0-3], [3-6], [6-7] → 3 segments
        assert len(segments) >= 2
        # The last token of chunk 0 should be the first token of chunk 1
        last_token_chunk0 = segments[0].text.split()[-1]
        first_token_chunk1 = segments[1].text.split()[0]
        assert last_token_chunk0 == first_token_chunk1

    def test_single_token(self) -> None:
        chunker = StreamingChunker(max_tokens=5, overlap_tokens=1)
        segments = list(chunker.chunk("hello"))
        assert len(segments) == 1
        assert segments[0].text == "hello"

    def test_large_text_many_chunks(self) -> None:
        """Verify chunker handles moderately large text (10K tokens)."""
        words = [f"word{i}" for i in range(10_000)]
        text = " ".join(words)
        chunker = StreamingChunker(max_tokens=500, overlap_tokens=50)
        segments = list(chunker.chunk(text))

        # 10K tokens / step(450) ≈ 23 chunks
        assert len(segments) >= 20
        assert len(segments) <= 25

        # Verify no gaps: first chunk starts at 0, last chunk covers end
        assert segments[0].start_char == 0
        assert segments[-1].end_char == len(text)

        # Verify segment IDs are sequential
        for i, seg in enumerate(segments):
            assert seg.segment_id == f"seg-{i}"

    def test_segment_char_boundaries_are_correct(self) -> None:
        chunker = StreamingChunker(max_tokens=2, overlap_tokens=0)
        text = "hello world foo bar baz"
        segments = list(chunker.chunk(text))

        for seg in segments:
            assert text[seg.start_char:seg.end_char] == seg.text

    def test_whitespace_preservation(self) -> None:
        chunker = StreamingChunker(max_tokens=3, overlap_tokens=0)
        text = "a   b\tc\n\nd   e   f"
        segments = list(chunker.chunk(text))
        # 6 tokens: a, b, c, d, e, f
        assert len(segments) == 2
        # Each segment's text should be a valid slice of the original
        for seg in segments:
            assert text[seg.start_char:seg.end_char] == seg.text


class TestEstimateTokenCount:
    def test_empty(self) -> None:
        assert estimate_token_count("") == 0

    def test_single_word(self) -> None:
        assert estimate_token_count("hello") == 1

    def test_typical_sentence(self) -> None:
        count = estimate_token_count("The quick brown fox jumps")
        assert count == 5  # 4 spaces + 1

    def test_multiline(self) -> None:
        count = estimate_token_count("line1\nline2\nline3")
        assert count >= 3

    def test_tabs(self) -> None:
        count = estimate_token_count("col1\tcol2\tcol3")
        assert count >= 3
