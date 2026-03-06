from pii_anon.segmentation import Segmenter


def test_segmenter_splits_large_text() -> None:
    segmenter = Segmenter()
    text = " ".join(["x"] * 120)
    segments = segmenter.segment(text, max_tokens=50, overlap_tokens=10)
    assert len(segments) >= 3
    assert segments[0].segment_id == "seg-0"
