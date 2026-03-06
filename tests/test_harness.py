from pii_anon.harness import StreamSynthesizer, reconstruction_attack_score


def test_stream_synthesizer_modes() -> None:
    synth = StreamSynthesizer()
    events = [{"text": "abcdef"}, {"text": "hello"}]

    split = synth.run(events, "split")
    assert len(split) == 4
    assert split[0]["disturbance"] == "split-a"

    reorder = synth.run(events, "reorder")
    assert reorder[0]["text"] == "hello"

    duplicate = synth.run(events, "duplicate")
    assert len(duplicate) == 4
    assert duplicate[1]["disturbance"] == "duplicate"

    truncate = synth.run(events, "truncate")
    assert len(truncate[0]["text"]) < len(events[0]["text"])
    assert truncate[0]["disturbance"] == "truncate"

    passthrough = synth.run(events, "unknown")
    assert passthrough == events


def test_reconstruction_attack_score() -> None:
    out = reconstruction_attack_score(
        masked_texts=["tok_a in output", "safe"],
        known_tokens=["tok_a", "tok_b"],
    )
    assert out["attack"] == "reconstruction_rank_style"
    assert out["recovered"] == 1
    assert out["total"] == 2
