"""Regression tests for the upheld findings from the round-16 (certified-path) close."""

from __future__ import annotations

from pii_anon.assurance import (
    AssuranceConfig,
    PipelineAdapter,
    run_assurance_report,
    verify_report_files,
    verify_reproduction,
)
from pii_anon.assurance.adjudication import RecordEval, Substrate
from pii_anon.assurance.assessors import assess_detection
from pii_anon.assurance.claim_strength import ClaimStrength, MeasurementMode, PowerThresholds
from pii_anon.assurance.spans import match_counts, match_type

_THR = PowerThresholds()


# --- #1 common type synonyms reconcile (no phantom-0 from disjoint vocabularies) --
def test_type_synonyms_reconcile() -> None:
    assert match_type("EMAIL_ADDRESS") == match_type("email") == match_type("EMAIL")
    pred = [("EMAIL_ADDRESS", 0, 5), ("PHONE_NUMBER", 6, 12), ("PERSON_NAME", 13, 20)]
    gold = [("EMAIL", 0, 5), ("PHONE", 6, 12), ("PERSON", 13, 20)]
    c = match_counts(pred, gold)
    assert (c.tp, c.fp, c.fn) == (3, 0, 0)  # reconciled, not F1=0


def _substrate(reference, user_pred, n=80):
    recs = tuple(
        RecordEval(record_id=f"r{i}", original_text="x" * 80, language="en", reference=reference,
                   pred_by_system={"sys": user_pred, "pii-anon": reference},
                   transform_by_system={"sys": None, "pii-anon": None})
        for i in range(n)
    )
    return Substrate(recs, ("sys", "pii-anon"), MeasurementMode.LABELED, "gold")


def test_aliased_vocab_is_not_a_phantom_loss() -> None:
    # gold uses pii-anon's native vocab; the user emits short tags at the SAME offsets
    gold = (("EMAIL_ADDRESS", 0, 5), ("PERSON_NAME", 6, 12))
    user = (("EMAIL", 0, 5), ("PERSON", 6, 12))
    sub = _substrate(gold, user)
    results, _ = assess_detection(sub, baseline_system="pii-anon", thresholds=_THR, seed=1, n_resamples=200)
    assert results["sys"].value == 1.0  # reconciled to a perfect match, not 0
    assert results["sys"].metadata.get("type_schema_mismatch") is not True


# --- #1b general safeguard: a non-reconcilable disjoint vocabulary is DEMOTED ------
def test_disjoint_vocab_demoted_to_advisory() -> None:
    gold = (("EMAIL", 0, 5), ("PHONE", 6, 12))
    user = (("WIDGET", 0, 5), ("GADGET", 6, 12))  # correct spans, non-aliased types
    sub = _substrate(gold, user)
    results, _ = assess_detection(sub, baseline_system="pii-anon", thresholds=_THR, seed=1, n_resamples=200)
    res = results["sys"]
    assert res.metadata.get("type_schema_mismatch") is True
    assert res.claim_strength is not ClaimStrength.MEASURED  # never a publishable type-artifact
    assert res.metadata["boundary_f1"] > (res.value or 0.0)  # right spans, wrong labels


# --- end-to-end: short-tag gold vs pii-anon yields no phantom win -----------------
def test_short_tag_gold_no_phantom_win_vs_pii_anon(tmp_path) -> None:
    import re
    rx = re.compile(r"\S+@\S+\.\S+")
    rows = []
    for i in range(80):
        s = f"user{i}@host.example.com"
        text = f"Record {i}: contact {s} now."
        st = text.index(s)
        rows.append({"id": f"r{i}", "text": text, "labels": [["EMAIL", st, st + len(s)]]})  # SHORT tag

    def weak(text):  # detects emails but as short "EMAIL" — same canonical as pii-anon's EMAIL_ADDRESS
        return [("EMAIL", m.start(), m.end()) for m in rx.finditer(text)]

    report = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="acme", detect=weak, transform=lambda t: t,
                                               version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=42, n_resamples=200))
    # pii-anon (EMAIL_ADDRESS) must NOT score 0 against gold "EMAIL" (reconciled)
    assert report.dimensions["detection"]["pii-anon"].value and report.dimensions["detection"]["pii-anon"].value > 0.5


# --- end-to-end: multi-type short-tag gold (PERSON/EMAIL/PHONE) vs an email-only user --
# Closes the disjoint-vocabulary coverage gap: gold uses the conventional short NER tags
# while pii-anon emits its native PERSON_NAME / EMAIL_ADDRESS / PHONE_NUMBER. Reconciliation
# must keep pii-anon's F1 off the phantom-0 floor AND the head-to-head must point the right
# way — the email-only regex is the legitimate loser; pii-anon is NEVER declared a
# significant loser purely because of a type-string schema difference.
def test_short_tag_gold_three_types_pii_anon_not_phantom_loser(tmp_path) -> None:
    import re
    rx = re.compile(r"\S+@\S+\.\S+")
    names = ("Maria Gomez", "John Carter", "Aisha Khan", "Liu Wei", "Sara Olsen", "Tom Reed")
    rows = []
    for i in range(80):
        name, email, phone = names[i % len(names)], f"user{i}@host.example.com", f"(555) {200 + i:03d}-{1000 + i:04d}"
        text = f"Patient {name} can be reached at {email} or by phone {phone}."
        labels = [  # SHORT NER tags, disjoint from pii-anon's native PERSON_NAME/EMAIL_ADDRESS/PHONE_NUMBER
            ["PERSON", text.index(name), text.index(name) + len(name)],
            ["EMAIL", text.index(email), text.index(email) + len(email)],
            ["PHONE", text.index(phone), text.index(phone) + len(phone)],
        ]
        rows.append({"id": f"r{i}", "text": text, "labels": labels})

    def email_only(text):  # a regex pipeline that only finds emails — genuinely weaker than pii-anon
        return [("EMAIL", m.start(), m.end()) for m in rx.finditer(text)]

    report = run_assurance_report(AssuranceConfig(
        records=rows, pipeline=PipelineAdapter(name="acme", detect=email_only, transform=lambda t: t,
                                               version="1", deterministic=True),
        dimensions=("detection",), outputs=("json",), out_dir=str(tmp_path), seed=42, n_resamples=200))

    pii = report.dimensions["detection"]["pii-anon"]
    assert pii.value is not None and pii.value > 0.5          # reconciled — NOT the phantom-0 type artifact
    assert pii.metadata.get("type_schema_mismatch") is not True  # pii-anon's own labels reconcile to gold

    # exactly one head-to-head (the non-baseline user system); pii-anon must not be the loser
    (user_key,) = tuple(report.detection_comparisons)
    comp = report.detection_comparisons[user_key]
    assert comp["delta"] < 0                                  # user worse than pii-anon (pii-anon NOT the loser)
    assert not (comp["significant_holm"] and comp["delta"] > 0)  # no fabricated 'user beats pii-anon' win


# --- #2 verify CLI/lib fails closed (no raw traceback) on malformed input ---------
def test_verify_malformed_input_fails_closed(tmp_path) -> None:
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 50000 + "]" * 50000, encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    assert verify_report_files(deep, empty).reproduced is False              # RecursionError -> closed
    assert verify_reproduction({"provenance": "notadict"}, {}).reproduced is False  # non-dict prov
    assert verify_report_files(tmp_path / "nope.json", empty).reproduced is False   # missing file
    from pii_anon.assurance.__main__ import main
    assert main(["verify", "--report", str(deep), "--against", str(empty)]) == 1


# --- #3 per-entity-type breakdown is capped -------------------------------------
def test_breakdown_row_cap() -> None:
    reference = tuple((f"TYPE{i}", i * 2, i * 2 + 1) for i in range(250))  # 250 distinct types
    sub = _substrate(reference, reference, n=2)
    results, _ = assess_detection(sub, baseline_system="pii-anon", thresholds=_THR, seed=1, n_resamples=200)
    md = results["sys"].metadata
    assert len(md["per_entity_type"]) <= 200
    assert md["per_entity_type_total"] >= 250
