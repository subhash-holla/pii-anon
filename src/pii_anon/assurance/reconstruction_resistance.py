"""sp7 Phase C — LLM-reconstruction-resistance report.

Produces a reproducible, shareable, SCIENTIFICALLY HONEST measurement of how
resistant pii-anon's masking is to reconstruction / re-identification. The
report claims a **measured BOUND against a specified, disclosed adversary
class** — never "no LLM can reconstruct any PII" (an unprovable negative over an
unbounded adversary class).

Three measured axes, each with its own honest caveat:

1. **Verbatim leakage** — the fraction of ground-truth PII values that survive
   VERBATIM in the masked output (a direct reconstruction bound: anything left
   verbatim is trivially reconstructable). Wilson 95% CI on the integer count.

2. **Tier-3 re-identification** (RRS/QIC/BSL) via
   ``attacks.reid_tier3.RepresentativeTier3ReidAttack`` — the success rate of a
   deterministic representative adversary re-linking a masked record to its
   identity among the candidate set, reported for masked output AND an unmasked
   baseline (the delta is the masking's protection). Wilson 95% CI.

3. **Membership inference** (LiRA-shape) via
   ``attacks.mia.RepresentativeMiaAttack`` — TPR at low FPR that a masked
   record's residual signal betrays its membership. Representative stand-in;
   the real LiRA@128 is a labelled SWITCH-POINT(DATA).

Honesty invariants (do not remove):
- the representative adversaries are labelled ``provisional_status:
  AGENT_SIMULATED``; a real-LLM-API adversary run is a separate, budget-gated,
  distinctly-labelled row.
- every rate carries its integer counts, its seed, and the NON-STRIPPABLE
  anti-anonymity caveat (NFR-016).
- ``observed_signals`` on a target are RE-EXTRACTED from the masked text, never
  copied from gold (the reidx-01 de-circularization discipline).
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pii_anon.eval_framework.attacks.mia import (
    RepresentativeMiaAttack,
    score_mia_attack,
)
from pii_anon.eval_framework.attacks.reid import (
    ReidPersona,
    ReidTarget,
    score_reid_attack,
)
from pii_anon.eval_framework.attacks.reid_tier3 import (
    RepresentativeTier3ReidAttack,
    wilson_interval,
)

#: The non-strippable anti-anonymity caveat (NFR-016) — carried on every report.
CAVEAT = (
    "MEASURED BOUND, NOT A GUARANTEE OF IMPOSSIBILITY. This report measures "
    "resistance against the DISCLOSED representative adversary class at the "
    "stated power; it does NOT prove that no adversary can ever reconstruct any "
    "PII. Representative adversaries are AGENT_SIMULATED deterministic "
    "stand-ins, not real LLM-API adversaries. Absence of leakage in this sample "
    "is not a guarantee for out-of-sample inputs."
)

MaskFn = Callable[[str], str]
SignalFn = Callable[[str], tuple[str, ...]]

_SIGNAL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'@._-]{2,}")


@dataclass(frozen=True)
class CorpusRecord:
    """One corpus record with its ground-truth PII values (the quasi-identifiers
    an adversary would link against). ``pii_types``, when supplied, is a
    parallel tuple of entity types enabling the per-type leakage breakdown."""

    record_id: str
    text: str
    pii_values: tuple[str, ...]
    pii_types: tuple[str, ...] = ()


def default_signals(text: str) -> tuple[str, ...]:
    """Surviving linkable tokens an adversary re-extracts from (masked) text.

    Deterministic; length>=3 alnum-ish tokens. Placeholder/mask tokens (all
    upper-case bracketed tags like ``[PERSON_NAME]`` / ``<REDACTED>``) are
    excluded — a mask token is not a linkable signal."""
    out: list[str] = []
    for m in _SIGNAL_RE.finditer(text):
        tok = m.group(0)
        # a mask placeholder is not a signal (it carries no identity)
        lo, hi = m.start(), m.end()
        before = text[lo - 1] if lo > 0 else ""
        after = text[hi] if hi < len(text) else ""
        if before in "[<{" and after in "]>}":
            continue
        out.append(tok)
    return tuple(out)


def measure_verbatim_leakage(
    records: Sequence[CorpusRecord], mask_fn: MaskFn
) -> dict[str, Any]:
    """Count ground-truth PII values that survive VERBATIM in the masked output.

    The most direct reconstruction bound: a value left verbatim is trivially
    reconstructable. Returns integer counts + a Wilson 95% CI + examples."""
    total = 0
    leaked = 0
    examples: list[dict[str, str]] = []
    per_type: dict[str, list[int]] = {}  # type -> [total, leaked]
    for rec in records:
        masked = mask_fn(rec.text)
        types = rec.pii_types if len(rec.pii_types) == len(rec.pii_values) else ()
        for i, val in enumerate(rec.pii_values):
            if not val or len(val) < 2:
                continue
            total += 1
            etype = types[i] if types else "UNKNOWN"
            bucket = per_type.setdefault(etype, [0, 0])
            bucket[0] += 1
            if val in masked:
                leaked += 1
                bucket[1] += 1
                if len(examples) < 25:
                    examples.append({"record_id": rec.record_id, "value": val, "type": etype})
    rate = leaked / total if total else 0.0
    lo, hi = wilson_interval(leaked, total) if total else (0.0, 0.0)
    per_type_out = {
        t: {
            "total": c[0],
            "leaked": c[1],
            "leak_rate": (c[1] / c[0] if c[0] else 0.0),
            "wilson95": list(wilson_interval(c[1], c[0])) if c[0] else [0.0, 0.0],
        }
        for t, c in sorted(per_type.items(), key=lambda kv: -kv[1][1])
    }
    return {
        "total_pii_values": total,
        "leaked_verbatim": leaked,
        "leak_rate": rate,
        "wilson95": [lo, hi],
        "per_type": per_type_out,
        "examples": examples,
    }


def assemble_paired_set(
    records: Sequence[CorpusRecord], mask_fn: MaskFn, signal_fn: SignalFn
) -> tuple[list[ReidPersona], list[ReidTarget]]:
    """Build the (personas, masked-targets) paired set — the load-bearing
    corpus->attack-substrate adapter. ``target_id == persona_id`` is the
    ground-truth link; ``observed_signals`` are RE-EXTRACTED from the masked
    text (reidx-01)."""
    personas: list[ReidPersona] = []
    targets: list[ReidTarget] = []
    for rec in records:
        personas.append(
            ReidPersona(
                persona_id=rec.record_id,
                quasi_identifiers=tuple(rec.pii_values),
                source_text=rec.text,
            )
        )
        masked = mask_fn(rec.text)
        targets.append(
            ReidTarget(
                target_id=rec.record_id,
                anonymized_text=masked,
                observed_signals=signal_fn(masked),
            )
        )
    return personas, targets


def _reid_axis(
    records: Sequence[CorpusRecord], mask_fn: MaskFn, signal_fn: SignalFn
) -> dict[str, Any]:
    personas, targets_masked = assemble_paired_set(records, mask_fn, signal_fn)
    targets_unmasked = [
        ReidTarget(
            target_id=r.record_id,
            anonymized_text=r.text,
            observed_signals=signal_fn(r.text),
        )
        for r in records
    ]
    atk = RepresentativeTier3ReidAttack()
    n = len(personas)
    m_masked = score_reid_attack(
        atk.attack(targets_masked, personas, n),
        targets_masked,
        adversary_id=atk.adversary_id,
        deterministic=True,
        candidate_set_size=n,
    )
    m_unmasked = score_reid_attack(
        atk.attack(targets_unmasked, personas, n),
        targets_unmasked,
        adversary_id=atk.adversary_id,
        deterministic=True,
        candidate_set_size=n,
    )
    correct_masked = int(round(m_masked.reid_recall * n))
    lo, hi = wilson_interval(correct_masked, n) if n else (0.0, 0.0)
    return {
        "adversary": atk.adversary_id,
        "provisional_status": "AGENT_SIMULATED",
        "candidate_set_size": n,
        "masked": {
            "reid_success_rate": m_masked.reid_success_rate,
            "reid_recall": m_masked.reid_recall,
            "reid_precision": m_masked.reid_precision,
            "correct": correct_masked,
            "wilson95_recall": [lo, hi],
        },
        "unmasked_baseline": {
            "reid_success_rate": m_unmasked.reid_success_rate,
            "reid_recall": m_unmasked.reid_recall,
        },
        "protection_delta": m_unmasked.reid_recall - m_masked.reid_recall,
    }


def _mia_axis(
    records: Sequence[CorpusRecord], mask_fn: MaskFn, signal_fn: SignalFn
) -> dict[str, Any]:
    """Representative membership-inference axis. observed_loss proxy: a masked
    record that retains MORE of its own PII signal is more member-like (lower
    loss). Members = the corpus records; non-members = the same records with all
    signal stripped (the null floor). Representative stand-in only."""
    members: list[Any] = []
    non_members: list[Any] = []
    gold: list[bool] = []

    @dataclass(frozen=True)
    class _Rec:
        record_id: str
        observed_loss: float

    for rec in records:
        masked = mask_fn(rec.text)
        surviving = set(signal_fn(masked))
        pii_tokens = {t for v in rec.pii_values for t in signal_fn(v)}
        retained = len(surviving & pii_tokens)
        # lower loss == more member-like; retaining own PII lowers the loss.
        members.append(_Rec(rec.record_id, -float(retained)))
        gold.append(True)
        non_members.append(_Rec(rec.record_id + "#null", 0.0))
        gold.append(False)
    atk = RepresentativeMiaAttack()
    all_recs = members + non_members
    scores = atk.membership_scores(all_recs)
    report = score_mia_attack(
        scores, gold, adversary_id=atk.adversary_id, deterministic=True
    )
    return {
        "adversary": atk.adversary_id,
        "provisional_status": "AGENT_SIMULATED",
        "tpr_at_1e_3": report.tpr_at_1e_3,
        "tpr_at_1e_2": report.tpr_at_1e_2,
        "auc": report.auc,
        "n_members": report.n_members,
        "wilson95_tpr": [report.ci_low, report.ci_high],
        "note": "representative stand-in; real LiRA@128 is SWITCH-POINT(DATA)",
    }


def reconstruction_resistance_report(
    records: Sequence[CorpusRecord],
    mask_fn: MaskFn,
    *,
    seed: int,
    mask_label: str = "pii-anon",
    corpus_label: str = "corpus",
    signal_fn: SignalFn = default_signals,
) -> dict[str, Any]:
    """The reproducible reconstruction-resistance report (measured bounds).

    Deterministic given ``records`` + ``mask_fn`` + ``seed`` (the representative
    adversaries carry no RNG; ``seed`` is recorded for the corpus sampling the
    caller did). Returns a JSON-serialisable dict."""
    return {
        "report": "llm-reconstruction-resistance",
        "schema_version": "1.0",
        "seed": seed,
        "corpus_label": corpus_label,
        "mask_label": mask_label,
        "n_records": len(records),
        "verbatim_leakage": measure_verbatim_leakage(records, mask_fn),
        "reidentification_tier3": _reid_axis(records, mask_fn, signal_fn),
        "membership_inference": _mia_axis(records, mask_fn, signal_fn),
        "caveat": CAVEAT,
    }


def _pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def render_html(report: dict[str, Any]) -> str:
    """Render a report dict as a self-contained, shareable HTML page."""
    vl = report["verbatim_leakage"]
    rd = report["reidentification_tier3"]
    mi = report["membership_inference"]
    rd_m, rd_b = rd["masked"], rd["unmasked_baseline"]
    rows = "".join(
        f"<tr><td>{r['record_id']}</td><td>{r.get('type', '')}</td><td><code>{r['value']}</code></td></tr>"
        for r in vl["examples"]
    ) or "<tr><td colspan=3><em>none — zero verbatim leakage in this sample</em></td></tr>"
    pt = vl.get("per_type", {})
    pt_rows = "".join(
        f"<tr><td>{t}</td><td>{d['leaked']}/{d['total']}</td>"
        f"<td class=\"{'bad' if d['leak_rate'] > 0 else 'good'}\">{_pct(d['leak_rate'])}</td>"
        f"<td class=ci>[{_pct(d['wilson95'][0])}, {_pct(d['wilson95'][1])}]</td></tr>"
        for t, d in pt.items()
    )
    pt_table = (
        f"<details open><summary>Per-type leakage breakdown ({len(pt)} types)</summary>"
        f"<table><tr><th>type</th><th>leaked/total</th><th>rate</th><th>Wilson 95%</th></tr>"
        f"{pt_rows}</table></details>"
        if pt else ""
    )
    return f"""<!doctype html><meta charset=utf-8>
<title>LLM-Reconstruction-Resistance Report — {report['mask_label']}</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
 h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:1.8rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
 .metric{{display:inline-block;margin:.4rem 1.2rem .4rem 0}}
 .big{{font-size:1.6rem;font-weight:700}} .ci{{color:#666;font-size:.85rem}}
 .good{{color:#0a7d33}} .bad{{color:#b00020}}
 table{{border-collapse:collapse;width:100%;margin:.6rem 0}} td,th{{border:1px solid #ddd;padding:.3rem .5rem;text-align:left}}
 .caveat{{background:#fff8e1;border:1px solid #ffe082;padding:.8rem;border-radius:6px;font-size:.9rem;margin-top:1.5rem}}
 code{{background:#f5f5f5;padding:0 .2rem;border-radius:3px}}
</style>
<h1>LLM-Reconstruction-Resistance Report</h1>
<p><strong>Masking:</strong> {report['mask_label']} &middot; <strong>Corpus:</strong> {report['corpus_label']}
 ({report['n_records']} records) &middot; <strong>Seed:</strong> {report['seed']} &middot;
 <strong>Schema:</strong> {report['schema_version']}</p>

<h2>1. Verbatim leakage <span class=ci>(direct reconstruction bound)</span></h2>
<div class=metric><div class="big {'good' if vl['leaked_verbatim']==0 else 'bad'}">{_pct(vl['leak_rate'])}</div>
 of {vl['total_pii_values']} PII values survive verbatim
 <div class=ci>{vl['leaked_verbatim']} leaked &middot; Wilson 95% [{_pct(vl['wilson95'][0])}, {_pct(vl['wilson95'][1])}]</div></div>
{pt_table}
<details><summary>Leaked examples ({len(vl['examples'])})</summary>
 <table><tr><th>record</th><th>type</th><th>value surviving verbatim</th></tr>{rows}</table></details>

<h2>2. Tier-3 re-identification <span class=ci>(RRS/QIC/BSL — {rd['provisional_status']})</span></h2>
<div class=metric><div class="big {'good' if rd_m['reid_recall']<0.1 else 'bad'}">{_pct(rd_m['reid_recall'])}</div>
 masked re-id recall <div class=ci>{rd_m['correct']}/{rd['candidate_set_size']} &middot;
 Wilson 95% [{_pct(rd_m['wilson95_recall'][0])}, {_pct(rd_m['wilson95_recall'][1])}]</div></div>
<div class=metric><div class="big">{_pct(rd_b['reid_recall'])}</div> unmasked baseline
 <div class=ci>protection &Delta; {_pct(rd['protection_delta'])}</div></div>
<p class=ci>Adversary: <code>{rd['adversary']}</code>, candidate set size {rd['candidate_set_size']}.</p>

<h2>3. Membership inference <span class=ci>(LiRA-shape — {mi['provisional_status']})</span></h2>
<div class=metric><div class=big>{_pct(mi['tpr_at_1e_2'])}</div> TPR@FPR=1e-2
 <div class=ci>Wilson 95% [{_pct(mi['wilson95_tpr'][0])}, {_pct(mi['wilson95_tpr'][1])}]</div></div>
<div class=metric><div class=big>{_pct(mi['tpr_at_1e_3'])}</div> TPR@FPR=1e-3</div>
<div class=metric><div class=big>{mi['auc']:.3f}</div> AUC ({mi['n_members']} members)</div>
<p class=ci>{mi['note']}</p>

<div class=caveat><strong>Scope &amp; honesty.</strong> {report['caveat']}</div>
"""
