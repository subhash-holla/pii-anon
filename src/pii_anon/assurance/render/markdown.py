"""Technical Markdown renderer."""

from __future__ import annotations

import html
from typing import Any

_BADGE = {
    "measured": "✅ MEASURED",
    "advisory": "⚠️ ADVISORY",
    "not_assessable": "⬚ NOT ASSESSABLE",
}


def _cell(value: Any) -> str:
    """Encode an untrusted string for a Markdown table cell — neutralizes HTML
    (stored-XSS in rendered MD/HTML) and escapes the column separator / newlines
    (table corruption). The renderer's output-encoding concern, distinct from the
    numeric-integrity and PII-egress gates."""
    return html.escape(str(value)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _fmt(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _ci(ci: list[float] | None) -> str:
    if not ci:
        return "—"
    return f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _badge(cs: str) -> str:
    return _BADGE.get(cs, cs)


def render_markdown(report_dict: dict[str, Any]) -> dict[str, str]:
    d = report_dict
    ds = d["dataset"]
    lines: list[str] = []
    lines.append("# PII Handling Assurance Report\n")
    lines.append(f"**Dataset:** `{_cell(ds['name'])}` · **records:** {ds['n_records']} · "
                 f"**mode:** {ds['mode']} (reference: {ds['reference_kind']})\n")
    lines.append(f"**Dataset fingerprint:** `{ds['fingerprint'][:32]}…`\n")
    systems = ", ".join(_cell(s) for s in d["systems"])
    lines.append(f"**Generated:** {d.get('generated_at', '—')} · **systems:** "
                 f"{systems} (baseline: {_cell(d['baseline'])})\n")

    # claim-strength legend
    lines.append("\n> **Claim strength** — "
                 "✅ MEASURED = computed vs gold labels, publishable · "
                 "⚠️ ADVISORY = hedged (silver / under-powered / un-certified harness) · "
                 "⬚ NOT ASSESSABLE = required input absent (never scored as 0).\n")

    # dimensions
    for dim, by_system in d["dimensions"].items():
        lines.append(f"\n## Dimension: {dim}\n")
        lines.append("| System | Value | 95% CI | Claim strength | Notes |")
        lines.append("|--------|-------|--------|----------------|-------|")
        for sys, r in by_system.items():
            notes_bits = list(r.get("reasons", [])) + list(r.get("caveats", []))
            notes = _cell("; ".join(notes_bits)) if notes_bits else ""
            lines.append(
                f"| {_cell(sys)} | {_fmt(r['value'])} | {_ci(r['confidence_interval'])} | "
                f"{_badge(r['claim_strength'])} | {notes} |"
            )
        # per-entity-type breakdown for detection
        for sys, r in by_system.items():
            bd = r.get("metadata", {}).get("per_entity_type")
            if bd:
                lines.append(f"\n<details><summary>{_cell(sys)}: per-entity-type detection</summary>\n")
                lines.append("\n| Entity type | Precision | Recall | F1 | Support |")
                lines.append("|-------------|-----------|--------|-----|---------|")
                for etype, m in sorted(bd.items()):
                    lines.append(
                        f"| {_cell(etype)} | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
                        f"{_fmt(m.get('f1'))} | {int(m.get('support', 0))} |"
                    )
                lines.append("\n</details>\n")

    # head-to-head comparison (labeled)
    comps = d.get("detection_comparisons") or {}
    if comps:
        lines.append("\n## Head-to-head (detection F1, vs baseline)\n")
        lines.append("| System | Claim strength | Δ F1 vs baseline | Δ CI | p (raw) | significant (Holm) | effect size | n |")
        lines.append("|--------|----------------|------------------|------|---------|--------------------|-------------|---|")
        for sys, c in comps.items():
            cs = c.get("claim_strength", "advisory")
            if c.get("significant_holm"):
                sig = "yes"
            elif cs != "measured":
                sig = "not publishable (advisory)"
            else:
                sig = "no"
            es = c.get("effect_size")
            delta = c.get("delta")
            es_str = "maximal" if (es is None and delta not in (None, 0, 0.0)) else _fmt(es)
            lines.append(
                f"| {_cell(sys)} | {_badge(cs)} | {_fmt(delta)} | {_ci(c.get('ci'))} | "
                f"{_fmt(c.get('p_value'))} | {sig} | {es_str} | {int(c.get('n', 0))} |"
            )
        lines.append("\n_A head-to-head verdict is only as strong as its weaker system: it is "
                     "publishable (significant = yes) only when BOTH systems are MEASURED. "
                     "Family-wise error controlled with Holm-Bonferroni; effect size reported "
                     'because at large N a trivial Δ becomes "significant"._\n')

    # silver agreement
    ag = d.get("agreement")
    if ag:
        sa, sb = _cell(ag["system_a"]), _cell(ag["system_b"])
        ordering_u = ", ".join(_cell(x) for x in ag["ordering_union"])
        ordering_i = ", ".join(_cell(x) for x in ag["ordering_intersection"])
        lines.append("\n## Silver-mode agreement (advisory — no publishable winner)\n")
        lines.append(f"- Span Jaccard agreement ({sa} vs {sb}): **{_fmt(ag['span_jaccard'])}**")
        lines.append(f"- Character-level Cohen's κ (chance-adjusted): **{_fmt(ag['char_kappa'])}**")
        lines.append(f"- Disagreement: only {sa}={ag['only_a']}, only {sb}={ag['only_b']}, both={ag['both']}")
        robust = "robust" if ag["reference_robust"] else "**NOT robust** (ordering flips union↔intersection)"
        lines.append(f"- Reference sensitivity: {robust} (union→[{ordering_u}], intersection→[{ordering_i}])")

    # methodology, limitations, provenance
    lines.append("\n## Methodology\n")
    for m in d.get("methodology", []):
        lines.append(f"- {m}")
    lines.append("\n## Limitations\n")
    for limit in d.get("limitations", []):
        lines.append(f"- {limit}")

    prov = d.get("provenance", {})
    lines.append("\n## Provenance & reproducibility\n")
    lines.append(f"- pii-anon version: `{prov.get('pii_anon_version')}` · "
                 f"python `{prov.get('python_version')}` · seed `{prov.get('seed')}`")
    up = prov.get("user_pipeline", {})
    lines.append(f"- User pipeline: `{_cell(up.get('name'))}` v`{_cell(up.get('version'))}` "
                 f"(in-process determinism probe: {up.get('deterministic')} — cross-process "
                 f"reproducibility requires independent `python -m pii_anon.assurance verify`)")
    lines.append(f"- Harness adversarial-close certified: **{prov.get('harness_close_certified')}** "
                 f"({prov.get('harness_close_reference')})")
    stats = d.get("statistics", {})
    pt = stats.get("power_thresholds", {})
    lines.append(
        f"- Power gate (tighten-only floors): support ≥ {pt.get('min_support')}, "
        f"CI half-width ≤ {pt.get('max_ci_halfwidth')}, clusters ≥ {pt.get('min_clusters')} · "
        f"bootstrap B={stats.get('bootstrap_resamples')}, alpha={stats.get('alpha')}, "
        f"comparison family={stats.get('comparison_family_size')}"
    )
    lines.append(f"- {prov.get('reproducibility_claim')}")

    lines.append("\n---\n*Generated by the pii-anon Assurance Report harness.*\n")
    return {"report.md": "\n".join(lines) + "\n"}
