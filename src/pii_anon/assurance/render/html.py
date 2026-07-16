"""Self-contained HTML executive report renderer.

Renders ONLY the already-gated, PII-free ``report_dict`` — no raw text by
construction. Output-encoding-hardened: EVERY interpolated string (system /
dataset names, reasons, versions, group labels, methodology prose) is routed
through :func:`html.escape` (quote=True), so a hostile label cannot inject
markup. No JavaScript; charts are inline ``<svg>`` with numeric, gate-validated
geometry only. The PII-egress gate still runs on the final bytes before write.
"""

from __future__ import annotations

import html
from typing import Any

_BADGE = {
    "measured": ("MEASURED", "#0a7d33", "#e6f4ea"),
    "advisory": ("ADVISORY", "#8a6d00", "#fdf6e3"),
    "not_assessable": ("NOT ASSESSABLE", "#5b6166", "#eef0f2"),
}
_BAR_COLOR = {"measured": "#0a7d33", "advisory": "#c69214", "not_assessable": "#aab0b6"}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def _ci(ci: list[float] | None) -> str:
    return "—" if not ci else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def _badge(cs: str) -> str:
    label, fg, bg = _BADGE.get(cs, (cs, "#5b6166", "#eef0f2"))
    return (
        f'<span class="badge" style="color:{fg};background:{bg}">{_esc(label)}</span>'
    )


def _clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return 0.0 if value < 0 else 1.0 if value > 1 else float(value)


def _bar(value: float | None, cs: str) -> str:
    """A fixed-geometry inline-SVG bar; width is the gate-validated [0,1] value."""
    w = round(_clamp01(value) * 200.0, 2)
    color = _BAR_COLOR.get(cs, "#aab0b6")
    label = "n/a" if value is None else f"{value:.3f}"
    return (
        f'<svg width="240" height="18" role="img" aria-label="{_esc(label)}">'
        f'<rect x="0" y="2" width="200" height="14" fill="#eef0f2" rx="2"/>'
        f'<rect x="0" y="2" width="{w}" height="14" fill="{color}" rx="2"/>'
        f'<text x="208" y="14" font-size="11" fill="#333">{_esc(label)}</text>'
        f"</svg>"
    )


_STYLE = """
:root{--ink:#1a1d20;--muted:#5b6166;--line:#dfe3e6;--bg:#fff}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
  background:var(--bg);margin:0;padding:32px;line-height:1.45}
.wrap{max-width:960px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
.sub{color:var(--muted);font-size:13px;margin:2px 0}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:2px 7px;border-radius:10px;letter-spacing:.02em}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0 4px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
.legend{background:#f7f8f9;border:1px solid var(--line);border-radius:6px;padding:10px 14px;font-size:12.5px;color:var(--muted)}
.note{font-size:12px;color:var(--muted)}
code{background:#f2f3f5;padding:1px 4px;border-radius:3px;font-size:12px}
.foot{margin-top:28px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:10px}
"""


def _dimension_table(dims: dict[str, Any]) -> str:
    rows: list[str] = []
    for dim, by_system in dims.items():
        for sys, r in by_system.items():
            cs = r.get("claim_strength", "not_assessable")
            notes = "; ".join(list(r.get("reasons", [])) + list(r.get("caveats", [])))
            rows.append(
                "<tr>"
                f"<td>{_esc(dim)}</td><td>{_esc(sys)}</td>"
                f"<td>{_bar(r.get('value'), cs)}</td>"
                f"<td>{_ci(r.get('confidence_interval'))}</td>"
                f"<td>{_badge(cs)}</td>"
                f'<td class="note">{_esc(notes)}</td>'
                "</tr>"
            )
    return (
        "<table><thead><tr><th>Dimension</th><th>System</th><th>Score</th>"
        "<th>95% CI</th><th>Claim strength</th><th>Notes</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _comparisons(comps: dict[str, Any]) -> str:
    if not comps:
        return ""
    rows: list[str] = []
    for sys, c in comps.items():
        cs = c.get("claim_strength", "advisory")
        if c.get("significant_holm"):
            sig = "yes"
        elif cs != "measured":
            sig = "not publishable (advisory)"
        else:
            sig = "no"
        rows.append(
            "<tr>"
            f"<td>{_esc(sys)}</td><td>{_badge(cs)}</td><td>{_fmt(c.get('delta'))}</td>"
            f"<td>{_ci(c.get('ci'))}</td><td>{_fmt(c.get('p_value'))}</td>"
            f"<td>{_esc(sig)}</td><td>{_fmt(c.get('effect_size'))}</td><td>{int(c.get('n', 0))}</td>"
            "</tr>"
        )
    return (
        "<h2>Head-to-head (detection F1, vs baseline)</h2>"
        "<table><thead><tr><th>System</th><th>Claim strength</th><th>Δ F1</th><th>Δ CI</th>"
        "<th>p (raw)</th><th>significant (Holm)</th><th>effect size</th><th>n</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        '<p class="note">A head-to-head verdict is publishable (significant = yes) only when '
        "BOTH systems are MEASURED; family-wise error is Holm-Bonferroni controlled, and effect "
        "size is reported because at large N a trivial Δ becomes &ldquo;significant&rdquo;.</p>"
    )


def render_html(report_dict: dict[str, Any]) -> dict[str, str]:
    d = report_dict
    ds = d["dataset"]
    prov = d.get("provenance", {})
    up = prov.get("user_pipeline", {})
    stats = d.get("statistics", {})

    parts: list[str] = []
    parts.append("<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">")
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>PII Handling Assurance Report — {_esc(ds['name'])}</title>")
    parts.append(f"<style>{_STYLE}</style></head><body><div class=\"wrap\">")

    parts.append("<h1>PII Handling Assurance Report</h1>")
    parts.append(
        f'<p class="sub"><strong>Dataset:</strong> <code>{_esc(ds["name"])}</code> · '
        f"<strong>records:</strong> {int(ds['n_records'])} · "
        f"<strong>mode:</strong> {_esc(ds['mode'])} (reference: {_esc(ds['reference_kind'])})</p>"
    )
    parts.append(f'<p class="sub"><strong>Fingerprint:</strong> <code>{_esc(ds["fingerprint"][:32])}…</code></p>')
    parts.append(
        f'<p class="sub"><strong>Generated:</strong> {_esc(d.get("generated_at", "—"))} · '
        f"<strong>systems:</strong> {_esc(', '.join(d['systems']))} "
        f"(baseline: {_esc(d['baseline'])})</p>"
    )

    parts.append(
        '<div class="legend">'
        "<strong>Claim strength</strong> — "
        f"{_badge('measured')} computed vs gold labels, publishable · "
        f"{_badge('advisory')} hedged (silver / under-powered / stand-in / capability axis) · "
        f"{_badge('not_assessable')} required input absent (never scored as 0)."
        "</div>"
    )

    parts.append("<h2>Dimension scorecard</h2>")
    parts.append(_dimension_table(d.get("dimensions", {})))
    parts.append(_comparisons(d.get("detection_comparisons") or {}))

    ag = d.get("agreement")
    if ag:
        parts.append("<h2>Silver-mode agreement (advisory — no publishable winner)</h2>")
        parts.append(
            "<table><tbody>"
            f"<tr><th>Span Jaccard</th><td>{_fmt(ag.get('span_jaccard'))}</td></tr>"
            f"<tr><th>Character κ (chance-adjusted)</th><td>{_fmt(ag.get('char_kappa'))}</td></tr>"
            f"<tr><th>Disagreement</th><td>only {_esc(ag['system_a'])}={int(ag['only_a'])}, "
            f"only {_esc(ag['system_b'])}={int(ag['only_b'])}, both={int(ag['both'])}</td></tr>"
            f"<tr><th>Reference robust</th><td>{_esc(ag.get('reference_robust'))}</td></tr>"
            "</tbody></table>"
        )

    parts.append("<h2>Methodology</h2><ul>")
    for m in d.get("methodology", []):
        parts.append(f"<li class=\"note\">{_esc(m)}</li>")
    parts.append("</ul>")

    parts.append("<h2>Limitations</h2><ul>")
    for lim in d.get("limitations", []):
        parts.append(f"<li class=\"note\">{_esc(lim)}</li>")
    parts.append("</ul>")

    parts.append("<h2>Provenance &amp; reproducibility</h2>")
    parts.append(
        '<table><tbody>'
        f"<tr><th>pii-anon version</th><td><code>{_esc(prov.get('pii_anon_version'))}</code></td></tr>"
        f"<tr><th>python</th><td><code>{_esc(prov.get('python_version'))}</code> · seed "
        f"<code>{_esc(prov.get('seed'))}</code></td></tr>"
        f"<tr><th>user pipeline</th><td><code>{_esc(up.get('name'))}</code> "
        f"v<code>{_esc(up.get('version'))}</code> (determinism probe: {_esc(up.get('deterministic'))})</td></tr>"
        f"<tr><th>harness close certified</th><td>{_esc(prov.get('harness_close_certified'))}</td></tr>"
        f"<tr><th>bootstrap</th><td>B={int(stats.get('bootstrap_resamples', 0))}, "
        f"alpha={_esc(stats.get('alpha'))}, family={int(stats.get('comparison_family_size', 0))}</td></tr>"
        "</tbody></table>"
    )
    parts.append(f'<p class="note">{_esc(prov.get("reproducibility_claim"))}</p>')
    parts.append(
        '<p class="note">Independently verify: '
        "<code>python -m pii_anon.assurance verify --report &lt;published&gt; --against &lt;rerun&gt;</code></p>"
    )

    parts.append('<div class="foot">Generated by the pii-anon Assurance Report harness. '
                 "Every number passes a no-fabrication gate; every artifact passes a byte-level "
                 "PII-egress gate before write.</div>")
    parts.append("</div></body></html>")
    return {"report.html": "".join(parts) + "\n"}
