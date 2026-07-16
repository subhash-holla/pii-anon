"""One-page PII-handling assurance summary (a trust-center / marketing certificate).

A compact, print-friendly, self-contained HTML page over the already-gated, PII-free
``report_dict``. Same output-encoding hardening as the executive renderer: every
interpolated string is :func:`html.escape`-d (quote=True); no JavaScript. The
PII-egress gate runs on the final bytes before write.

It surfaces ONLY publishable-context posture: the per-dimension headline score with
its claim-strength badge, the dataset/version fingerprint, and the verify command —
never an unqualified superlative (MEASURED and ADVISORY values are never blended).
"""

from __future__ import annotations

import html
from typing import Any

_BADGE = {
    "measured": ("MEASURED", "#0a7d33", "#e6f4ea"),
    "advisory": ("ADVISORY", "#8a6d00", "#fdf6e3"),
    "not_assessable": ("NOT ASSESSABLE", "#5b6166", "#eef0f2"),
}


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _badge(cs: str) -> str:
    label, fg, bg = _BADGE.get(cs, (cs, "#5b6166", "#eef0f2"))
    return f'<span class="badge" style="color:{fg};background:{bg}">{_esc(label)}</span>'


_STYLE = """
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1d20;margin:0;padding:36px}
.card{max-width:720px;margin:0 auto;border:1px solid #dfe3e6;border-radius:10px;padding:28px 32px}
h1{font-size:21px;margin:0} .tag{color:#5b6166;font-size:13px;margin:4px 0 18px}
.row{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:1px solid #eef0f2}
.row:last-child{border-bottom:none}
.dim{font-weight:600;text-transform:capitalize} .val{font-variant-numeric:tabular-nums;margin-right:10px}
.badge{display:inline-block;font-size:10.5px;font-weight:700;padding:2px 7px;border-radius:10px}
.caveats{margin:14px 0 0;padding-left:18px;font-size:11px;color:#5b6166;line-height:1.5}
.caveats li{margin:2px 0}
.fp{margin-top:14px;font-size:11.5px;color:#5b6166;line-height:1.5}
code{background:#f2f3f5;padding:1px 4px;border-radius:3px}
.hdr{display:flex;justify-content:space-between;align-items:baseline}
"""


def render_summary(report_dict: dict[str, Any]) -> dict[str, str]:
    d = report_dict
    ds = d["dataset"]
    prov = d.get("provenance", {})
    up = prov.get("user_pipeline", {})

    rows: list[str] = []
    caveats: list[str] = []
    seen: set[str] = set()
    for dim, by_system in d.get("dimensions", {}).items():
        for sys, r in by_system.items():
            cs = r.get("claim_strength", "not_assessable")
            rows.append(
                '<div class="row">'
                f'<span class="dim">{_esc(dim)} · {_esc(sys)}</span>'
                f'<span><span class="val">{_fmt(r.get("value"))}</span>{_badge(cs)}</span>'
                "</div>"
            )
            # surface every dimension's caveats verbatim — the re-id ANTI_ANONYMITY_CAVEAT is
            # non-strippable (a bare re-id score on a marketing certificate with no "a low rate
            # is NOT a guarantee of anonymity" counter-text reads as the very guarantee it forbids).
            for c in r.get("caveats", []):
                if c not in seen:
                    seen.add(c)
                    caveats.append(c)
    caveats_html = (
        '<ul class="caveats">' + "".join(f"<li>{_esc(c)}</li>" for c in caveats) + "</ul>"
        if caveats else ""
    )

    parts: list[str] = []
    parts.append('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">')
    parts.append('<meta name="viewport" content="width=device-width, initial-scale=1">')
    parts.append(f"<title>PII Assurance Summary — {_esc(ds['name'])}</title>")
    parts.append(f"<style>{_STYLE}</style></head><body><div class=\"card\">")
    parts.append(
        '<div class="hdr"><h1>PII Handling Assurance</h1>'
        f'<span class="tag">{_esc(d.get("generated_at", "—"))}</span></div>'
    )
    parts.append(
        f'<p class="tag">Dataset <code>{_esc(ds["name"])}</code> · {int(ds["n_records"])} records · '
        f"mode {_esc(ds['mode'])} · systems {_esc(', '.join(d['systems']))} "
        f"(baseline {_esc(d['baseline'])})</p>"
    )
    parts.append("".join(rows))
    parts.append(caveats_html)
    parts.append(
        '<p class="fp">'
        f"Dataset fingerprint <code>{_esc(ds['fingerprint'][:24])}…</code> · "
        f"pii-anon <code>{_esc(prov.get('pii_anon_version'))}</code> · "
        f"pipeline <code>{_esc(up.get('name'))}</code> v<code>{_esc(up.get('version'))}</code> · "
        f"seed <code>{_esc(prov.get('seed'))}</code><br>"
        "MEASURED = computed vs gold labels &amp; statistically powered (publishable). "
        "ADVISORY = hedged (silver / under-powered / representative stand-in / capability axis). "
        "NOT ASSESSABLE = required input absent (never scored as 0). MEASURED and ADVISORY values "
        "are never blended into one headline.<br>"
        "Independently reproducible by a holder of the dataset via "
        "<code>python -m pii_anon.assurance verify</code>."
        "</p>"
    )
    parts.append("</div></body></html>")
    return {"summary.html": "".join(parts) + "\n"}
