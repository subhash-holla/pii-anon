"""Pipeline adapters (Port 1).

A :class:`PipelineAdapter` wraps a user's PII pipeline. It may expose
``detect`` (``text -> spans``) and/or ``transform`` (``text -> redacted text``);
a dimension that needs an input the adapter doesn't expose self-marks
``NOT_ASSESSABLE`` (no phantom 0).

PII-safety: ``safe_detect`` / ``safe_transform`` NEVER raise and NEVER propagate
an exception MESSAGE (the existing reuse path leaks ``exc`` text into the report).
On failure they return a SCRUBBED error string — the exception *type name* only.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from dataclasses import dataclass

Span = tuple[str, int, int]

# A committed ceiling on spans per record — far above any plausible real detection count.
# Bounds an unbounded/enormous predictor iterable (infinite generator hang, 20M-span OOM)
# so a broken pipeline degrades to a scrubbed adapter-error, never a hang/crash (design §14).
_MAX_SPANS_PER_RECORD = 100_000
DetectFn = Callable[[str], Iterable[tuple[str, int, int]]]
TransformFn = Callable[[str], str]


@dataclass(frozen=True)
class Capabilities:
    can_detect: bool
    can_transform: bool


def _normalize_spans(raw: Iterable[tuple[str, int, int]], text_len: int) -> list[Span]:
    """Coerce predictor output to validated ``(type, start, end)`` spans; drop malformed."""
    out: list[Span] = []
    for item in raw:
        try:
            etype = str(item[0])
            start = int(item[1])
            end = int(item[2])
        except Exception:  # noqa: BLE001 - any malformed element (dict span, custom __getitem__, ...) is dropped
            continue
        if etype and 0 <= start < end <= text_len:
            out.append((etype, start, end))
    return out


@dataclass
class PipelineAdapter:
    """A user (or first-party) PII pipeline wrapped for the assurance harness."""

    name: str
    detect: DetectFn | None = None
    transform: TransformFn | None = None
    version: str = "unknown"
    deterministic: bool = True

    def capabilities(self) -> Capabilities:
        return Capabilities(self.detect is not None, self.transform is not None)

    def safe_detect(self, text: str) -> tuple[list[Span], str | None]:
        """Return (spans, scrubbed_error). Never raises; never leaks exc text."""
        if self.detect is None:
            return [], "adapter exposes no detect()"
        try:
            # bounded consume: islice stops pulling at the cap, so an infinite/enormous
            # predictor iterable cannot hang or OOM the harness.
            raw = list(itertools.islice(self.detect(text), _MAX_SPANS_PER_RECORD + 1))
            if len(raw) > _MAX_SPANS_PER_RECORD:
                return [], f"detect returned > {_MAX_SPANS_PER_RECORD} spans (capped, treated as error)"
            spans = _normalize_spans(raw, len(text))  # inside try: normalization can also raise
        except BaseException as exc:  # noqa: BLE001 - scrub ANY raise (incl. SystemExit/KeyboardInterrupt) so raw text never leaks via the message
            return [], f"detect raised {type(exc).__name__}"
        return spans, None

    def safe_transform(self, text: str) -> tuple[str | None, str | None]:
        """Return (output, scrubbed_error). Never raises; never leaks exc text.

        A ``None`` output (no transform / failure / non-str) is the fail-CLOSED
        signal the leakage assessor treats as NOT_ASSESSABLE — never as 0 leakage.
        """
        if self.transform is None:
            return None, "adapter exposes no transform()"
        try:
            out = self.transform(text)
        except BaseException as exc:  # noqa: BLE001 - scrub ANY raise so raw text never leaks
            return None, f"transform raised {type(exc).__name__}"
        if not isinstance(out, str):
            return None, f"transform returned {type(out).__name__}, not str"
        return out, None


# ---------------------------------------------------------------------------
# Redaction helper + first-party (pii-anon) adapter
# ---------------------------------------------------------------------------


def redact_spans(text: str, spans: Iterable[tuple[str, int, int]], *, mask: str = "[REDACTED]") -> str:
    """Replace each detected span with a mask token. Deterministic; offset-safe
    (applies right-to-left). This is pii-anon's redaction transform composed from
    detection — it does NOT depend on the user-WIP orchestrator (the S2-03 block)."""
    normalized = sorted(_normalize_spans(spans, len(text)), key=lambda s: s[1], reverse=True)
    out = text
    for _etype, start, end in normalized:
        out = out[:start] + mask + out[end:]
    return out


def pii_anon_adapter(*, swarm: bool = False, mask: str = "[REDACTED]") -> PipelineAdapter:
    """Express pii-anon's own offering as an adapter (detect + redaction transform).

    Heavy detection machinery is imported INSIDE this factory (mirrors
    ``first_party.py``), so importing this module pulls no models.
    """
    from pii_anon import __version__ as _pkg_version  # noqa: PLC0415
    from pii_anon.eval_framework.first_party import (  # noqa: PLC0415
        pii_anon_predictor,
        pii_anon_swarm_predictor,
    )

    detect: DetectFn = pii_anon_swarm_predictor if swarm else pii_anon_predictor

    def transform(text: str) -> str:
        return redact_spans(text, detect(text), mask=mask)

    return PipelineAdapter(
        name="pii-anon-swarm" if swarm else "pii-anon",
        detect=detect,
        transform=transform,
        version=str(_pkg_version),
        deterministic=True,
    )
