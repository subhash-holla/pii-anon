"""Shared progress indicator for long-running Makefile workflows.

Provides :class:`ProgressTracker`, a single-line, 60s-cadence, TTY-aware
progress bar with 0.01%-resolution overall progress, buffered phase-log,
and an optional background heartbeat for cases where the caller cannot
drive ``advance()`` at least once per refresh interval.

Two usage patterns are supported:

1. **Direct** (``train_swarm.py``): the caller knows ``total_work``
   up-front and synchronously calls
   :meth:`set_phase` / :meth:`advance` / :meth:`finish_phase`.

2. **Message-driven** (``run_competitor_benchmark.py``): a library deep
   in the call stack emits ``TOTAL:<n>|msg`` / ``WORK:<n>|msg`` strings
   to a progress hook, and the caller funnels them through
   :meth:`hook_message`.

Zero third-party dependencies — writes to ``sys.stderr`` with stdlib
facilities only. On non-TTY streams, the tracker falls back to emitting
one :mod:`logging`-routed line per refresh interval so piped output
(``| tee logfile``) stays readable instead of littered with ``\\r``
bytes.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from threading import Event, Thread
from typing import Any

logger = logging.getLogger("progress")

_RE_TOTAL = re.compile(r"^TOTAL:([\d.]+)\|(.*)$")
_RE_WORK = re.compile(r"^WORK:([\d.]+)\|(.*)$")

_LINE_WIDTH = 120
_BAR_WIDTH = 30


class ProgressTracker:
    """Single-line progress bar shared by training and benchmark scripts."""

    def __init__(
        self,
        total_work: int = 0,
        label: str = "Working",
        *,
        refresh_s: float = 60.0,
        stream: Any = None,
        initial_completed: int = 0,
        heartbeat: bool = False,
    ) -> None:
        self._total = max(0, int(total_work))
        self._completed = float(initial_completed)
        self._label = label
        self._refresh_s = refresh_s
        self._stream = stream if stream is not None else sys.stderr
        self._is_tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._t_start = time.monotonic()
        self._last_print = 0.0
        self._rate_ema = 0.0
        self._phase = label
        self._phase_start = self._t_start
        self._phase_completed = 0.0
        self._phase_total = 0
        self._heartbeat_enabled = heartbeat
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._started = False
        self._finished = False
        self.phase_log: list[str] = []

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Paint the initial line and start the heartbeat thread (if any)."""
        if self._started:
            return
        self._started = True
        self._t_start = time.monotonic()
        self._phase_start = self._t_start
        self._render(force=True)

        if self._heartbeat_enabled and not self._is_tty:
            def _beat() -> None:
                while not self._stop_event.wait(timeout=self._refresh_s):
                    self._render(force=True)

            self._thread = Thread(target=_beat, daemon=True)
            self._thread.start()

    def finish(self) -> None:
        """Clear the in-place line and stop the heartbeat. Idempotent."""
        if self._finished:
            return
        self._finished = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        elapsed = time.monotonic() - self._t_start
        if self._is_tty:
            try:
                self._stream.write("\r" + " " * _LINE_WIDTH + "\r")
                self._stream.flush()
            except Exception:
                pass
        final_completed = self._completed
        final_pct = self._percent()
        rate = final_completed / max(elapsed, 1e-9)
        self.phase_log.append(
            f"[{final_pct:6.2f}%] Pipeline complete: {int(final_completed):,} units in "
            f"{_format_elapsed(elapsed)} ({rate:.0f} units/s)"
        )

    def __enter__(self) -> "ProgressTracker":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.finish()

    # ── Total-work management ─────────────────────────────────────────────

    def set_total(self, total: int) -> None:
        self._total = max(0, int(total))

    def extend_total(self, extra: int) -> None:
        self._total = max(0, self._total + int(extra))

    # ── Phase & advance ───────────────────────────────────────────────────

    def set_phase(self, name: str, phase_total: int = 0) -> None:
        self._phase = name
        self._phase_start = time.monotonic()
        self._phase_completed = 0.0
        self._phase_total = max(0, int(phase_total))
        self._render(force=True)

    def set_message(self, message: str) -> None:
        """Update the status message without resetting phase counters."""
        self._phase = message
        self._render(force=True)

    def advance(self, n: float = 1) -> None:
        self._completed += n
        self._phase_completed += n

        elapsed = time.monotonic() - self._t_start
        if elapsed > 0:
            rate = self._completed / elapsed
            if self._rate_ema <= 0:
                self._rate_ema = rate
            else:
                self._rate_ema = 0.95 * self._rate_ema + 0.05 * rate

        now = time.monotonic()
        if now - self._last_print >= self._refresh_s:
            self._render()

    def finish_phase(self, message: str = "") -> None:
        phase_elapsed = time.monotonic() - self._phase_start
        pct = self._percent()
        if message:
            self.phase_log.append(
                f"[{pct:6.2f}%] {message} ({_format_elapsed(phase_elapsed)})"
            )
        self._render(force=True)

    # ── Message-driven adapter (TOTAL: / WORK: protocol) ──────────────────

    def hook_message(self, message: str) -> None:
        """Consume a structured or plain progress message.

        ``TOTAL:<n>|text`` declares the denominator for the overall bar.
        ``WORK:<n>|text`` increments ``completed`` by ``n`` units.
        Anything else is treated as a plain status update (no counter
        change).  The latest text is used as the phase label.
        """
        total_match = _RE_TOTAL.match(message)
        if total_match:
            self.set_total(int(float(total_match.group(1))))
            self.set_message(total_match.group(2))
            return

        work_match = _RE_WORK.match(message)
        if work_match:
            n = float(work_match.group(1))
            text = work_match.group(2)
            self.advance(n)
            # Update the label without another render — advance() already
            # handled the refresh-cadence gating.
            self._phase = text
            return

        self.set_message(message)

    # ── Rendering ─────────────────────────────────────────────────────────

    def _percent(self) -> float:
        if self._total <= 0:
            return 0.0
        return min(100.0, self._completed / self._total * 100.0)

    def _render(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_print < self._refresh_s:
            return
        self._last_print = now

        pct = self._percent()
        remaining = max(0.0, self._total - self._completed)
        eta_s = remaining / self._rate_ema if self._rate_ema > 0 else 0
        elapsed = now - self._t_start

        if self._is_tty:
            filled = int(_BAR_WIDTH * pct / 100.0)
            bar = "\u2588" * filled + "\u2591" * (_BAR_WIDTH - filled)
            total_s = f"{self._total:,}" if self._total else "?"
            line = (
                f"\r  {bar} {pct:6.2f}% | {int(self._completed):,}/{total_s} | "
                f"{self._rate_ema:.0f} u/s | ETA {_format_elapsed(eta_s)} | "
                f"{self._phase}"
            )
            try:
                self._stream.write(line.ljust(_LINE_WIDTH))
                self._stream.flush()
            except Exception:
                pass
        else:
            logger.info(
                "[%6.2f%%] %s — %s elapsed, ETA %s, %.0f u/s",
                pct, self._phase, _format_elapsed(elapsed),
                _format_elapsed(eta_s), self._rate_ema,
            )


# ── Module-level formatters (exported so callers can reuse) ──────────────


def _format_elapsed(seconds: float) -> str:
    if seconds < 0:
        return "--"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def format_elapsed(seconds: float) -> str:
    """Public alias of the duration formatter for callers."""
    return _format_elapsed(seconds)
