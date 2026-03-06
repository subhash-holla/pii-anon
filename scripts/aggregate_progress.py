#!/usr/bin/env python3
"""Aggregate progress from parallel profile benchmark log files.

Reads all ``*.log`` files in a directory, parses the structured
``TOTAL:`` and ``WORK:`` markers emitted by ``_ProgressReporter``,
and renders a single unified progress bar to stdout at the specified
interval (default: 60 seconds).

The output matches the non-TTY heartbeat format of
``_ProgressReporter`` so it looks identical to the sequential
progress display, just aggregated across all profiles.

Usage:
    python scripts/aggregate_progress.py artifacts/benchmarks/logs/ [--interval 60]
"""
from __future__ import annotations

import argparse
import re
import signal
import sys
import time
from pathlib import Path

_RE_TOTAL = re.compile(r"^.*TOTAL:([\d.]+)\|(.*)$")
_RE_WORK = re.compile(r"^.*WORK:([\d.]+)\|(.*)$")

BAR_WIDTH = 30


def _format_duration(seconds: float) -> str:
    if seconds < 0:
        return "--:--"
    total = int(seconds)
    if total < 3600:
        m, s = divmod(total, 60)
        return f"{m:02d}:{s:02d}"
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    return f"{h}:{m:02d}:{s:02d}"


class ProfileState:
    """Track TOTAL and WORK for a single profile log file."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.profile = log_path.stem  # e.g. "short_chat"
        self.total_work: float = 0.0
        self.work_done: float = 0.0
        self.last_message: str = "waiting"
        self._file_pos: int = 0  # track read offset for incremental reads
        self.done: bool = False

    def update(self) -> None:
        """Read new lines from the log file and update state."""
        if not self.log_path.exists():
            return
        try:
            with self.log_path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._file_pos)
                new_data = fh.read()
                self._file_pos = fh.tell()
        except OSError:
            return

        for line in new_data.splitlines():
            total_m = _RE_TOTAL.match(line)
            work_m = _RE_WORK.match(line)
            if total_m:
                self.total_work = float(total_m.group(1))
                self.last_message = total_m.group(2)
            elif work_m:
                self.work_done += float(work_m.group(1))
                self.last_message = work_m.group(2)

            # Detect completion
            if "complete" in line.lower() and "100.00%" in line:
                self.done = True
            if "wrote " in line and "benchmark" in line:
                self.done = True

    @property
    def percent(self) -> float:
        if self.total_work <= 0:
            return 0.0
        return min(99.99, (self.work_done / self.total_work) * 100.0)


def render_aggregate(
    profiles: list[ProfileState],
    start_time: float,
) -> str:
    """Render a single aggregated progress line."""
    total_work = sum(p.total_work for p in profiles)
    work_done = sum(p.work_done for p in profiles)

    if total_work > 0:
        percent = min(99.99, (work_done / total_work) * 100.0)
    else:
        percent = 0.0

    elapsed = time.monotonic() - start_time
    if work_done > 0 and elapsed > 0 and total_work > 0:
        rate = work_done / elapsed
        remaining = total_work - work_done
        eta = remaining / rate if remaining > 0 else 0.0
    else:
        eta = -1.0

    filled = int(BAR_WIDTH * percent / 100.0)
    bar = "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)
    eta_str = _format_duration(eta) if eta >= 0 else "--:--"
    work_str = f"{work_done:.0f}/{total_work:.0f}" if total_work else "n/a"

    # Build per-profile summary
    done_count = sum(1 for p in profiles if p.done)
    active_count = sum(1 for p in profiles if p.total_work > 0 and not p.done)
    waiting_count = sum(1 for p in profiles if p.total_work == 0 and not p.done)

    profile_summary = f"{done_count} done, {active_count} active"
    if waiting_count > 0:
        profile_summary += f", {waiting_count} waiting"

    timestamp = time.strftime("%H:%M:%S")
    return (
        f"[progress {timestamp}] "
        f"|{bar}| {percent:6.2f}% | "
        f"{_format_duration(elapsed)} elapsed | "
        f"ETA {eta_str} | "
        f"work: {work_str} | "
        f"profiles: {profile_summary}"
    )


def render_per_profile(profiles: list[ProfileState]) -> str:
    """Render a compact per-profile status breakdown."""
    lines = []
    for p in profiles:
        if p.done:
            status = "\033[32mDONE\033[0m"
        elif p.total_work > 0:
            status = f"{p.percent:5.1f}%"
        else:
            status = "wait"
        label = p.profile[:14]
        lines.append(f"    {label:<16s} {status}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate progress from parallel benchmark logs"
    )
    parser.add_argument(
        "log_dir",
        help="Directory containing per-profile *.log files",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between progress updates (default: 60)",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"Log directory not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    # Discover log files (may appear later as processes start)
    profiles: dict[str, ProfileState] = {}
    start_time = time.monotonic()

    # Graceful shutdown on SIGTERM/SIGINT
    stop = False

    def _handle_signal(signum: int, frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(
        f"[aggregate] Watching {log_dir} for profile logs "
        f"(updates every {args.interval}s)",
        flush=True,
    )

    while not stop:
        # Discover new log files
        for log_file in sorted(log_dir.glob("*.log")):
            name = log_file.stem
            if name not in profiles and name != "merge":
                profiles[name] = ProfileState(log_file)

        # Update all profile states
        for p in profiles.values():
            p.update()

        # Render aggregate progress
        profile_list = sorted(profiles.values(), key=lambda p: p.profile)
        if profile_list:
            print(render_aggregate(profile_list, start_time), flush=True)
            print(render_per_profile(profile_list), flush=True)

        # Check if all profiles are done
        if profile_list and all(p.done for p in profile_list):
            total_work = sum(p.total_work for p in profile_list)
            work_done = sum(p.work_done for p in profile_list)
            elapsed = time.monotonic() - start_time
            timestamp = time.strftime("%H:%M:%S")
            bar = "\u2588" * BAR_WIDTH
            print(
                f"[progress {timestamp}] "
                f"|{bar}| 100.00% | "
                f"{_format_duration(elapsed)} elapsed | "
                f"work: {work_done:.0f}/{total_work:.0f} | "
                f"ALL {len(profile_list)} PROFILES COMPLETE",
                flush=True,
            )
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
