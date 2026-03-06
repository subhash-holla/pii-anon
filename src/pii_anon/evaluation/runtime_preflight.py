from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _CompetitorProbeSpec:
    import_name: str
    model_probe: str | None = None
    model_hint: str | None = None


_COMPETITOR_PROBES: dict[str, _CompetitorProbeSpec] = {
    "presidio": _CompetitorProbeSpec(import_name="presidio_analyzer"),
    "scrubadub": _CompetitorProbeSpec(import_name="scrubadub"),
    "llm_guard": _CompetitorProbeSpec(import_name="llm_guard"),
    "gliner": _CompetitorProbeSpec(
        import_name="gliner",
        model_probe=(
            "import warnings; "
            "warnings.filterwarnings('ignore', category=FutureWarning); "
            "warnings.filterwarnings('ignore', message='.*copying from a non-meta parameter.*'); "
            "from gliner import GLiNER; "
            "GLiNER.from_pretrained('knowledgator/gliner-pii-base-v1.0'); "
            "import sys; sys.exit(0)"
        ),
        model_hint="Install GLiNER: pip install gliner",
    ),
}


def _run_python_probe(
    *,
    python_executable: str,
    code: str,
    timeout_sec: int = 60,
) -> tuple[bool, int, str]:
    try:
        proc = subprocess.run(
            [python_executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        output = "\n".join(
            part for part in [proc.stdout.strip(), proc.stderr.strip()] if part
        ).strip()
        return proc.returncode == 0, proc.returncode, output
    except subprocess.TimeoutExpired:
        return False, 124, "probe timed out"
    except Exception as exc:  # pragma: no cover - defensive
        return False, 125, f"probe failed to execute: {exc}"


def _check_shared_memory() -> tuple[bool, str, str]:
    """Check for shared-memory support.

    On Linux, ``/dev/shm`` must exist and be writable.
    On macOS (Darwin), POSIX shared memory is provided by the kernel via
    ``shm_open(3)`` and does **not** use a ``/dev/shm`` mount, so the
    filesystem check is skipped and we return *ok* directly.
    """
    system = platform.system().lower()
    if system == "darwin":
        # macOS provides POSIX shared memory via the Mach kernel; no
        # /dev/shm mount is used or expected.
        return True, "posix_shm (darwin)", "ok (macOS uses kernel shm_open)"

    shm_path = "/dev/shm"
    if not os.path.exists(shm_path):
        return False, shm_path, "missing /dev/shm"
    if not os.path.isdir(shm_path):
        return False, shm_path, "/dev/shm is not a directory"
    if not os.access(shm_path, os.R_OK | os.W_OK):
        return False, shm_path, "/dev/shm is not writable"
    return True, shm_path, "ok"


def run_benchmark_runtime_preflight(
    *,
    strict_runtime: bool = False,
    require_all_competitors: bool = False,
    require_native_competitors: bool = False,
    python_executable: str | None = None,
) -> dict[str, Any]:
    python_bin = python_executable or sys.executable
    system_name = platform.system().lower()
    shared_memory_ok, shared_memory_path, shared_memory_reason = _check_shared_memory()

    competitors: dict[str, dict[str, Any]] = {}
    available_competitors: list[str] = []
    unavailable_competitors: dict[str, str] = {}

    for name, spec in _COMPETITOR_PROBES.items():
        import_ok, import_code, import_output = _run_python_probe(
            python_executable=python_bin,
            code=f"import {spec.import_name}",
        )

        model_ok = True
        model_code = 0
        model_output = ""
        if spec.model_probe:
            model_ok, model_code, model_output = _run_python_probe(
                python_executable=python_bin,
                code=spec.model_probe,
            )

        native_ready = import_ok and model_ok
        available = native_ready if require_native_competitors else import_ok
        reason_parts: list[str] = []
        if not import_ok:
            reason_parts.append(
                f"import `{spec.import_name}` failed (code={import_code})"
            )
            if import_output:
                reason_parts.append(import_output)
        if import_ok and spec.model_probe and not model_ok:
            reason_parts.append(f"required model/resources missing (code={model_code})")
            if model_output:
                reason_parts.append(model_output)
            if spec.model_hint:
                reason_parts.append(spec.model_hint)
        reason = "; ".join(reason_parts) if reason_parts else "ok"

        competitors[name] = {
            "import_name": spec.import_name,
            "import_ok": import_ok,
            "import_return_code": import_code,
            "import_output": import_output,
            "model_check_required": spec.model_probe is not None,
            "model_ready": model_ok if spec.model_probe else None,
            "model_return_code": model_code if spec.model_probe else None,
            "model_output": model_output if spec.model_probe else None,
            "native_ready": native_ready,
            "available": available,
            "reason": reason,
        }

        if available:
            available_competitors.append(name)
        else:
            unavailable_competitors[name] = reason

    failures: list[str] = []
    _SUPPORTED_RUNTIMES = {"linux", "darwin"}
    runtime_ok = system_name in _SUPPORTED_RUNTIMES
    # Keep legacy key name for backwards-compat but widen the semantics.
    linux_runtime_ok = runtime_ok
    if strict_runtime and not runtime_ok:
        failures.append(
            f"unsupported runtime `{system_name}`: canonical benchmark requires "
            f"one of {sorted(_SUPPORTED_RUNTIMES)}"
        )
    if strict_runtime and not shared_memory_ok:
        failures.append(
            f"shared memory check failed at `{shared_memory_path}`: {shared_memory_reason}"
        )

    if require_all_competitors and unavailable_competitors:
        failures.append(
            "not all configured competitors are available: "
            + ", ".join(sorted(unavailable_competitors.keys()))
        )

    ready = not failures
    return {
        "schema_version": "2026-02-16.v1",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_executable": python_bin,
        "python_version": platform.python_version(),
        "platform_system": system_name,
        "platform_release": platform.release(),
        "strict_runtime": strict_runtime,
        "require_all_competitors": require_all_competitors,
        "require_native_competitors": require_native_competitors,
        "linux_runtime_ok": linux_runtime_ok,
        "shared_memory_ok": shared_memory_ok,
        "shared_memory_path": shared_memory_path,
        "shared_memory_reason": shared_memory_reason,
        "expected_competitors": list(_COMPETITOR_PROBES.keys()),
        "available_competitors": available_competitors,
        "unavailable_competitors": unavailable_competitors,
        "all_competitors_available": not unavailable_competitors,
        "competitors": competitors,
        "ready": ready,
        "failures": failures,
    }

