from __future__ import annotations

import json
import os
import subprocess
import sys


def test_continuity_benchmark_script_generates_outputs(tmp_path) -> None:
    output_json = tmp_path / "continuity.json"
    output_md = tmp_path / "continuity.md"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_continuity_benchmark.py",
            "--max-samples",
            "20",
            "--long-token-count",
            "10000",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_md),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert 0.0 <= payload["alias_link_f1"] <= 1.0
    assert 0.0 <= payload["pseudonym_consistency"] <= 1.0
    assert 0.0 <= payload["anonymize_placeholder_consistency"] <= 1.0
    assert payload["ambiguous_cases"] >= 0
    assert 0.0 <= payload["ambiguous_overlink_rate"] <= 1.0
    assert 0.0 <= payload["long_context_alias_recall"] <= 1.0
    assert 0.0 <= payload["long_context_alias_precision"] <= 1.0
    assert payload["mentions_expected"] > 0
    assert payload["mentions_linked"] > 0
    assert "continuity_gate_pass" in payload
    assert "Continuity Gate Report" in output_md.read_text(encoding="utf-8")
