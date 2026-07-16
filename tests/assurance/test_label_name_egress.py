"""Label-egress gap: a separator-joined LOWERCASE name in a user-supplied label
(pipeline name / version, dataset name) must not survive into report.json.

The PII-egress gate (``scan_output``) covers record fields only — it never scans
these labels — so ``scrub_label`` is the sole defense for them. The detector pass
inside ``scrub_label`` normalizes ``[_-]`` to spaces before probing, but the
underlying NER does not flag lowercase names (``john smith``), so a label like
``dlp-for-john-smith`` previously slipped through verbatim.

The conservative containment heuristic redacts a run of >=3 separator-joined
alphabetic tokens. The floor is 3 by construction: the safe DEFAULT labels
``user-dataset`` / ``user-pipeline`` are 2-token runs and must stay readable, so a
2-token floor is impossible. A lowercase name and a lowercase phrase are not
distinguishable, so 3+-token labels over-redact in the safe direction.
"""

from __future__ import annotations

from pathlib import Path

from pii_anon.assurance import AssuranceConfig, PipelineAdapter, run_assurance_report
from pii_anon.assurance.adapters import pii_anon_adapter
from pii_anon.assurance.pii_egress import scrub_label


def _benign_records(n: int = 25) -> list[dict[str, str]]:
    return [{"id": f"r{i}", "text": f"benign record {i} padding padding"} for i in range(n)]


# --- unit: scrub_label redacts >=3-token separator-joined lowercase names ----------
def test_scrub_label_redacts_hyphen_joined_lowercase_name() -> None:
    det = pii_anon_adapter().detect
    out = scrub_label("dlp-for-john-smith", det)
    assert "john" not in out and "smith" not in out, out


def test_scrub_label_redacts_underscore_joined_lowercase_name() -> None:
    det = pii_anon_adapter().detect
    out = scrub_label("patient_jane_doe_export", det)
    assert "jane" not in out and "doe" not in out, out


def test_scrub_label_redacts_lowercase_run_without_a_detector() -> None:
    # the heuristic is a pure regex pass: it must hold even when no detector is wired.
    out = scrub_label("patient_jane_doe_export")
    assert "jane" not in out and "doe" not in out, out


# --- unit: the safe DEFAULT labels (2-token) must stay readable --------------------
def test_scrub_label_keeps_safe_default_labels() -> None:
    det = pii_anon_adapter().detect
    assert scrub_label("user-dataset", det) == "user-dataset"
    assert scrub_label("user-pipeline", det) == "user-pipeline"


def test_scrub_label_keeps_two_token_readable_label() -> None:
    # a legitimate short 2-token label is below the redaction floor and stays readable.
    assert scrub_label("acme-dlp", pii_anon_adapter().detect) == "acme-dlp"


# --- end-to-end: the leak repro — name must not reach report.json ------------------
def test_separator_joined_lowercase_name_does_not_survive_into_report(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_assurance_report(
        AssuranceConfig(
            records=_benign_records(),
            dataset_name="patient_jane_doe_export",
            pipeline=PipelineAdapter(
                name="dlp-for-john-smith", transform=lambda t: "[X]", version="v1"
            ),
            dimensions=("leakage",),
            outputs=("json",),
            out_dir=str(out),
            seed=1,
            n_resamples=10,
        )
    )
    report = (out / "report.json").read_text()
    for leaked in ("john-smith", "john", "smith", "jane", "doe", "patient"):
        assert leaked not in report, f"{leaked!r} leaked into report.json"
