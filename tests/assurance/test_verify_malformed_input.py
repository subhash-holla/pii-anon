"""Regression: external `verify` mode fails CLOSED (no raw traceback) on adversarial input.

`python -m pii_anon.assurance verify --report <a> --against <b>` must never leak a raw
stack trace on a malformed/adversarial published report. Three classes are pinned here,
each through BOTH the library (`verify_report_files`) and the CLI (`_verify_main`):

  1. a deeply-nested JSON array — `json.loads` raises `RecursionError`, NOT `JSONDecodeError`;
  2. a report whose `provenance` is a non-dict (e.g. a bare string);
  3. a missing report path — `read_text()` raises `FileNotFoundError` (an `OSError`).

Every case must yield NOT REPRODUCED (`reproduced=False`) and CLI exit code 1 with no
exception. verify only ever compares one-way hashes of published, PII-free reports, so the
fixtures here carry no PII.
"""

from __future__ import annotations

import json

from pii_anon.assurance import verify_report_files, verify_reproduction
from pii_anon.assurance.__main__ import main as cli_main
from pii_anon.assurance.verify import _hashes

# enough nesting to exceed CPython's parse recursion limit (raises RecursionError, not JSONDecodeError)
_DEEP = "[" * 50000 + "]" * 50000
_GOOD = {"provenance": {"dataset_fingerprint": "F",
                        "user_pipeline": {"output_hash": "O", "transform_output_hash": "T"}}}


def _write(path, text: str):
    path.write_text(text, encoding="utf-8")
    return path


def _cli(report, against) -> int:
    return cli_main(["verify", "--report", str(report), "--against", str(against)])


# --- #1 deeply-nested JSON (RecursionError) -> fail closed, either position -------
def test_deeply_nested_file_fails_closed(tmp_path) -> None:
    good = _write(tmp_path / "good.json", json.dumps(_GOOD))
    deep = _write(tmp_path / "deep.json", _DEEP)
    # report-side AND against-side both parse in the same guarded block
    assert verify_report_files(deep, good).reproduced is False
    assert verify_report_files(good, deep).reproduced is False
    assert _cli(deep, good) == 1
    assert _cli(good, deep) == 1


# --- #2 non-dict provenance -> all-None hashes -> fail closed --------------------
def test_string_provenance_fails_closed(tmp_path) -> None:
    # unit: a non-dict provenance yields all-None hashes (so it can never MATCH)
    assert _hashes({"provenance": "notadict"}) == {
        "dataset_fingerprint": None, "output_hash": None, "transform_output_hash": None}
    assert verify_reproduction({"provenance": "notadict"}, _GOOD).reproduced is False

    good = _write(tmp_path / "good.json", json.dumps(_GOOD))
    strprov = _write(tmp_path / "strprov.json", '{"provenance": "notadict"}')
    # string-provenance loaded FROM A FILE, in either position, via lib + CLI
    assert verify_report_files(strprov, good).reproduced is False
    assert verify_report_files(good, strprov).reproduced is False
    assert _cli(strprov, good) == 1
    assert _cli(good, strprov) == 1


# --- #3 missing report file (FileNotFoundError) -> fail closed -------------------
def test_missing_file_fails_closed(tmp_path) -> None:
    good = _write(tmp_path / "good.json", json.dumps(_GOOD))
    missing = tmp_path / "does_not_exist.json"
    assert verify_report_files(missing, good).reproduced is False
    assert verify_report_files(good, missing).reproduced is False
    assert _cli(missing, good) == 1
    assert _cli(good, missing) == 1
