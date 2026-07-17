"""sp7 A2 — labeled-field value bridge (sp6 mining candidate #3).

A shared, domain-general cue->value bridge: when a value sits behind an
unambiguous field-label cue (``SSN:``, ``Date of Birth:``, ``Account
Number:``), extract the VALUE span and type it FROM THE LABEL. This recovers
the single largest detection-recall class after fusion (~2,570 FNs across the
five external datasets) — values whose bare shape the core patterns miss or
mistype but whose label makes them unambiguous.

Discipline
----------
* **Additive + leak-safe.** The bridge only ADDS findings; the adapter runs it
  on BOTH the production and eval paths (over-masking a labeled value is the
  safe direction). ``label-wins`` arbitration is a leak-SAFE relabel — the
  value span stays masked, only its type changes.
* **Supported types only.** Every cue maps to a type in
  ``orchestrator.SUPPORTED_ENTITY_TYPES`` so the orchestrator actually masks
  the value (mapping to an unsupported type would drop it — the sp6 presidio
  inversion leak class).
* **Anti-template.** Cue vocabulary is public, domain-general field-label
  language (never test gold or generator templates).
* **v1 scope.** Requires the strong ``:``/``=`` separator with a per-kind
  value-shape gate; the FP-prone copula bridge (``is|was|of``) and runaway
  ADDRESS capture are deferred to a later, separately-closed pass.
"""
from __future__ import annotations

import re

#: (cue phrases, entity type, value kind). Order matters only within the flat
#: alternation, which is built longest-cue-first so "social security number"
#: wins over "social security". Every entity type here is asserted to be in
#: SUPPORTED_ENTITY_TYPES by tests/test_labeled_fields_sp7.py.
_CUE_SPEC: tuple[tuple[tuple[str, ...], str, str], ...] = (
    (("social security number", "social security no", "social security", "ssn"), "US_SSN", "number"),
    (("date of birth", "birth date", "birthdate", "born on", "dob"), "DATE_OF_BIRTH", "date"),
    (("phone number", "telephone", "mobile number", "cell phone", "contact number",
      "phone", "mobile", "cell", "fax", "tel"), "PHONE_NUMBER", "number"),
    (("credit card number", "debit card number", "credit card", "card number",
      "card no", "cc number"), "CREDIT_CARD", "number"),
    (("aba routing number", "routing number", "aba number", "routing no"), "ROUTING_NUMBER", "number"),
    (("bank account number", "account number", "acct number", "bank account",
      "account no", "acct no"), "BANK_ACCOUNT", "alnum_id"),
    (("iban",), "IBAN", "token"),
    (("card verification value", "security code", "cvv2", "cvv", "cvc"), "CVV", "number"),
    (("pin number", "pin code", "pin"), "PIN", "number"),
    (("ip address",), "IP_ADDRESS", "ipish"),
    (("mac address",), "MAC_ADDRESS", "token"),
    (("national identification number", "national id number", "national id"), "NATIONAL_ID", "alnum_id"),
    (("medical record number", "medical record no", "medical record", "mrn",
      "patient id"), "MEDICAL_RECORD_NUMBER", "alnum_id"),
    (("employee number", "employee id", "staff id", "badge number", "emp id"), "EMPLOYEE_ID", "alnum_id"),
    (("customer id", "customer number", "customer no", "client id", "client number",
      "account holder id", "member id", "subscriber id"), "CUSTOMER_ID", "alnum_id"),
    (("invoice number", "invoice no"), "INVOICE_NUMBER", "alnum_id"),
    (("insurance policy number", "policy number", "insurance number", "policy no"),
     "INSURANCE_POLICY_NUMBER", "alnum_id"),
    (("docket number", "docket no", "docket"), "DOCKET_NUMBER", "alnum_id"),
    (("court case number", "case number", "cause number", "case no"), "COURT_CASE_NUMBER", "alnum_id"),
    (("attorney bar number", "state bar number", "bar number", "bar no"), "BAR_NUMBER", "alnum_id"),
    (("driver's license number", "driver license number", "driving licence",
      "driver's license", "drivers license", "license number", "dl number"),
     "DRIVERS_LICENSE", "alnum_id"),
    (("passport number", "passport no", "passport"), "PASSPORT", "token"),
    (("vehicle identification number", "vin"), "VIN", "token"),
    (("license plate number", "registration number", "license plate",
      "plate number", "number plate"), "LICENSE_PLATE", "token"),
    (("wallet address", "crypto wallet", "bitcoin address"), "CRYPTO_WALLET", "token"),
    (("user name", "username", "user id", "login id", "screen name", "login",
      "handle"), "USERNAME", "token"),
    (("password", "passwd", "pwd"), "PASSWORD", "token"),
    (("annual salary", "gross salary", "annual income", "compensation", "salary",
      "income"), "SALARY", "number"),
    (("age",), "AGE", "age"),
    (("full name", "first name", "last name", "given name", "surname",
      "customer name", "patient name", "employee name", "cardholder name",
      "account holder name", "account holder", "contact name"), "PERSON_NAME", "name"),
)

#: cue phrase -> (entity_type, value_kind)
_CUE_MAP: dict[str, tuple[str, str]] = {
    cue: (etype, kind) for cues, etype, kind in _CUE_SPEC for cue in cues
}

#: All cue phrases, longest first so multi-word cues win the alternation.
_ALL_CUES = sorted(_CUE_MAP, key=len, reverse=True)

#: cue (word-bounded, not a substring) + separator + optional wrapper run.
_CUE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(c) for c in _ALL_CUES) + r")\b"
    r"(?:[ \t]|[*_`\"'\[])*[:=](?:[ \t]|[*_`\"'\[])*",
    re.IGNORECASE,
)

#: values that a strong cue is often "set to" but which are not PII.
_NULL_VALUES = frozenset({
    "n/a", "na", "none", "null", "nil", "unknown", "unspecified", "tbd", "tba",
    "pending", "redacted", "not provided", "not available", "not set", "n.a.",
    "xxx", "xxxx", "test", "example", "sample",
})

_CONF_NUMERIC = 0.90
_CONF_NAME = 0.92

#: SPDX / open-source software-license identifiers: a value behind a
#: "license number" cue that leads with one of these is a software licence,
#: not a driver's licence (keeps the DRIVERS_LICENSE cue from eating
#: "License Number: GPL-30000").
_SPDX_PREFIXES = frozenset({
    "gpl", "lgpl", "agpl", "mit", "bsd", "apache", "mpl", "epl", "cc", "isc",
    "zlib", "cddl", "unlicense", "wtfpl", "artistic", "boost", "ncsa", "ofl",
})
_LEAD_ALPHA_RE = re.compile(r"^[A-Za-z]+")

# ── per-kind value parsers: (region) -> (start, end) within region, or None ──

_NUMBER_RE = re.compile(r"[$£€]?[ \t]*(\(?\d[\d ()\-./]*\d|\d)")
_ALNUM_ID_RE = re.compile(r"([A-Za-z0-9][A-Za-z0-9\-/#.]{1,38})")
_TOKEN_RE = re.compile(r"(\S{2,60})")
_IPISH_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{1,4}){2,7})")
_AGE_RE = re.compile(r"(\d{1,3})(?!\d)")
_DATE_RE = re.compile(
    r"(\d{1,4}[/\-.]\d{1,2}[/\-.]\d{1,4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{2,4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{2,4})"
)
_NAME_RE = re.compile(
    r"((?:[A-Z][a-z'’.\-]+|[A-Z]\.)(?:\s+(?:[A-Z][a-z'’.\-]+|[A-Z]\.)){0,3})"
)

_TRIM = " \t.,;:!?)]}'\"`*_"


def _digit_count(s: str) -> int:
    return sum(c.isdigit() for c in s)


def _parse(kind: str, region: str, etype: str) -> tuple[int, int] | None:
    if kind == "number":
        m = _NUMBER_RE.match(region)
        if not m or _digit_count(m.group(1)) < 2:
            return None
        return m.start(1), m.end(1)
    if kind == "age":
        m = _AGE_RE.match(region)
        if not m:
            return None
        return m.start(1), m.end(1)
    if kind == "alnum_id":
        m = _ALNUM_ID_RE.match(region)
        if not m:
            return None
        val = m.group(1)
        if _digit_count(val) < 1 or len(val) < 3 or val.lower() in _NULL_VALUES:
            return None
        # SPDX software-licence veto for the DRIVERS_LICENSE cue family.
        if etype == "DRIVERS_LICENSE":
            lead = _LEAD_ALPHA_RE.match(val)
            if lead and lead.group(0).lower() in _SPDX_PREFIXES:
                return None
        return m.start(1), m.end(1)
    if kind == "token":
        m = _TOKEN_RE.match(region)
        if not m:
            return None
        val = m.group(1).strip(_TRIM)
        # VIN has a fixed canonical length (11-17 alnum); a shorter value
        # behind a "VIN:" cue is not a VIN (respects the pattern's guard).
        min_len = 11 if etype == "VIN" else 3
        if len(val) < min_len or val.lower() in _NULL_VALUES:
            return None
        start = m.start(1) + (m.group(1).index(val) if val in m.group(1) else 0)
        return start, start + len(val)
    if kind == "ipish":
        m = _IPISH_RE.match(region)
        return (m.start(1), m.end(1)) if m else None
    if kind == "date":
        m = _DATE_RE.match(region)
        return (m.start(1), m.end(1)) if m else None
    if kind == "name":
        m = _NAME_RE.match(region)
        if not m or m.group(1).strip(_TRIM).lower() in _NULL_VALUES:
            return None
        return m.start(1), m.end(1)
    return None


def extract_labeled_fields(text: str) -> list[tuple[str, int, int, float]]:
    """Return ``(entity_type, span_start, span_end, confidence)`` for every
    value found behind a recognised field-label cue in *text*.

    The span is the VALUE only (never the label token). Additive and
    leak-safe: callers may emit these findings on any path.
    """
    out: list[tuple[str, int, int, float]] = []
    for cm in _CUE_RE.finditer(text):
        cue = cm.group(1).lower()
        spec = _CUE_MAP.get(cue)
        if spec is None:
            continue
        etype, kind = spec
        vstart = cm.end()
        region = text[vstart : vstart + 80]
        parsed = _parse(kind, region, etype)
        if parsed is None:
            continue
        rs, re_ = parsed
        abs_s, abs_e = vstart + rs, vstart + re_
        # trim trailing punctuation the parser may have admitted
        while abs_e > abs_s and text[abs_e - 1] in _TRIM:
            abs_e -= 1
        if abs_e <= abs_s:
            continue
        conf = _CONF_NAME if kind == "name" else _CONF_NUMERIC
        out.append((etype, abs_s, abs_e, conf))
    return out
