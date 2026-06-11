"""Context-aware confidence scoring for regex PII detections.

Inspired by Microsoft Presidio's ``LemmaContextAwareEnhancer``, this module
adjusts detection confidence based on surrounding text.  When context keywords
appear near a matched span (e.g., "social security" near a 9-digit number),
confidence is boosted.  When context is absent for high false-positive entity
types (US_SSN, PERSON_NAME), confidence is penalized.

Algorithm
---------
1. Extract a window of *±CONTEXT_WINDOW* characters around the matched span.
2. Negative-context arbitration (two rules — see the *NEGATIVE_CONTEXT_WORDS*
   comment): VALUE-POSITION types (US_SSN, PHONE_NUMBER) scan the pre/post
   window segments EXCLUDING the span text for a foreign field label; the
   LABEL-POSITION type (ORGANIZATION) demotes only when a negative phrase
   starts inside the span AND label-position evidence (optional qualifier +
   separator + value) follows the span. Either way, an own-type keyword in
   the *NEGATIVE_ADJACENCY_WINDOW* chars immediately before the span
   (trailing whitespace stripped) protects. A demotable match is demoted
   by *NEGATIVE_CONTEXT_PENALTY* (plus *CONTEXT_PENALTY* for
   *HIGH_FP_TYPES*), floored at *CONFIDENCE_FLOOR*.
3. Tokenize the window into lowercase words.
4. Intersect tokens with the entity-type's keyword set.
5. If intersection is non-empty → boost by *CONTEXT_BOOST* (capped at 0.99).
6. If empty **and** the entity type is in *HIGH_FP_TYPES* → penalize by
   *CONTEXT_PENALTY* (floored at 0.40).
7. Otherwise → return the base confidence unchanged.

This approach avoids NLP dependencies (no lemmatization needed) while still
providing meaningful confidence differentiation.
"""

from __future__ import annotations

import re

# ── Context keyword sets per entity type ───────────────────────────────────
# Each set contains lowercase tokens that commonly appear near genuine PII
# of the given type.

CONTEXT_WORDS: dict[str, set[str]] = {
    "US_SSN": {
        "social", "security", "ssn", "tax", "tin", "taxpayer",
        "identification", "ss#",
    },
    "CREDIT_CARD": {
        "credit", "card", "visa", "mastercard", "amex", "discover",
        "payment", "debit", "charge", "cc", "cardnumber",
        # ES / FR / DE / PT
        "tarjeta", "crédito", "credito", "carte", "crédit", "credit",
        "kreditkarte", "karte", "cartão",
        # ZH / JA
        "信用卡", "クレジット", "カード",
    },
    "PHONE_NUMBER": {
        "call", "phone", "tel", "telephone", "mobile", "fax",
        "cell", "contact", "reach", "cellphone",
        # ES / FR / DE / PT
        "teléfono", "telefono", "móvil", "movil", "téléphone",
        "portable", "telefon", "handy", "mobilnummer", "telefone",
        # ZH / JA / KO / AR
        "电话", "手机", "電話", "携帯", "전화", "هاتف", "جوال",
    },
    "EMAIL_ADDRESS": {
        "email", "mail", "e-mail", "contact", "send", "address",
        # ES / FR / DE / PT
        "correo", "electrónico", "electronico", "courriel",
        "e-mail-adresse", "mail-adresse", "correio",
        # ZH / JA / KO / AR
        "邮箱", "电子邮件", "メール", "이메일", "بريد",
    },
    "IP_ADDRESS": {
        "ip", "host", "server", "network", "ipv4", "ipv6",
    },
    "IBAN": {
        "iban", "bank", "transfer", "wire", "international", "bic",
    },
    "ROUTING_NUMBER": {
        "routing", "aba", "transit", "bank", "sort",
    },
    "PERSON_NAME": {
        "name", "person", "patient", "employee", "client",
        "member", "user", "resident", "beneficiary", "customer",
        # ES / FR / DE / PT
        "nombre", "persona", "paciente", "empleado", "cliente",
        "nom", "personne", "employé", "employe",
        "patient", "mitarbeiter", "kunde",
        "nome", "pessoa", "paciente", "funcionário", "funcionario",
        # ZH / JA / KO / AR
        "姓名", "名字", "病人", "员工", "客户",
        "氏名", "患者", "従業員", "顧客",
        "이름", "환자", "직원",
        "اسم", "مريض", "موظف",
    },
    "AGE": {
        "age", "aged", "years", "old", "born", "birthday",
    },
    "MEDICAL_LICENSE": {
        "npi", "dea", "license", "provider", "physician", "prescriber",
    },
    # --- Entity types added for broader context coverage ---
    "DATE_OF_BIRTH": {
        "born", "birth", "dob", "birthday", "birthdate", "date",
        "nacimiento", "naissance",
    },
    "BANK_ACCOUNT": {
        "account", "bank", "checking", "savings", "deposit",
        "acct", "cuenta", "compte", "routing", "wire",
        "transfer", "iban", "swift", "ach",
    },
    "DRIVERS_LICENSE": {
        "driver", "license", "licence", "dl", "driving", "permit",
        "licencia", "permis",
    },
    "PASSPORT": {
        "passport", "pasaporte", "passeport", "travel", "document",
        "visa", "immigration",
    },
    "NATIONAL_ID": {
        "national", "identity", "id", "identification", "cedula",
        "dni", "citizen",
    },
    "VIN": {
        "vin", "vehicle", "car", "automobile", "chassis",
        "identification", "registration",
    },
    "MAC_ADDRESS": {
        "mac", "hardware", "device", "interface", "ethernet",
        "wifi", "adapter", "network",
    },
    "EMPLOYEE_ID": {
        "employee", "emp", "staff", "personnel", "worker",
        "badge", "payroll", "contractor",
    },
    "ORGANIZATION": {
        "company", "corporation", "inc", "ltd", "llc", "org",
        "enterprise", "firm", "business", "employer",
    },
    "LOCATION": {
        "location", "city", "state", "country", "region",
        "address", "place", "area", "district", "province",
    },
    "ADDRESS": {
        "address", "street", "avenue", "road", "boulevard",
        "suite", "apt", "apartment", "residence", "domicilio",
        # ES / FR / DE / PT
        "dirección", "direccion", "calle", "avenida", "carretera",
        "adresse", "rue", "avenue", "boulevard", "domicile",
        "straße", "strasse", "adresse", "anschrift", "wohnort",
        "endereço", "endereco", "rua", "avenida",
        # ZH / JA / KO / AR
        "地址", "住址", "街道",
        "住所", "番地",
        "주소", "거리",
        "عنوان", "شارع",
    },
    "LICENSE_PLATE": {
        "plate", "license plate", "tag", "vehicle", "registration",
        "car", "truck",
        # ES / FR / DE / PT
        "matrícula", "matricula", "placa", "patente",
        "plaque", "immatriculation", "véhicule", "vehicule",
        "kennzeichen", "nummernschild", "fahrzeug",
        "matrícula", "placa",
        # ZH / JA / KO / AR
        "车牌", "车辆", "登记",
        "ナンバー", "車両", "登録番号",
        "번호판", "차량",
        "لوحة", "مركبة", "تسجيل",
    },
    # ── Phase 3 gap-closure types (paper v11 §5.6) ────────────────────
    # Every Phase 3 pattern is context-gated at the regex level — the
    # keyword MUST appear adjacent to the captured group to match.  The
    # entries below add a second, wider ±50-char context-boost layer:
    # if additional supporting keywords appear in the broader context,
    # confidence gets a further bump.  This layered gating is what
    # keeps precision > 0.90 on these ambiguous numeric shapes.
    "CVV": {
        "cvv", "cvv2", "cvc", "cvc2", "cid", "card", "credit", "debit",
        "visa", "mastercard", "amex", "security", "verification",
        "código", "codigo", "seguridad",   # ES
        "code", "sécurité", "securite",    # FR
        "sicherheitscode", "prüfziffer",   # DE
        "安全码", "セキュリティ",           # ZH / JA
    },
    "PIN": {
        "pin", "passcode", "atm", "debit", "card", "bank", "mobile",
        "sim", "unlock", "code",
        "clave", "código", "codigo",
        "identifiant", "confidentiel",
        "kennziffer",
        "암호", "비밀번호",
    },
    "PASSWORD": {
        "password", "passwd", "pwd", "pass", "login", "credential",
        "credentials", "auth", "secret", "token", "api", "key",
        "contraseña", "clave",
        "mot", "passe",
        "passwort", "kennwort",
        "senha",
        "パスワード", "密码",
        "비밀번호",
    },
    "COURT_CASE_NUMBER": {
        "case", "court", "docket", "filing", "judgment", "judgement",
        "no.", "no", "number", "v.", "vs", "versus", "plaintiff",
        "defendant", "court", "file", "filing",
        "tribunal", "caso", "expediente",
        "dossier", "affaire",
        "rechtssache", "akten",
    },
    "DOCKET_NUMBER": {
        "docket", "case", "court", "filing", "no.", "no", "number",
        "index", "reference", "ref",
        "expediente", "caso",
        "dossier", "affaire",
    },
    "BAR_NUMBER": {
        "bar", "state", "sbn", "attorney", "lawyer", "counsel",
        "license", "licence", "admission", "admitted", "esq.", "esq",
        "abogado", "abogada",
        "avocat", "maître",
        "rechtsanwalt",
    },
    "INVOICE_NUMBER": {
        "invoice", "inv", "billing", "bill", "receipt", "order",
        "po", "purchase", "so", "sales",
        "factura",
        "facture",
        "rechnung",
        "fattura",
        "invoice", "请求书", "インボイス",
    },
    "INSURANCE_POLICY_NUMBER": {
        "policy", "policyholder", "insurance", "insured", "coverage",
        "premium", "claim", "member", "plan", "subscriber",
        "póliza", "asegurado",
        "police", "assuré", "contrat",
        "versicherung", "versichert", "vertrag",
        "polizza",
        "保険", "保险",
    },
    "SALARY": {
        "salary", "compensation", "pay", "payroll", "wage", "wages",
        "income", "earnings", "base", "annual", "yearly", "monthly",
        "remuneration", "ctc",
        "salario", "sueldo", "remuneración",
        "salaire", "rémunération",
        "gehalt", "lohn", "vergütung",
        "salário",
        "給与", "薪水",
        "월급", "연봉",
    },
}

# Entity types where absence of context should *penalize* confidence.
# Expanded based on benchmark analysis showing high false-positive rates
# for these entity types when context keywords are absent.
HIGH_FP_TYPES: frozenset[str] = frozenset({
    "US_SSN",
    "PERSON_NAME",
    "LOCATION",
    "ORGANIZATION",
    "ADDRESS",
    "PHONE_NUMBER",
    "EMPLOYEE_ID",
    "IP_ADDRESS",
    "EMAIL_ADDRESS",
    # CREDIT_CARD stays OUT of this set: adding it would penalize Luhn-valid
    # full cards lacking nearby context (-0.15), and the fragment spec sets no
    # context_type so it never reaches adjust_confidence.
    "BANK_ACCOUNT",
    # ── Phase 3: structurally-ambiguous identifiers ────────────────
    # Added because their regex surface is permissive enough that the
    # number alone produces many single-engine false positives.  The
    # context-absence penalty is what keeps precision up when the
    # swarm's Layer 1 fast-pass routes a regex hit past fusion.
    "VIN",                # 17 alphanum chars — hex strings fit
    "LICENSE_PLATE",      # 5-8 alphanum — many product codes fit
    "NATIONAL_ID",        # generic alphanum — order/serial numbers fit
})

# ── Negative-context demotion (SP1 Task 9; leak-direction fixes) ───────────
# Text mentioning ANOTHER identifier's field label is strong evidence the
# matched span belongs to that identifier, not to this entity type: the
# 9-digit run inside "National Id Number: NID-422736198" is the
# national-id value, not a US_SSN; the 10-digit run after "NPI:" is a
# provider number, not a PHONE_NUMBER; "Health Insurance" before
# ": HI-12345" is a field label, not an ORGANIZATION.
#
# PHRASE MATCHING (the substring-vs-token asymmetry fix): SHORT BARE-WORD
# phrases (single all-alphabetic tokens: "policy", "npi", "invoice",
# "routing", "insurance", "iban") match as WHOLE WORDS (\b-anchored) —
# plain substring containment suppressed the genuine phone in
# "Policyholder … (415) 555-0142" because "policy" substring-hit
# "policyholder" (a TRUE detection suppressed = unredacted PII).
# Multi-word labels ("account number", "member id", "national id") and
# punctuated/prefix-shaped forms ("nid-") keep substring containment.
# Matchers are precomputed at import (snapshot semantics, like
# _NON_LATIN_CONTEXT_KW): runtime mutation of NEGATIVE_CONTEXT_WORDS is
# not a supported surface.
#
# TWO COMPOSITION RULES (the span-self-trigger fix — the scan previously
# included the span itself, so a genuine org whose OWN NAME contains a
# negative phrase, e.g. "Social Security Administration", self-triggered
# the demotion and went unredacted):
#
# RULE A — VALUE-POSITION types (US_SSN, PHONE_NUMBER; any entry not in
# _LABEL_POSITION_TYPES). The span is a digit run; a foreign label can
# only live in the SURROUNDING text. Scan the pre-span and post-span
# window segments separately, NEVER the span text (spans are digit runs,
# so exclusion is safe and principled). Demotion fires when a negative
# phrase hits either segment AND no own-type context keyword appears in
# the NEGATIVE_ADJACENCY_WINDOW chars immediately BEFORE the span (the
# field-label position; trailing whitespace between label and span is
# stripped first, so "Phone:" + padding still protects). An own-type
# keyword elsewhere in the window does NOT protect: the dominant FP
# class carries a DISTANT own-type word — the neighbouring field's
# "contact:"/"phone:" label — while the foreign label sits directly
# before the digits (any-window-hit protection would rescue 62/85 of the
# measured phone FP class). A genuine "Phone: (415) 555-0142" keeps its
# boost even with "account number" later in the window because its own
# label is adjacent.
#
# RULE B — LABEL-POSITION types (_LABEL_POSITION_TYPES; ORGANIZATION).
# Here the FP class and the genuine mention are THE SAME TEXT SHAPE
# ("Health Insurance" the field label vs "…National Health Insurance
# Company" the org), distinguished by what FOLLOWS: a field label is
# followed by a separator + value (": HI-12345"), a real org mention is
# not. Demote ONLY when (a) a negative phrase STARTS INSIDE the span
# (the span is, or begins, the label — a phrase living entirely in the
# consumed tail, e.g. a parenthetical "(Health Insurance)", never
# demotes the org before it) AND (b) the span is followed by
# label-position evidence per _LABEL_VALUE_EVIDENCE_RE: either a
# separator [:\-#] + value token after an optional label-qualifier word
# (number/no/id/#/handle — "handle" covers the measured "Social Media
# Handle: @…" class; "id" covers "Migraine Insurance ID: INS-…", whose
# phrase straddles the span boundary), or a BARE qualifier word (the
# measured deposition class "…your Social Security Number, for the
# record?" — the field is NAMED but no value follows). Own-label
# adjacency protection applies unchanged.
#
# The demoted branch scores the span as context-ABSENT (distant own-type
# evidence is attributed to the neighbouring field), so for
# HIGH_FP_TYPES the absence penalty STACKS with the demotion:
#   max(CONFIDENCE_FLOOR, base - NEGATIVE_CONTEXT_PENALTY - CONTEXT_PENALTY)
# which lands every demotable base ≤ 0.94 below the adapter's 0.50 emit
# floor (0.65 SSN-nodash → 0.40 clamped; 0.80 phone → 0.40 clamped;
# 0.92 +1-format phone → 0.47; 0.78/0.80 ORGANIZATION → 0.40 clamped).
# EXCEPTION (pinned by test_demoted_finding_below_emit_floor_under_
# default_constants): the dashed-SSN spec's base 0.97 demotes to 0.52
# and SURVIVES the emit floor — a validated XXX-XX-XXXX value outranks
# contextual demotion; forcing it below the floor would be
# suppression-strengthening (the leak direction).
#
# BANK_ACCOUNT deliberately has NO entry: IBAN truth maps canonically to
# BANK_ACCOUNT, so BANK_ACCOUNT detections inside IBANs are real census
# true positives — suppressing them would regress G6.
NEGATIVE_CONTEXT_WORDS: dict[str, frozenset[str]] = {
    "US_SSN": frozenset({
        "national id", "nid-", "routing", "insurance", "policy",
        "npi", "member id", "account number", "invoice",
    }),
    "PHONE_NUMBER": frozenset({
        "npi", "routing", "account number", "policy", "member id",
        "iban", "invoice",
    }),
    "ORGANIZATION": frozenset({
        "health insurance", "social security", "insurance policy",
        # Measured additions (n=2000 seed=8314): field-label phrases of
        # non-ORG record fields captured as ORGANIZATION spans
        # ("Migraine Insurance" before "Insurance ID: INS-…"; "Social
        # Media" from "Social Media Handle: @…"); 0 genuine-TP windows
        # contain either phrase. Bare "insurance"/"insurance:" were
        # REJECTED: they appear in genuine "Provider: Stark Industries …
        # Insurance: …" windows whose "provider:" label is not an
        # ORGANIZATION context word, so adjacency would not protect them.
        "insurance id", "social media",
    }),
}

NEGATIVE_CONTEXT_PENALTY: float = 0.30

# Width of the label-position prefix consulted for own-type protection.
# Sized to cover the longest own-type label form ("social security
# number: " = 24 chars to the value) while excluding the PREVIOUS
# field's label ("phone: +1 (XXX) XXX-XXXX " puts "phone" ≥ 26 chars
# back from the next field's value).
NEGATIVE_ADJACENCY_WINDOW: int = 25

# Entity types whose demotion follows RULE B (label-position; see the
# NEGATIVE_CONTEXT_WORDS comment). Everything else with an entry follows
# RULE A (value-position).
_LABEL_POSITION_TYPES: frozenset[str] = frozenset({"ORGANIZATION"})

# RULE B label-position evidence, anchored at the span end. Two shapes:
# (B1) optional parenthesised qualifier + optional label-qualifier word +
#      separator + value token — "Health Insurance: HI-12345",
#      "Migraine Insurance ID: INS-…";
# (B2) a bare label-qualifier WORD (whole word, no separator/value
#      required) — the measured deposition-dialogue class "…your Social
#      Security Number, for the record?" names the FIELD, not the org
#      (tuning iteration 2: B1 alone resurrected 6 of these as ORG FPs
#      on the n=2000 seed=8314 draw).
# A real org mention ("…Administration in Baltimore", "…Insurance
# Company<EOS>") matches neither.
_LABEL_VALUE_EVIDENCE_RE = re.compile(
    r"\s*(?:\([^)]{1,24}\)\s*)?"
    r"(?:(?:number|no|id|#|handle)?\s*[:\-#]\s*\S"  # B1: separator + value
    r"|(?:number|no|id|handle)\b)",  # B2: bare qualifier word
    re.IGNORECASE,
)


def _build_negative_matchers() -> tuple[
    dict[str, re.Pattern[str]], dict[str, tuple[str, ...]]
]:
    """Split each entity type's negative phrases by shape (import-time).

    Single all-alphabetic tokens become one word-boundary-anchored
    alternation pattern; everything else (multi-word labels, punctuated
    prefixes) stays a substring tuple. See the NEGATIVE_CONTEXT_WORDS
    comment for the asymmetry rationale.
    """
    word_res: dict[str, re.Pattern[str]] = {}
    substrings: dict[str, tuple[str, ...]] = {}
    for entity_type, phrases in NEGATIVE_CONTEXT_WORDS.items():
        bare_words = sorted(p for p in phrases if p.isalpha())
        subs = tuple(sorted(p for p in phrases if not p.isalpha()))
        if bare_words:
            word_res[entity_type] = re.compile(
                r"\b(?:" + "|".join(re.escape(w) for w in bare_words) + r")\b"
            )
        if subs:
            substrings[entity_type] = subs
    return word_res, substrings


_NEG_BARE_WORD_RES, _NEG_SUBSTRING_PHRASES = _build_negative_matchers()

# Tuning constants — module-level defaults.
# These are overridden at runtime when CoreConfig.confidence is provided
# to the regex engine adapter via ``configure_from_config()``.
CONTEXT_BOOST: float = 0.10
CONTEXT_PENALTY: float = 0.15
CONTEXT_WINDOW: int = 50
CONFIDENCE_CAP: float = 0.99
CONFIDENCE_FLOOR: float = 0.40


def configure_from_config(
    *,
    context_boost: float | None = None,
    context_penalty: float | None = None,
    context_window: int | None = None,
    confidence_cap: float | None = None,
    confidence_floor: float | None = None,
    negative_context_penalty: float | None = None,
) -> None:
    """Override module-level tuning constants from external configuration.

    Called by the regex engine adapter during initialization so that
    ``CoreConfig.confidence`` values take effect.

    ``negative_context_penalty`` is wired for symmetry with the other
    runtime-tunable constants (the adapter forwards the dict key when
    present). NOTE the coupling pinned by
    ``test_demoted_finding_below_emit_floor_under_default_constants``:
    under defaults the demotion lands demotable findings below the
    adapter's 0.50 emit floor — lowering this value (or raising
    ``confidence_floor``) can turn suppression into mere dampening.
    """
    global CONTEXT_BOOST, CONTEXT_PENALTY, CONTEXT_WINDOW  # noqa: PLW0603
    global CONFIDENCE_CAP, CONFIDENCE_FLOOR  # noqa: PLW0603
    global NEGATIVE_CONTEXT_PENALTY  # noqa: PLW0603
    if context_boost is not None:
        CONTEXT_BOOST = context_boost
    if context_penalty is not None:
        CONTEXT_PENALTY = context_penalty
    if context_window is not None:
        CONTEXT_WINDOW = context_window
    if confidence_cap is not None:
        CONFIDENCE_CAP = confidence_cap
    if confidence_floor is not None:
        CONFIDENCE_FLOOR = confidence_floor
    if negative_context_penalty is not None:
        NEGATIVE_CONTEXT_PENALTY = negative_context_penalty

# Pre-compiled alphabetic tokenizer.
#
# Uses ``[A-Za-zÀ-ÿ]+`` (ASCII letters + Latin-1 supplement) so the
# tokenizer splits on punctuation AND on underscores (``\w`` treats
# underscore as a word char, which would leave ``social_security_number``
# as a single token and miss every context keyword inside it).
# Non-Latin scripts (CJK, Cyrillic, Arabic) are handled by the regex
# engine falling through to ``context_text.split()`` for whitespace-
# separated tokens — CJK is typically written without word spaces so
# keyword matches there rely on substring containment via the CONTEXT_WORDS
# set being sized accordingly.
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")

# S7-03 (FR-038): the containment map the comment above has always promised.
# Keywords with NO Latin-alphabet run (CJK / Hangul / Arabic — scripts the
# tokenizer can never produce as tokens) match by lowercase substring
# containment instead. Precomputed once at import (the S6-01 hoist lesson);
# snapshots CONTEXT_WORDS at import time — runtime mutation of CONTEXT_WORDS
# (not a supported surface) does not refresh it. Latin keywords (incl.
# accented "teléfono" and dead-but-Latin aliases like "e-mail") stay on the
# token path EXCLUSIVELY, keeping ASCII/Latin behavior byte-identical.
_NON_LATIN_CONTEXT_KW: dict[str, tuple[str, ...]] = {
    entity_type: non_latin
    for entity_type, keywords in CONTEXT_WORDS.items()
    if (
        non_latin := tuple(
            sorted(kw for kw in keywords if _WORD_RE.search(kw) is None)
        )
    )
}


def extract_context(text: str, start: int, end: int) -> str:
    """Return lowercased text surrounding the matched span.

    Parameters
    ----------
    text:
        Full input string.
    start:
        Start offset of the matched span.
    end:
        End offset of the matched span.

    Returns
    -------
    str
        Lowercase substring of *text* from ``max(0, start - CONTEXT_WINDOW)``
        to ``min(len(text), end + CONTEXT_WINDOW)``.
    """
    ctx_start = max(0, start - CONTEXT_WINDOW)
    ctx_end = min(len(text), end + CONTEXT_WINDOW)
    return text[ctx_start:ctx_end].lower()


def has_context_words(entity_type: str, context_text: str) -> bool:
    """Check if any context keywords for *entity_type* appear in *context_text*.

    Uses set intersection of word-boundary-tokenized context against the
    keyword set.  Returns ``False`` if the entity type has no configured
    keywords.

    Previously used ``str.split()`` which tokenized only on whitespace
    and missed matches like ``"social_security_number"`` (one token →
    no hit) or ``"ssn=123"`` (one token → no hit for ``"ssn"``).  The
    ``\\b\\w+\\b`` tokenizer splits on every word-boundary, which
    correctly surfaces the three tokens ``social``, ``security``,
    ``number`` from the first example and the two tokens ``ssn`` and
    ``123`` from the second.  Materially improves context-match recall
    on log-style and underscored keywords without changing the
    context-keyword sets themselves.
    """
    words = CONTEXT_WORDS.get(entity_type)
    if not words:
        return False
    # ``not words.isdisjoint(...)`` short-circuits on the first common
    # element and avoids materialising a full list.  The context_text
    # is already lowercased by ``extract_context``.
    if not words.isdisjoint(_WORD_RE.findall(context_text)):
        return True
    # S7-03 (FR-038): non-Latin keywords (CJK/Hangul/Arabic) can never be
    # produced as tokens by ``_WORD_RE`` — they match by substring
    # containment, activating the multilingual context feature the keyword
    # sets have carried all along. Latin keywords never take this path.
    non_latin = _NON_LATIN_CONTEXT_KW.get(entity_type)
    if non_latin:
        for keyword in non_latin:
            if keyword in context_text:
                return True
    return False


def has_negative_context(entity_type: str, context_text: str) -> bool:
    """Check if any negative-context phrase for *entity_type* appears in
    *context_text*.

    Shape-dependent matching (see the NEGATIVE_CONTEXT_WORDS comment):
    short bare-word phrases (``"policy"``, ``"npi"``) match as WHOLE
    WORDS — substring containment suppressed the genuine phone in
    "Policyholder … (415) 555-0142" — while multi-word field labels
    (``"account number"``) and punctuated forms (``"nid-"``) keep
    lowercase substring containment. Returns ``False`` if the entity
    type has no configured negative phrases.
    """
    word_re = _NEG_BARE_WORD_RES.get(entity_type)
    if word_re is not None and word_re.search(context_text) is not None:
        return True
    return any(
        phrase in context_text
        for phrase in _NEG_SUBSTRING_PHRASES.get(entity_type, ())
    )


def _negative_scan_segments(text: str, start: int, end: int) -> tuple[str, str]:
    """Lowercased pre-span / post-span window segments, EXCLUDING the span.

    RULE A scans these separately so the span's own text can never
    self-trigger a negative phrase (and a phrase can never be fabricated
    across the span boundary by concatenation).
    """
    ctx_start = max(0, start - CONTEXT_WINDOW)
    ctx_end = min(len(text), end + CONTEXT_WINDOW)
    return text[ctx_start:start].lower(), text[end:ctx_end].lower()


def _label_position_negative(entity_type: str, text: str, start: int, end: int) -> bool:
    """RULE B: the span is (or begins) a foreign field label.

    Demotable only when label-position evidence follows the span AND a
    negative phrase STARTS inside the span. The phrase scan zone is the
    span plus the evidence tail the regex consumed, so a phrase that
    straddles the span boundary ("Migraine Insurance| ID:") hits, while
    a phrase entirely in the tail ("Acme Corp |(Health Insurance):")
    does not.
    """
    evidence = _LABEL_VALUE_EVIDENCE_RE.match(text, end)
    if evidence is None:
        return False
    zone = text[start : evidence.end()].lower()
    span_len = end - start
    word_re = _NEG_BARE_WORD_RES.get(entity_type)
    if word_re is not None:
        match = word_re.search(zone)
        if match is not None and match.start() < span_len:
            return True
    for phrase in _NEG_SUBSTRING_PHRASES.get(entity_type, ()):
        idx = zone.find(phrase)
        if idx != -1 and idx < span_len:
            return True
    return False


def _has_demotable_negative_context(
    entity_type: str, text: str, start: int, end: int
) -> bool:
    """Dispatch the negative-context scan to the entity type's rule.

    RULE B (label-position) for _LABEL_POSITION_TYPES; RULE A
    (value-position, span-excluded window segments) for every other
    NEGATIVE_CONTEXT_WORDS entry. False when the type has no entry.
    """
    if entity_type not in NEGATIVE_CONTEXT_WORDS:
        return False
    if entity_type in _LABEL_POSITION_TYPES:
        return _label_position_negative(entity_type, text, start, end)
    pre_segment, post_segment = _negative_scan_segments(text, start, end)
    return has_negative_context(entity_type, pre_segment) or has_negative_context(
        entity_type, post_segment
    )


def _own_label_adjacent(entity_type: str, text: str, start: int) -> bool:
    """Check if an own-type context keyword sits directly before the span.

    Consults only the ``NEGATIVE_ADJACENCY_WINDOW`` chars immediately
    preceding *start* — the field-label position. Trailing whitespace
    between the label and the span is stripped BEFORE taking the prefix,
    so ``"Phone:"`` followed by padding still protects. Reuses
    ``has_context_words`` so the S7-03 non-Latin containment path applies
    to the prefix exactly as it does to the full window.
    """
    label_end = start
    while label_end > 0 and text[label_end - 1].isspace():
        label_end -= 1
    prefix = text[max(0, label_end - NEGATIVE_ADJACENCY_WINDOW) : label_end].lower()
    return has_context_words(entity_type, prefix)


def _negative_context_demotion(entity_type: str, base_confidence: float) -> float:
    """Demoted-branch arithmetic: demotion + (for HIGH_FP_TYPES) the
    context-absence penalty, applied BEFORE the floor clamp."""
    penalized = base_confidence - NEGATIVE_CONTEXT_PENALTY
    if entity_type in HIGH_FP_TYPES:
        penalized -= CONTEXT_PENALTY
    return max(CONFIDENCE_FLOOR, penalized)


def apply_negative_context(
    entity_type: str,
    base_confidence: float,
    text: str,
    start: int,
    end: int,
) -> float:
    """Apply ONLY the negative-context demotion (no boost/penalty).

    Entry point for pattern specs WITHOUT ``context_type`` (e.g. the
    ORGANIZATION specs), which never reach :func:`adjust_confidence`.
    Findings whose rule does not fire (see NEGATIVE_CONTEXT_WORDS) — or
    with an own-type label adjacent — are returned unchanged, so
    behaviour for everything outside the targeted FP classes is
    byte-identical.
    """
    if _has_demotable_negative_context(
        entity_type, text, start, end
    ) and not _own_label_adjacent(entity_type, text, start):
        return _negative_context_demotion(entity_type, base_confidence)
    return base_confidence


def adjust_confidence(
    entity_type: str,
    base_confidence: float,
    text: str,
    start: int,
    end: int,
) -> float:
    """Boost or penalize *base_confidence* based on surrounding context.

    The negative-context check runs FIRST: a foreign field label
    evidenced per the entity type's rule (RULE A span-excluded window
    scan / RULE B label-position evidence — see ``NEGATIVE_CONTEXT_WORDS``)
    with no own-type label adjacent demotes the finding, and the own-type
    boost is NOT applied — the adjacent foreign label owns the span.
    Otherwise the pre-existing boost/penalty logic applies unchanged.

    Parameters
    ----------
    entity_type:
        The PII entity type (e.g., ``"US_SSN"``).
    base_confidence:
        Raw confidence from pattern matching or validation.
    text:
        Full input text containing the match.
    start:
        Start offset of the matched span.
    end:
        End offset of the matched span.

    Returns
    -------
    float
        Adjusted confidence in [0.40, 0.99].
    """
    if _has_demotable_negative_context(
        entity_type, text, start, end
    ) and not _own_label_adjacent(entity_type, text, start):
        return _negative_context_demotion(entity_type, base_confidence)
    ctx = extract_context(text, start, end)
    if has_context_words(entity_type, ctx):
        return min(CONFIDENCE_CAP, base_confidence + CONTEXT_BOOST)
    if entity_type in HIGH_FP_TYPES:
        return max(CONFIDENCE_FLOOR, base_confidence - CONTEXT_PENALTY)
    return base_confidence
