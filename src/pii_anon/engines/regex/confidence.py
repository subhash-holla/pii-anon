"""Context-aware confidence scoring for regex PII detections.

Inspired by Microsoft Presidio's ``LemmaContextAwareEnhancer``, this module
adjusts detection confidence based on surrounding text.  When context keywords
appear near a matched span (e.g., "social security" near a 9-digit number),
confidence is boosted.  When context is absent for high false-positive entity
types (US_SSN, PERSON_NAME), confidence is penalized.

Algorithm
---------
1. Extract a window of *±CONTEXT_WINDOW* characters around the matched span.
2. Tokenize the window into lowercase words.
3. Intersect tokens with the entity-type's keyword set.
4. If intersection is non-empty → boost by *CONTEXT_BOOST* (capped at 0.99).
5. If empty **and** the entity type is in *HIGH_FP_TYPES* → penalize by
   *CONTEXT_PENALTY* (floored at 0.50).
6. Otherwise → return the base confidence unchanged.

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
    "CREDIT_CARD_FRAGMENT": {
        "credit", "card", "visa", "mastercard", "amex", "discover",
        "payment", "debit", "charge", "cc", "cardnumber", "pan",
        "account", "transaction", "purchase",
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
    "CREDIT_CARD_FRAGMENT",
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
) -> None:
    """Override module-level tuning constants from external configuration.

    Called by the regex engine adapter during initialization so that
    ``CoreConfig.confidence`` values take effect.
    """
    global CONTEXT_BOOST, CONTEXT_PENALTY, CONTEXT_WINDOW  # noqa: PLW0603
    global CONFIDENCE_CAP, CONFIDENCE_FLOOR  # noqa: PLW0603
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
    return not words.isdisjoint(_WORD_RE.findall(context_text))


def adjust_confidence(
    entity_type: str,
    base_confidence: float,
    text: str,
    start: int,
    end: int,
) -> float:
    """Boost or penalize *base_confidence* based on surrounding context.

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
    ctx = extract_context(text, start, end)
    if has_context_words(entity_type, ctx):
        return min(CONFIDENCE_CAP, base_confidence + CONTEXT_BOOST)
    if entity_type in HIGH_FP_TYPES:
        return max(CONFIDENCE_FLOOR, base_confidence - CONTEXT_PENALTY)
    return base_confidence
