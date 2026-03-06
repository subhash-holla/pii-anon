#!/usr/bin/env python3
"""Generate deterministic, comprehensive PII benchmark dataset (v1.0.0).

Produces a single unified JSONL dataset:

* **pii_anon_benchmark_v1.jsonl** – 50 000 records organized across 7 research-grade
  evaluation dimensions:

  1. Entity Consistency (20% weight): 2 000+ records testing consistent entity identity
     across aliases, coreference chains, and repeated mentions in lengthy documents.
  2. Multilingual Support (15% weight): 12 languages with 500+ samples per language,
     including locale-specific PII patterns and cross-language code-switching.
  3. Context Preservation (20% weight): 2 000+ conversational samples (chat threads,
     email chains, dialogue) testing whether anonymization preserves semantic flow.
  4. PII Type Coverage (20% weight): 22 entity types with 2 500+ boost records ensuring
     200+ samples per entity category including thin types (LICENSE_PLATE, IBAN, etc.).
  5. Edge Cases (10% weight): 2 000+ records with overlapping entities, false positive
     triggers, dense PII, Unicode contexts, and PII in URLs/code.
  6. Format Variations (10% weight): 1 500+ records in JSON, CSV, XML, ASCII table,
     email header, and code/config formats.
  7. Temporal Consistency (5% weight): 1 500+ longitudinal records with medical
     timelines and financial histories requiring temporal coherence preservation.

  Total composition: 35 700 core detection records + 2 800 entity tracking records
  + 11 500 evaluation dimension records = 50 000 records.

Design informed by:
  - PII-Rate-Elo composite evaluation framework (Bradley-Terry/Glicko rating)
  - ai4privacy/pii-masking-300k (27-47 entity classes, multilingual)
  - Microsoft PII-Bench (55 categories, multi-subject networks)
  - TAB Text Anonymization Benchmark (coreference, legal domain)
  - NVIDIA Nemotron-PII (100K production-quality records)
  - RAT-Bench (re-identification risk assessment)
  - TAU-EVAL (utility-privacy trade-off evaluation)
  - PrivaCI-Bench (contextual integrity, regulatory compliance)
  - PIILO (educational PII), SPY (medical/legal synthetic data)
  - PANORAMA (PII memorization in LLMs)

Every record is fully synthetic (CC0-1.0) or templated from public patterns
(CC-BY-4.0) and is guaranteed to contain zero real PII.

Usage:
    python scripts/generate_benchmark_dataset.py [--seed 42]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "packages" / "pii_anon_datasets" / "src" / "pii_anon_datasets" / "benchmarks" / "data"
UNIFIED_DATASET_FILE = DATA_DIR / "pii_anon_benchmark_v1.jsonl"
UNIFIED_METADATA_FILE = DATA_DIR / "pii_anon_benchmark_v1.metadata.json"

VERSION = "v1.0.0"

# ---------------------------------------------------------------------------
# Name pools — diverse, realistic, multi-cultural.  All synthetic.
# ---------------------------------------------------------------------------
FIRST_NAMES = {
    "en": [
        "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael",
        "Linda", "David", "Elizabeth", "William", "Barbara", "Richard", "Susan",
        "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Daniel",
        "Lisa", "Matthew", "Nancy", "Anthony", "Betty", "Mark", "Sandra",
        "Steven", "Ashley", "Andrew", "Dorothy", "Joshua", "Kimberly",
        "Kevin", "Emily", "Brian", "Donna", "George", "Michelle", "Timothy",
        "Carol", "Ronald", "Amanda", "Jason", "Melissa", "Edward", "Deborah",
        "Ryan", "Stephanie", "Jacob", "Rebecca", "Gary", "Sharon", "Nicholas",
        "Laura", "Eric", "Cynthia", "Jonathan", "Kathleen", "Larry", "Amy",
        "Justin", "Angela", "Scott", "Shirley", "Brandon", "Brenda", "Benjamin",
        "Emma", "Samuel", "Anna", "Raymond", "Pamela", "Gregory", "Nicole",
        "Frank", "Samantha", "Alexander", "Katherine", "Patrick", "Christine",
        "Jack", "Debra", "Dennis", "Rachel", "Jerry", "Carolyn", "Tyler",
        "Janet", "Aaron", "Catherine", "Jose", "Maria", "Adam", "Heather",
    ],
    "es": [
        "Carlos", "Ana", "Miguel", "Lucia", "Javier", "Carmen", "Diego",
        "Isabel", "Alejandro", "Sofia", "Fernando", "Elena", "Pablo",
        "Marta", "Raul", "Rosa", "Luis", "Pilar", "Jorge", "Dolores",
        "Manuel", "Teresa", "Antonio", "Cristina", "Francisco", "Laura",
        "Alberto", "Beatriz", "Pedro", "Patricia", "Rafael", "Alicia",
    ],
    "fr": [
        "Pierre", "Marie", "Jean", "Sophie", "Michel", "Isabelle", "Francois",
        "Nathalie", "Philippe", "Catherine", "Nicolas", "Sylvie", "Laurent",
        "Valerie", "Christophe", "Monique", "Stephane", "Veronique",
        "Guillaume", "Florence", "Olivier", "Sandrine", "Thierry", "Brigitte",
        "Antoine", "Helene", "Julien", "Martine", "Benoit", "Celine",
    ],
    "de": [
        "Hans", "Ursula", "Klaus", "Petra", "Stefan", "Monika", "Thomas",
        "Andrea", "Wolfgang", "Sabine", "Markus", "Claudia", "Juergen",
        "Karin", "Andreas", "Renate", "Peter", "Gabriele", "Bernd", "Heike",
    ],
    "it": [
        "Marco", "Giulia", "Luca", "Francesca", "Giovanni", "Chiara",
        "Alessandro", "Sara", "Andrea", "Valentina", "Matteo", "Silvia",
        "Giuseppe", "Laura", "Davide", "Federica", "Simone", "Eleonora",
    ],
    "pt": [
        "Joao", "Ana", "Pedro", "Maria", "Carlos", "Rita", "Miguel",
        "Sofia", "Antonio", "Teresa", "Paulo", "Catarina", "Rui", "Marta",
    ],
    "nl": [
        "Jan", "Maria", "Pieter", "Anna", "Willem", "Sophie", "Dirk",
        "Lotte", "Hendrik", "Emma", "Bas", "Fleur", "Jeroen", "Eva",
    ],
    "ja": [
        "Takeshi", "Yuki", "Hiroshi", "Sakura", "Kenji", "Aoi", "Daisuke",
        "Haruka", "Shota", "Miku", "Ryota", "Nanami", "Yuto", "Hina",
    ],
    "ar": [
        "Omar", "Layla", "Youssef", "Noura", "Khaled", "Mariam", "Hassan",
        "Salma", "Karim", "Rania", "Samir", "Dina", "Tariq", "Huda",
    ],
    "hi": [
        "Aarav", "Ananya", "Rohan", "Priya", "Vikram", "Neha", "Arjun",
        "Kavya", "Rahul", "Isha", "Nikhil", "Pooja", "Manish", "Sneha",
    ],
    "zh": [
        "Wei", "Li", "Jun", "Mei", "Hao", "Ling", "Tao", "Yan",
        "Lei", "Na", "Jie", "Fang", "Bo", "Xin",
    ],
    "ko": [
        "Minjun", "Seoyeon", "Jiho", "Soojin", "Hyunwoo", "Jiyoon", "Donghyun",
        "Eunji", "Taeyang", "Nari", "Jisoo", "Yuna", "Sungmin", "Haeun",
    ],
}

LAST_NAMES = {
    "en": [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
        "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
        "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
        "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
        "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
        "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz",
        "Parker", "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris",
        "Morales", "Murphy", "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan",
    ],
    "es": [
        "Lopez", "Garcia", "Martinez", "Rodriguez", "Hernandez", "Gonzalez",
        "Perez", "Sanchez", "Ramirez", "Torres", "Flores", "Rivera", "Gomez",
        "Diaz", "Cruz", "Reyes", "Morales", "Gutierrez", "Ortiz", "Ramos",
    ],
    "fr": [
        "Martin", "Dubois", "Dupont", "Moreau", "Laurent", "Simon", "Michel",
        "Lefevre", "Leroy", "Roux", "David", "Bertrand", "Robert", "Richard",
        "Petit", "Durand", "Garnier", "Bonnet", "Lambert", "Fontaine",
    ],
    "de": [
        "Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer",
        "Wagner", "Becker", "Schulz", "Hoffmann", "Schaefer", "Koch",
    ],
    "it": [
        "Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano",
        "Colombo", "Ricci", "Marino", "Greco", "Bruno", "Gallo",
    ],
    "pt": [
        "Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira",
        "Alves", "Pereira", "Lima", "Gomes", "Costa", "Ribeiro",
    ],
    "nl": [
        "de Vries", "van Dijk", "Jansen", "Bakker", "Visser", "Smit",
        "de Boer", "Mulder", "de Groot", "Bos", "Peters", "Hendriks",
    ],
    "ja": [
        "Tanaka", "Suzuki", "Takahashi", "Watanabe", "Ito", "Yamamoto",
        "Nakamura", "Kobayashi", "Saito", "Kato", "Yoshida", "Matsumoto",
    ],
    "ar": [
        "Haddad", "Nasser", "Khalil", "Farah", "Saad", "Rahman",
        "Sharif", "Mansour", "Aziz", "Salim", "Jabari", "Hamdan",
    ],
    "hi": [
        "Sharma", "Patel", "Singh", "Gupta", "Kumar", "Verma",
        "Joshi", "Mehta", "Rao", "Iyer", "Nair", "Malhotra",
    ],
    "zh": [
        "Wang", "Li", "Zhang", "Liu", "Chen", "Yang",
        "Huang", "Zhao", "Wu", "Zhou", "Xu", "Sun",
    ],
    "ko": [
        "Kim", "Lee", "Park", "Choi", "Jung", "Kang",
        "Cho", "Yoon", "Jang", "Im", "Han", "Shin",
    ],
}

HONORIFICS = {
    "en": ["Mr.", "Ms.", "Mrs.", "Dr.", "Prof."],
    "es": ["Sr.", "Sra.", "Dra.", "Prof."],
    "fr": ["M.", "Mme", "Dr", "Prof."],
    "de": ["Herr", "Frau", "Dr.", "Prof."],
    "it": ["Sig.", "Sig.ra", "Dott.", "Prof."],
    "pt": ["Sr.", "Sra.", "Dr.", "Prof."],
    "nl": ["Dhr.", "Mevr.", "Dr.", "Prof."],
    "ja": ["Mr.", "Ms.", "Dr.", "Prof."],
    "ar": ["Mr.", "Ms.", "Dr.", "Prof."],
    "hi": ["Mr.", "Ms.", "Dr.", "Prof."],
    "zh": ["Mr.", "Ms.", "Dr.", "Prof."],
    "ko": ["Mr.", "Ms.", "Dr.", "Prof."],
}

STREET_NAMES = [
    "Oak Street", "Main Avenue", "Elm Boulevard", "Cedar Lane", "Pine Road",
    "Maple Drive", "Birch Court", "Willow Way", "Spruce Circle", "Ash Place",
    "Walnut Terrace", "Cherry Lane", "Poplar Drive", "Hickory Boulevard",
    "Chestnut Avenue", "Sycamore Road", "Juniper Way", "Alder Lane",
    "Hawthorn Circle", "Dogwood Drive",
]

CITIES = [
    "Springfield", "Riverside", "Fairview", "Georgetown", "Madison",
    "Oakland", "Burlington", "Greenville", "Bristol", "Lexington",
    "Ashland", "Dover", "Milford", "Salem", "Franklin",
]

STATES = ["CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]

COMPANIES = [
    "Acme Corp", "Globex Industries", "Initech LLC", "Umbrella Group",
    "Soylent Inc", "Cyberdyne Systems", "Wayne Enterprises", "Stark Labs",
    "Oscorp Technologies", "Massive Dynamic", "Pied Piper Inc",
    "Hooli Technologies", "Dunder Mifflin", "Sterling Cooper",
    "Prestige Worldwide", "Vandelay Industries", "BlueStar Airlines",
    "Tyrell Corporation", "Aperture Science", "Weyland Industries",
]

MEDICAL_CONDITIONS = [
    "Type 2 Diabetes Mellitus", "Essential Hypertension", "Major Depressive Disorder",
    "Generalized Anxiety Disorder", "Chronic Obstructive Pulmonary Disease",
    "Rheumatoid Arthritis", "Atrial Fibrillation", "Chronic Kidney Disease Stage 3",
    "Hypothyroidism", "Osteoarthritis", "Systemic Lupus Erythematosus",
    "Irritable Bowel Syndrome", "Migraine with Aura", "Asthma, Moderate Persistent",
    "Bipolar Disorder Type I", "Post-Traumatic Stress Disorder",
]

MEDICATIONS = [
    "Metformin 500mg", "Lisinopril 10mg", "Atorvastatin 20mg",
    "Levothyroxine 50mcg", "Amlodipine 5mg", "Omeprazole 20mg",
    "Sertraline 100mg", "Metoprolol 25mg", "Losartan 50mg",
    "Albuterol 90mcg inhaler", "Gabapentin 300mg", "Prednisone 10mg",
    "Fluoxetine 20mg", "Hydrochlorothiazide 25mg", "Pantoprazole 40mg",
]

DRUG_CODES = [
    "NDC-0002-1433-01", "NDC-0069-0150-30", "NDC-0378-1805-01",
    "NDC-0591-3740-01", "NDC-0781-1506-01", "NDC-0904-5852-61",
]

EMAIL_DOMAINS = {
    "en": ["gmail.com", "outlook.com", "yahoo.com", "protonmail.com", "icloud.com"],
    "es": ["gmail.com", "outlook.es", "yahoo.es", "protonmail.com"],
    "fr": ["gmail.com", "outlook.fr", "orange.fr", "laposte.net"],
    "de": ["gmail.com", "web.de", "gmx.de", "outlook.de"],
    "it": ["gmail.com", "libero.it", "virgilio.it", "outlook.it"],
    "pt": ["gmail.com", "sapo.pt", "outlook.pt", "iol.pt"],
    "nl": ["gmail.com", "ziggo.nl", "kpnmail.nl", "outlook.nl"],
    "ja": ["gmail.com", "yahoo.co.jp", "outlook.jp", "icloud.com"],
    "ar": ["gmail.com", "outlook.com", "yahoo.com"],
    "hi": ["gmail.com", "outlook.com", "yahoo.com"],
    "zh": ["gmail.com", "outlook.com", "qq.com"],
    "ko": ["gmail.com", "outlook.com", "naver.com"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _det_hash(seed: int, *parts: str) -> str:
    """Deterministic short hash for reproducible but varied generation."""
    raw = f"{seed}|{'|'.join(parts)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _pick(rng: random.Random, seq: list) -> Any:
    return seq[rng.randint(0, len(seq) - 1)]


class _PersonaFactory:
    """Generates synthetic personas with consistent, linkable identity variants."""

    def __init__(self, rng: random.Random, language: str, index: int):
        self.rng = rng
        self.lang = language
        self.idx = index

        lang_first = FIRST_NAMES.get(language, FIRST_NAMES["en"])
        lang_last = LAST_NAMES.get(language, LAST_NAMES["en"])
        lang_honor = HONORIFICS.get(language, HONORIFICS["en"])

        self.first = lang_first[index % len(lang_first)]
        self.last = lang_last[index % len(lang_last)]
        self.honorific = lang_honor[index % len(lang_honor)]
        self.full_name = f"{self.first} {self.last}"
        self.formal_name = f"{self.honorific} {self.last}"
        self.first_last_initial = f"{self.first} {self.last[0]}."
        self.email_user = f"{self.first.lower()}.{self.last.lower()}{index % 100}"
        email_doms = EMAIL_DOMAINS.get(language, EMAIL_DOMAINS["en"])
        self.email = f"{self.email_user}@{email_doms[index % len(email_doms)]}"
        self.corporate_email = f"{self.first[0].lower()}{self.last.lower()}@{_pick(rng, COMPANIES).lower().replace(' ', '')}.com"

        # Phone
        area = 200 + (index % 800)
        mid = 200 + (index * 7 % 800)
        end = 1000 + (index * 13 % 9000)
        self.phone_us = f"+1 ({area}) {mid}-{end:04d}"
        self.phone_intl = f"+44 20 {7000 + index % 3000} {1000 + index % 9000}"

        # SSN / Government IDs
        a = 100 + (index % 900)
        b = 10 + (index % 90)
        c = 1000 + (index % 9000)
        self.ssn = f"{a:03d}-{b:02d}-{c:04d}"
        self.passport = f"P{index % 10}{chr(65 + index % 26)}{100000 + index:06d}"
        self.drivers_license = f"DL-{chr(65 + index % 26)}{10000 + index:05d}-{index % 100:02d}"
        self.national_id = f"NID-{900000000 + index * 7:09d}"

        # Address
        num = 100 + index % 9900
        self.street = f"{num} {STREET_NAMES[index % len(STREET_NAMES)]}"
        self.city = CITIES[index % len(CITIES)]
        self.state = STATES[index % len(STATES)]
        self.zip_code = f"{10000 + index % 90000}"
        self.full_address = f"{self.street}, {self.city}, {self.state} {self.zip_code}"

        # Financial
        cc_base = 4000000000000000 + index * 37
        self.credit_card = f"{cc_base:016d}"
        self.credit_card_formatted = f"{str(self.credit_card)[:4]}-{str(self.credit_card)[4:8]}-{str(self.credit_card)[8:12]}-{str(self.credit_card)[12:16]}"
        self.bank_account = f"{10000000 + index * 13:08d}"
        self.routing_number = f"{index % 9 + 1}{21000000 + index:08d}"
        self.iban = f"GB{10 + index % 90:02d}ABCD{60000000 + index * 11:08d}"

        # Date of birth
        year = 1950 + index % 55
        month = 1 + index % 12
        day = 1 + index % 28
        self.dob = f"{year:04d}-{month:02d}-{day:02d}"
        self.dob_us = f"{month:02d}/{day:02d}/{year:04d}"

        # Medical
        self.mrn = f"MRN-{1000000 + index:07d}"

        # Company / Employment
        self.company = COMPANIES[index % len(COMPANIES)]
        self.employee_id = f"EMP-{10000 + index:05d}"

        # IP / Digital
        self.ipv4 = f"{10 + index % 240}.{index % 256}.{(index * 3) % 256}.{(index * 7) % 256}"
        self.mac_address = ":".join(f"{(index * p) % 256:02x}" for p in [3, 5, 7, 11, 13, 17])
        self.username = f"{self.first.lower()}{self.last.lower()}{index % 1000}"

        # Vehicle
        letters = string.ascii_uppercase
        self.license_plate = f"{letters[index % 26]}{letters[(index * 3) % 26]}{letters[(index * 7) % 26]}-{1000 + index % 9000}"

        # Cluster ID for entity linking
        self.cluster_id = f"person-{index:05d}"


def _label(
    text: str,
    needle: str,
    entity_type: str,
    *,
    start_search: int = 0,
    cluster: str = "none",
    variant: str = "none",
    context_group: str = "baseline",
) -> dict[str, Any] | None:
    """Build a label dict.  Returns None if needle not found (safety guard)."""
    start = text.find(needle, start_search)
    if start < 0:
        return None
    return {
        "entity_type": entity_type,
        "start": start,
        "end": start + len(needle),
        "entity_cluster_id": cluster,
        "mention_variant": variant,
        "context_group": context_group,
    }


def _labels_for(text: str, items: list[tuple[str, str, dict[str, str]]]) -> list[dict[str, Any]]:
    """Build labels list from (needle, entity_type, kwargs) tuples. Skips misses."""
    out: list[dict[str, Any]] = []
    for needle, etype, kwargs in items:
        lbl = _label(text, needle, etype, **kwargs)
        if lbl is not None:
            out.append(lbl)
    return out


# ---------------------------------------------------------------------------
# TEXT TEMPLATES — organised by SIZE TIER and SCENARIO CATEGORY
# ---------------------------------------------------------------------------

# ---- SIZE TIER: SMALL (< 100 chars) ----

def _small_simple_contact(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = f"Contact {p.full_name} at {p.email}."
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.email, "EMAIL_ADDRESS", {}),
    ])
    return text, labels


def _small_phone_ssn(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = f"Call {p.phone_us}, SSN {p.ssn}."
    labels = _labels_for(text, [
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.ssn, "US_SSN", {}),
    ])
    return text, labels


def _small_id_only(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = f"Passport: {p.passport}; DL: {p.drivers_license}."
    labels = _labels_for(text, [
        (p.passport, "PASSPORT", {}),
        (p.drivers_license, "DRIVERS_LICENSE", {}),
    ])
    return text, labels


def _small_financial_alert(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = f"Card ending {p.credit_card_formatted[-4:]} charged. Account {p.bank_account}."
    labels = _labels_for(text, [
        (p.credit_card_formatted[-4:], "CREDIT_CARD_FRAGMENT", {}),
        (p.bank_account, "BANK_ACCOUNT", {}),
    ])
    return text, labels


def _small_log_line(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = f"[INFO] User {p.username} logged in from {p.ipv4} at 2025-03-15T10:30:00Z."
    labels = _labels_for(text, [
        (p.username, "USERNAME", {}),
        (p.ipv4, "IP_ADDRESS", {}),
    ])
    return text, labels


def _small_dob_address(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = f"DOB: {p.dob_us}. Addr: {p.street}, {p.city} {p.state}."
    labels = _labels_for(text, [
        (p.dob_us, "DATE_OF_BIRTH", {}),
        (p.street, "ADDRESS", {}),
        (p.city, "LOCATION", {}),
    ])
    return text, labels


# ---- SIZE TIER: MEDIUM (100-300 chars) ----

def _medium_patient_note(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    cond = MEDICAL_CONDITIONS[p.idx % len(MEDICAL_CONDITIONS)]
    med = MEDICATIONS[p.idx % len(MEDICATIONS)]
    text = (
        f"Patient {p.full_name} (MRN: {p.mrn}, DOB: {p.dob}) presents with {cond}. "
        f"Current medication: {med}. Contact: {p.phone_us}. "
        f"Emergency contact at {p.email}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {}),
    ])
    return text, labels


def _medium_employee_record(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = (
        f"Employee {p.full_name} (ID: {p.employee_id}) at {p.company}. "
        f"SSN: {p.ssn}. Address: {p.full_address}. "
        f"Email: {p.corporate_email}. Phone: {p.phone_us}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.employee_id, "EMPLOYEE_ID", {}),
        (p.company, "ORGANIZATION", {}),
        (p.ssn, "US_SSN", {}),
        (p.full_address, "ADDRESS", {}),
        (p.corporate_email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
    ])
    return text, labels


def _medium_financial_transaction(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = (
        f"Wire transfer initiated by {p.full_name}. "
        f"Source account: {p.bank_account}, routing: {p.routing_number}. "
        f"IBAN destination: {p.iban}. Card on file: {p.credit_card_formatted}. "
        f"Verification sent to {p.email}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.bank_account, "BANK_ACCOUNT", {}),
        (p.routing_number, "ROUTING_NUMBER", {}),
        (p.iban, "IBAN", {}),
        (p.credit_card_formatted, "CREDIT_CARD", {}),
        (p.email, "EMAIL_ADDRESS", {}),
    ])
    return text, labels


def _medium_legal_notice(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = (
        f"Re: Case #{20000 + p.idx}. Defendant {p.formal_name} ({p.full_name}), "
        f"residing at {p.full_address}, is hereby notified. "
        f"Passport: {p.passport}. Counsel may reach the defendant at {p.phone_us} or {p.email}."
    )
    labels = _labels_for(text, [
        (p.formal_name, "PERSON_NAME", {}),
        (p.full_name, "PERSON_NAME", {}),
        (p.full_address, "ADDRESS", {}),
        (p.passport, "PASSPORT", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {}),
    ])
    return text, labels


def _medium_structured_form(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    text = (
        f"Name: {p.full_name}\n"
        f"Date of Birth: {p.dob_us}\n"
        f"SSN: {p.ssn}\n"
        f"Address: {p.full_address}\n"
        f"Phone: {p.phone_us}\n"
        f"Email: {p.email}\n"
        f"Employer: {p.company}\n"
        f"Employee ID: {p.employee_id}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.dob_us, "DATE_OF_BIRTH", {}),
        (p.ssn, "US_SSN", {}),
        (p.full_address, "ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.company, "ORGANIZATION", {}),
        (p.employee_id, "EMPLOYEE_ID", {}),
    ])
    return text, labels


def _medium_insurance_claim(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    policy = f"POL-{100000 + p.idx:06d}"
    claim = f"CLM-{200000 + p.idx:06d}"
    text = (
        f"Insurance claim {claim} filed by {p.full_name} under policy {policy}. "
        f"Policyholder DOB: {p.dob}, SSN: {p.ssn}. "
        f"Mailing address: {p.full_address}. Phone: {p.phone_us}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.ssn, "US_SSN", {}),
        (p.full_address, "ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
    ])
    return text, labels


# ---- SIZE TIER: LARGE (300-800 chars) ----

def _large_medical_discharge(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    cond = MEDICAL_CONDITIONS[p.idx % len(MEDICAL_CONDITIONS)]
    med1 = MEDICATIONS[p.idx % len(MEDICATIONS)]
    med2 = MEDICATIONS[(p.idx + 3) % len(MEDICATIONS)]
    text = (
        f"Discharge Summary — Patient: {p.full_name}, MRN: {p.mrn}, DOB: {p.dob}.\n\n"
        f"Admitted on 2025-01-{1 + p.idx % 28:02d} for management of {cond}. "
        f"During hospitalization, {p.formal_name} received {med1} and {med2}. "
        f"Vital signs stabilized by day three. Labs showed improvement in all markers.\n\n"
        f"The patient, {p.first}, was counselled on lifestyle modifications and medication adherence. "
        f"Follow-up appointment scheduled with Dr. {LAST_NAMES['en'][(p.idx + 5) % len(LAST_NAMES['en'])]} "
        f"at the outpatient clinic.\n\n"
        f"Contact: {p.phone_us}. Email: {p.email}. "
        f"Next of kin: {FIRST_NAMES['en'][(p.idx + 10) % len(FIRST_NAMES['en'])]} {p.last} "
        f"at {p.phone_intl}."
    )
    kin_first = FIRST_NAMES['en'][(p.idx + 10) % len(FIRST_NAMES['en'])]
    kin_name = f"{kin_first} {p.last}"
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "medical"}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "medical"}),
        (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "medical"}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (kin_name, "PERSON_NAME", {}),
        (p.phone_intl, "PHONE_NUMBER", {}),
    ])
    return text, labels


def _large_legal_deposition(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    witness_first = FIRST_NAMES["en"][(p.idx + 7) % len(FIRST_NAMES["en"])]
    witness_last = LAST_NAMES["en"][(p.idx + 7) % len(LAST_NAMES["en"])]
    witness_full = f"{witness_first} {witness_last}"
    witness_email = f"{witness_first.lower()}.{witness_last.lower()}@{_pick(p.rng, EMAIL_DOMAINS['en'])}"
    text = (
        f"DEPOSITION TRANSCRIPT — Case No. {30000 + p.idx}\n\n"
        f"Deponent: {p.full_name}, residing at {p.full_address}.\n"
        f"Represented by counsel, {p.formal_name} testified as follows:\n\n"
        f"Q: Please state your full name for the record.\n"
        f'A: My name is {p.full_name}. Some colleagues call me {p.first}, and my email '
        f"handle is {p.email_user}.\n\n"
        f"Q: What is your date of birth?\nA: {p.dob_us}.\n\n"
        f"Q: And your social security number?\nA: {p.ssn}.\n\n"
        f"Q: Can you confirm your contact information?\n"
        f"A: Yes, my phone is {p.phone_us} and my email is {p.email}.\n\n"
        f"The witness, {witness_full}, corroborated the account. "
        f"Witness can be reached at {witness_email}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "legal"}),
        (p.full_address, "ADDRESS", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "legal"}),
        (p.email_user, "USERNAME", {"cluster": p.cluster_id, "variant": "username", "context_group": "legal"}),
        (p.dob_us, "DATE_OF_BIRTH", {}),
        (p.ssn, "US_SSN", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "legal"}),
        (witness_full, "PERSON_NAME", {}),
        (witness_email, "EMAIL_ADDRESS", {}),
    ])
    # Also capture second occurrence of full_name in Q&A
    second = _label(text, p.full_name, "PERSON_NAME",
                    start_search=text.find(p.full_name) + len(p.full_name),
                    cluster=p.cluster_id, variant="full_name", context_group="legal")
    if second:
        labels.append(second)
    # Capture first name mention in Q&A
    first_mention = _label(text, f"call me {p.first}",
                           "PERSON_NAME",  # We label just the first-name portion
                           cluster=p.cluster_id, variant="first_name", context_group="legal")
    # Replace with exact first-name span
    if first_mention:
        offset = text.find(f"call me {p.first}")
        if offset >= 0:
            real_start = offset + len("call me ")
            labels.append({
                "entity_type": "PERSON_NAME",
                "start": real_start,
                "end": real_start + len(p.first),
                "entity_cluster_id": p.cluster_id,
                "mention_variant": "first_name",
                "context_group": "legal",
            })
    return text, labels


def _large_customer_support_thread(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    agent_first = FIRST_NAMES["en"][(p.idx + 15) % len(FIRST_NAMES["en"])]
    agent_last = LAST_NAMES["en"][(p.idx + 15) % len(LAST_NAMES["en"])]
    agent_name = f"{agent_first} {agent_last}"
    text = (
        f"Support Ticket #{50000 + p.idx}\n\n"
        f"Customer: {p.full_name}\nEmail: {p.email}\nPhone: {p.phone_us}\n"
        f"Account: {p.bank_account}\n\n"
        f"[{p.first}]: Hi, I noticed an unauthorized charge on my card ending "
        f"{p.credit_card_formatted[-4:]}. My address on file is {p.full_address}. "
        f"Can you help?\n\n"
        f"[Agent {agent_name}]: Hello {p.formal_name}, I can see your account. "
        f"I've verified your identity using the SSN ending {p.ssn[-4:]}. "
        f"Let me look into the transaction on card {p.credit_card_formatted}.\n\n"
        f"[{p.first}]: Thank you. My employee ID is {p.employee_id} in case "
        f"you need to cross-reference with the corporate account.\n\n"
        f"[Agent {agent_name}]: I've flagged the transaction. A confirmation "
        f"will be sent to {p.email}. Is there anything else, {p.first}?"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "support"}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "support"}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.bank_account, "BANK_ACCOUNT", {}),
        (p.full_address, "ADDRESS", {}),
        (p.credit_card_formatted, "CREDIT_CARD", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "support"}),
        (p.employee_id, "EMPLOYEE_ID", {}),
        (agent_name, "PERSON_NAME", {}),
    ])
    # Capture first-name mentions (multiple occurrences)
    search_from = 0
    first_occurrences = 0
    while first_occurrences < 5:
        lbl = _label(text, f"[{p.first}]", "PERSON_NAME", start_search=search_from,
                      cluster=p.cluster_id, variant="first_name_bracket", context_group="support")
        if lbl is None:
            break
        # Adjust to just the name inside brackets
        lbl["start"] += 1
        lbl["end"] -= 1
        labels.append(lbl)
        search_from = lbl["end"] + 1
        first_occurrences += 1
    # Second email occurrence
    second_email = _label(text, p.email, "EMAIL_ADDRESS",
                          start_search=text.find(p.email) + len(p.email),
                          cluster=p.cluster_id, variant="email", context_group="support")
    if second_email:
        labels.append(second_email)
    # Second agent name
    second_agent = _label(text, agent_name, "PERSON_NAME",
                          start_search=text.find(agent_name) + len(agent_name))
    if second_agent:
        labels.append(second_agent)
    # Last first-name mention by agent
    last_first = _label(text, f", {p.first}?", "PERSON_NAME", cluster=p.cluster_id,
                         variant="first_name", context_group="support")
    if last_first:
        last_first["start"] += 2  # skip ", "
        last_first["end"] -= 1   # remove "?"
        labels.append(last_first)
    return text, labels


def _large_hr_investigation(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    manager_first = FIRST_NAMES["en"][(p.idx + 20) % len(FIRST_NAMES["en"])]
    manager_last = LAST_NAMES["en"][(p.idx + 20) % len(LAST_NAMES["en"])]
    manager_name = f"{manager_first} {manager_last}"
    manager_email = f"{manager_first.lower()}.{manager_last.lower()}@{p.company.lower().replace(' ', '')}.com"
    text = (
        f"CONFIDENTIAL — HR Investigation Report\n\n"
        f"Subject: {p.full_name} (Employee ID: {p.employee_id})\n"
        f"Department: Engineering at {p.company}\n"
        f"Manager: {manager_name} ({manager_email})\n\n"
        f"On 2025-02-15, {p.formal_name} filed a complaint regarding workplace conduct. "
        f"During the interview, {p.first} provided detailed accounts of the incidents. "
        f"Records show {p.first_last_initial} has been employed since 2019.\n\n"
        f"Contact details verified:\n"
        f"- SSN: {p.ssn}\n"
        f"- Personal email: {p.email}\n"
        f"- Work email: {p.corporate_email}\n"
        f"- Home address: {p.full_address}\n"
        f"- Phone: {p.phone_us}\n"
        f"- DOB: {p.dob}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "hr"}),
        (p.employee_id, "EMPLOYEE_ID", {}),
        (p.company, "ORGANIZATION", {}),
        (manager_name, "PERSON_NAME", {}),
        (manager_email, "EMAIL_ADDRESS", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "hr"}),
        (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "hr"}),
        (p.first_last_initial, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_last_initial", "context_group": "hr"}),
        (p.ssn, "US_SSN", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "personal_email", "context_group": "hr"}),
        (p.corporate_email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "work_email", "context_group": "hr"}),
        (p.full_address, "ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
    ])
    return text, labels


# ---- SIZE TIER: VERY LARGE (800+ chars) ----

def _vlarge_multi_party_medical(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """Multi-party medical scenario: patient, doctor, nurse, next-of-kin, pharmacist."""
    doc_idx = p.idx + 30
    nurse_idx = p.idx + 40
    kin_idx = p.idx + 50
    pharm_idx = p.idx + 60

    doc = _PersonaFactory(rng, "en", doc_idx)
    nurse = _PersonaFactory(rng, "en", nurse_idx)
    kin = _PersonaFactory(rng, "en", kin_idx)
    pharm = _PersonaFactory(rng, "en", pharm_idx)

    cond = MEDICAL_CONDITIONS[p.idx % len(MEDICAL_CONDITIONS)]
    med1 = MEDICATIONS[p.idx % len(MEDICATIONS)]
    med2 = MEDICATIONS[(p.idx + 5) % len(MEDICATIONS)]
    drug_code = DRUG_CODES[p.idx % len(DRUG_CODES)]

    text = (
        f"COMPREHENSIVE MEDICAL RECORD\n"
        f"{'=' * 40}\n\n"
        f"Patient: {p.full_name}\n"
        f"MRN: {p.mrn} | DOB: {p.dob} | SSN: {p.ssn}\n"
        f"Address: {p.full_address}\n"
        f"Phone: {p.phone_us} | Email: {p.email}\n"
        f"Insurance ID: {p.national_id}\n\n"
        f"ATTENDING PHYSICIAN: Dr. {doc.full_name}\n"
        f"  License: {doc.drivers_license} | Phone: {doc.phone_us}\n"
        f"  Email: {doc.corporate_email}\n\n"
        f"NURSING STAFF: {nurse.full_name}\n"
        f"  Employee ID: {nurse.employee_id} | Phone: {nurse.phone_us}\n\n"
        f"ADMISSION NOTES (2025-01-{1 + p.idx % 28:02d}):\n"
        f"{p.formal_name} presented to the emergency department with acute exacerbation of {cond}. "
        f"The patient, {p.first}, reported worsening symptoms over the past week. "
        f"Dr. {doc.last} ordered immediate labs and imaging. Nurse {nurse.first} "
        f"administered initial treatment per protocol.\n\n"
        f"TREATMENT PLAN:\n"
        f"- {med1} ({drug_code}) — twice daily\n"
        f"- {med2} — once daily at bedtime\n"
        f"- Follow-up with Dr. {doc.full_name} in 2 weeks\n\n"
        f"DISCHARGE PLANNING:\n"
        f"Emergency contact: {kin.full_name} (relationship: spouse)\n"
        f"  Phone: {kin.phone_us} | Email: {kin.email}\n"
        f"  Address: {kin.full_address}\n\n"
        f"Prescriptions sent to: {pharm.full_name} at {pharm.company} Pharmacy\n"
        f"  Phone: {pharm.phone_us}\n\n"
        f"Patient {p.last} acknowledged understanding of discharge instructions. "
        f"Signed by {p.full_name} on 2025-01-{5 + p.idx % 24:02d}."
    )

    labels = _labels_for(text, [
        # Patient — multiple variants
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "medical_multiparty"}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.ssn, "US_SSN", {}),
        (p.full_address, "ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "medical_multiparty"}),
        (p.national_id, "NATIONAL_ID", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "medical_multiparty"}),
        # Doctor
        (doc.full_name, "PERSON_NAME", {"cluster": doc.cluster_id, "variant": "full_name", "context_group": "medical_multiparty"}),
        (doc.drivers_license, "DRIVERS_LICENSE", {}),
        (doc.phone_us, "PHONE_NUMBER", {}),
        (doc.corporate_email, "EMAIL_ADDRESS", {}),
        # Nurse
        (nurse.full_name, "PERSON_NAME", {}),
        (nurse.employee_id, "EMPLOYEE_ID", {}),
        (nurse.phone_us, "PHONE_NUMBER", {}),
        # Kin
        (kin.full_name, "PERSON_NAME", {}),
        (kin.phone_us, "PHONE_NUMBER", {}),
        (kin.email, "EMAIL_ADDRESS", {}),
        (kin.full_address, "ADDRESS", {}),
        # Pharmacist
        (pharm.full_name, "PERSON_NAME", {}),
        (pharm.phone_us, "PHONE_NUMBER", {}),
    ])
    # Patient first-name mention
    fn_lbl = _label(text, f"patient, {p.first},", "PERSON_NAME",
                     cluster=p.cluster_id, variant="first_name",
                     context_group="medical_multiparty")
    if fn_lbl:
        fn_lbl["start"] += len("patient, ")
        fn_lbl["end"] = fn_lbl["start"] + len(p.first)
        labels.append(fn_lbl)
    # Patient last-name mention
    ln_lbl = _label(text, f"Patient {p.last} acknowledged", "PERSON_NAME",
                     cluster=p.cluster_id, variant="last_name",
                     context_group="medical_multiparty")
    if ln_lbl:
        ln_lbl["start"] += len("Patient ")
        ln_lbl["end"] = ln_lbl["start"] + len(p.last)
        labels.append(ln_lbl)
    # Second full name at signature
    sig = _label(text, p.full_name, "PERSON_NAME",
                  start_search=text.rfind(p.full_name),
                  cluster=p.cluster_id, variant="full_name",
                  context_group="medical_multiparty")
    if sig and sig["start"] != labels[0]["start"]:
        labels.append(sig)
    # Doctor last name mention
    doc_last = _label(text, f"Dr. {doc.last} ordered", "PERSON_NAME",
                       cluster=doc.cluster_id, variant="last_name",
                       context_group="medical_multiparty")
    if doc_last:
        doc_last["end"] = doc_last["start"] + len(f"Dr. {doc.last}")
        labels.append(doc_last)
    # Second doc full name
    doc2 = _label(text, f"Dr. {doc.full_name} in", "PERSON_NAME",
                   cluster=doc.cluster_id, variant="full_name_titled",
                   context_group="medical_multiparty")
    if doc2:
        doc2["end"] = doc2["start"] + len(f"Dr. {doc.full_name}")
        labels.append(doc2)
    # Nurse first name
    nurse_fn = _label(text, f"Nurse {nurse.first}", "PERSON_NAME")
    if nurse_fn:
        nurse_fn["start"] += len("Nurse ")
        nurse_fn["end"] = nurse_fn["start"] + len(nurse.first)
        labels.append(nurse_fn)
    return text, labels


def _vlarge_financial_audit(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """Multi-entity financial audit with repeated PII references."""
    cfo_idx = p.idx + 25
    cfo = _PersonaFactory(rng, "en", cfo_idx)
    auditor_idx = p.idx + 35
    auditor = _PersonaFactory(rng, "en", auditor_idx)

    text = (
        f"CONFIDENTIAL — FINANCIAL AUDIT REPORT\n"
        f"{'=' * 40}\n\n"
        f"Audit Period: Q1-Q3 2025\n"
        f"Entity: {p.company}\n"
        f"Prepared by: {auditor.full_name}, CPA (License: {auditor.drivers_license})\n"
        f"  Contact: {auditor.email} | {auditor.phone_us}\n\n"
        f"EXECUTIVE SUMMARY:\n"
        f"During the audit of {p.company}, irregularities were identified in "
        f"accounts managed by {p.full_name} (Employee ID: {p.employee_id}). "
        f"CFO {cfo.full_name} was notified on 2025-07-15.\n\n"
        f"FINDINGS:\n\n"
        f"1. Account #{p.bank_account} (routing: {p.routing_number}) showed "
        f"unexplained transfers totaling $47,250 to IBAN {p.iban}.\n\n"
        f"2. Corporate card {p.credit_card_formatted} assigned to {p.formal_name} "
        f"had 12 transactions flagged for review.\n\n"
        f"3. Expense reports submitted by {p.first} contained duplicate receipts "
        f"referencing vendor payments to account {cfo.bank_account}.\n\n"
        f"PERSONNEL INVOLVED:\n\n"
        f"Primary Subject: {p.full_name}\n"
        f"  SSN: {p.ssn} | DOB: {p.dob}\n"
        f"  Address: {p.full_address}\n"
        f"  Email: {p.email} | Work: {p.corporate_email}\n"
        f"  Phone: {p.phone_us}\n\n"
        f"CFO: {cfo.full_name}\n"
        f"  Employee ID: {cfo.employee_id}\n"
        f"  Email: {cfo.corporate_email}\n"
        f"  Phone: {cfo.phone_us}\n\n"
        f"This report was reviewed by {auditor.formal_name} and transmitted "
        f"to {cfo.formal_name} via secure channel. {p.last} has been placed "
        f"on administrative leave pending investigation."
    )

    labels = _labels_for(text, [
        (p.company, "ORGANIZATION", {}),
        (auditor.full_name, "PERSON_NAME", {}),
        (auditor.drivers_license, "DRIVERS_LICENSE", {}),
        (auditor.email, "EMAIL_ADDRESS", {}),
        (auditor.phone_us, "PHONE_NUMBER", {}),
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "financial_audit"}),
        (p.employee_id, "EMPLOYEE_ID", {}),
        (cfo.full_name, "PERSON_NAME", {"cluster": cfo.cluster_id, "variant": "full_name", "context_group": "financial_audit"}),
        (p.bank_account, "BANK_ACCOUNT", {}),
        (p.routing_number, "ROUTING_NUMBER", {}),
        (p.iban, "IBAN", {}),
        (p.credit_card_formatted, "CREDIT_CARD", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "financial_audit"}),
        (p.ssn, "US_SSN", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.full_address, "ADDRESS", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "personal_email", "context_group": "financial_audit"}),
        (p.corporate_email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "work_email", "context_group": "financial_audit"}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (cfo.employee_id, "EMPLOYEE_ID", {}),
        (cfo.corporate_email, "EMAIL_ADDRESS", {}),
        (cfo.phone_us, "PHONE_NUMBER", {}),
        (auditor.formal_name, "PERSON_NAME", {}),
        (cfo.formal_name, "PERSON_NAME", {"cluster": cfo.cluster_id, "variant": "formal", "context_group": "financial_audit"}),
    ])
    # Second full name occurrence
    fn2 = _label(text, p.full_name, "PERSON_NAME",
                  start_search=text.find(p.full_name) + len(p.full_name),
                  cluster=p.cluster_id, variant="full_name", context_group="financial_audit")
    if fn2:
        labels.append(fn2)
    # First name mention
    first_lbl = _label(text, f"by {p.first} contained", "PERSON_NAME",
                         cluster=p.cluster_id, variant="first_name", context_group="financial_audit")
    if first_lbl:
        first_lbl["start"] += len("by ")
        first_lbl["end"] = first_lbl["start"] + len(p.first)
        labels.append(first_lbl)
    # Last name at end
    last_lbl = _label(text, f"{p.last} has been placed", "PERSON_NAME",
                        cluster=p.cluster_id, variant="last_name", context_group="financial_audit")
    if last_lbl:
        last_lbl["end"] = last_lbl["start"] + len(p.last)
        labels.append(last_lbl)
    # Second CFO full name mention
    cfo2 = _label(text, cfo.full_name, "PERSON_NAME",
                   start_search=text.find(cfo.full_name) + len(cfo.full_name),
                   cluster=cfo.cluster_id, variant="full_name", context_group="financial_audit")
    if cfo2:
        labels.append(cfo2)
    # Second company mention
    co2 = _label(text, p.company, "ORGANIZATION",
                  start_search=text.find(p.company) + len(p.company))
    if co2:
        labels.append(co2)
    return text, labels


def _vlarge_immigration_case(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """Immigration / multi-jurisdiction case with passport, national ID, addresses in multiple countries."""
    attorney_idx = p.idx + 45
    attorney = _PersonaFactory(rng, "en", attorney_idx)
    sponsor_idx = p.idx + 55
    sponsor = _PersonaFactory(rng, "en", sponsor_idx)

    text = (
        f"IMMIGRATION CASE FILE — Ref: IMM-{70000 + p.idx}\n"
        f"{'=' * 40}\n\n"
        f"Applicant: {p.full_name}\n"
        f"  Passport: {p.passport}\n"
        f"  National ID: {p.national_id}\n"
        f"  Date of Birth: {p.dob}\n"
        f"  Current Address: {p.full_address}\n"
        f"  Phone: {p.phone_us} | Intl: {p.phone_intl}\n"
        f"  Email: {p.email}\n\n"
        f"ATTORNEY OF RECORD:\n"
        f"  {attorney.full_name}, Esq.\n"
        f"  Bar License: {attorney.drivers_license}\n"
        f"  Office: {attorney.full_address}\n"
        f"  Phone: {attorney.phone_us} | Email: {attorney.email}\n\n"
        f"SPONSOR INFORMATION:\n"
        f"  Name: {sponsor.full_name}\n"
        f"  SSN: {sponsor.ssn}\n"
        f"  Employer: {sponsor.company} (ID: {sponsor.employee_id})\n"
        f"  Address: {sponsor.full_address}\n"
        f"  Phone: {sponsor.phone_us} | Email: {sponsor.email}\n\n"
        f"CASE NOTES:\n"
        f"Applicant {p.formal_name} (also known as {p.first}) filed Form I-130 "
        f"on 2025-03-01. The petition was sponsored by {sponsor.formal_name}, "
        f"who has been employed at {sponsor.company} for 8 years.\n\n"
        f"During the interview, {p.last} provided biometric data and confirmed "
        f"identity via passport {p.passport}. Attorney {attorney.last} submitted "
        f"supporting documentation including tax returns for SSN {p.ssn}.\n\n"
        f"Vehicle registration: {p.license_plate} ({p.state})\n"
        f"IP of online submission: {p.ipv4}\n"
        f"MAC address of biometric device: {p.mac_address}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "immigration"}),
        (p.passport, "PASSPORT", {}),
        (p.national_id, "NATIONAL_ID", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.full_address, "ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.phone_intl, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "immigration"}),
        (attorney.full_name, "PERSON_NAME", {}),
        (attorney.drivers_license, "DRIVERS_LICENSE", {}),
        (attorney.full_address, "ADDRESS", {}),
        (attorney.phone_us, "PHONE_NUMBER", {}),
        (attorney.email, "EMAIL_ADDRESS", {}),
        (sponsor.full_name, "PERSON_NAME", {"cluster": sponsor.cluster_id, "variant": "full_name", "context_group": "immigration"}),
        (sponsor.ssn, "US_SSN", {}),
        (sponsor.company, "ORGANIZATION", {}),
        (sponsor.employee_id, "EMPLOYEE_ID", {}),
        (sponsor.full_address, "ADDRESS", {}),
        (sponsor.phone_us, "PHONE_NUMBER", {}),
        (sponsor.email, "EMAIL_ADDRESS", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "immigration"}),
        (sponsor.formal_name, "PERSON_NAME", {"cluster": sponsor.cluster_id, "variant": "formal", "context_group": "immigration"}),
        (p.license_plate, "LICENSE_PLATE", {}),
        (p.ipv4, "IP_ADDRESS", {}),
        (p.mac_address, "MAC_ADDRESS", {}),
        (p.ssn, "US_SSN", {}),
    ])
    # Re-mentions
    p_last = _label(text, f"{p.last} provided biometric", "PERSON_NAME",
                     cluster=p.cluster_id, variant="last_name", context_group="immigration")
    if p_last:
        p_last["end"] = p_last["start"] + len(p.last)
        labels.append(p_last)
    p_first = _label(text, f"known as {p.first})", "PERSON_NAME",
                      cluster=p.cluster_id, variant="first_name", context_group="immigration")
    if p_first:
        p_first["start"] += len("known as ")
        p_first["end"] = p_first["start"] + len(p.first)
        labels.append(p_first)
    # Second passport mention
    pp2 = _label(text, p.passport, "PASSPORT",
                  start_search=text.find(p.passport) + len(p.passport))
    if pp2:
        labels.append(pp2)
    # Attorney last name
    att_last = _label(text, f"Attorney {attorney.last} submitted", "PERSON_NAME")
    if att_last:
        att_last["start"] += len("Attorney ")
        att_last["end"] = att_last["start"] + len(attorney.last)
        labels.append(att_last)
    # Second sponsor company
    co2 = _label(text, sponsor.company, "ORGANIZATION",
                  start_search=text.find(sponsor.company) + len(sponsor.company))
    if co2:
        labels.append(co2)
    return text, labels


# ---- LLM CONTEXT-LOSS SCENARIOS ----

def _scenario_context_loss_pronoun(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Traditional anonymization replaces names but leaves pronouns dangling,
    breaking coreference for an LLM that needs to understand who 'he/she' refers to."""
    text = (
        f"{p.full_name} submitted the quarterly report on 2025-04-01. "
        f"The report contained projections that concerned the board. "
        f"When asked for clarification, {p.formal_name} explained that the figures "
        f"reflected seasonal adjustments. The board noted that {p.first} had flagged "
        f"similar patterns last year. They asked {p.first_last_initial} to present "
        f"at the next meeting. Contact: {p.email}, {p.phone_us}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "context_loss_pronoun"}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "context_loss_pronoun"}),
        (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "context_loss_pronoun"}),
        (p.first_last_initial, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_last_initial", "context_group": "context_loss_pronoun"}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "context_loss_pronoun"}),
        (p.phone_us, "PHONE_NUMBER", {}),
    ])
    return text, labels


def _scenario_context_loss_relational(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """Relational PII where anonymizing one entity breaks the semantic link to another."""
    colleague_idx = p.idx + 12
    colleague = _PersonaFactory(rng, "en", colleague_idx)
    text = (
        f"{p.full_name} and {colleague.full_name} co-authored the patent application. "
        f"The lead inventor, {p.formal_name}, contributed the algorithm design while "
        f"{colleague.first} handled the hardware implementation. "
        f"Their shared lab at {p.company} (employee IDs {p.employee_id} and "
        f"{colleague.employee_id}) was where the breakthrough occurred. "
        f"Correspondence went to {p.email} and {colleague.email}. "
        f"The patent lists {p.full_name} as primary and {colleague.full_name} as secondary inventor."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "context_loss_relational"}),
        (colleague.full_name, "PERSON_NAME", {"cluster": colleague.cluster_id, "variant": "full_name", "context_group": "context_loss_relational"}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "context_loss_relational"}),
        (colleague.first, "PERSON_NAME", {"cluster": colleague.cluster_id, "variant": "first_name", "context_group": "context_loss_relational"}),
        (p.company, "ORGANIZATION", {}),
        (p.employee_id, "EMPLOYEE_ID", {}),
        (colleague.employee_id, "EMPLOYEE_ID", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "context_loss_relational"}),
        (colleague.email, "EMAIL_ADDRESS", {"cluster": colleague.cluster_id, "variant": "email", "context_group": "context_loss_relational"}),
    ])
    # Second mentions of full names
    fn2 = _label(text, p.full_name, "PERSON_NAME",
                  start_search=text.find(p.full_name) + len(p.full_name),
                  cluster=p.cluster_id, variant="full_name", context_group="context_loss_relational")
    if fn2:
        labels.append(fn2)
    cfn2 = _label(text, colleague.full_name, "PERSON_NAME",
                   start_search=text.find(colleague.full_name) + len(colleague.full_name),
                   cluster=colleague.cluster_id, variant="full_name", context_group="context_loss_relational")
    if cfn2:
        labels.append(cfn2)
    return text, labels


def _scenario_context_loss_temporal(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Temporal PII where dates are semantically linked to the person. Random date
    replacement destroys the medical timeline coherence."""
    cond = MEDICAL_CONDITIONS[p.idx % len(MEDICAL_CONDITIONS)]
    text = (
        f"Patient {p.full_name} (DOB: {p.dob}, MRN: {p.mrn}) was first diagnosed "
        f"with {cond} on 2022-06-{1 + p.idx % 28:02d}. Treatment started on "
        f"2022-07-{1 + p.idx % 28:02d}. By 2023-01-15, {p.formal_name} showed "
        f"significant improvement. A relapse was noted on 2024-03-{1 + p.idx % 28:02d} "
        f"when {p.first} reported recurring symptoms. The treatment was adjusted "
        f"and SSN {p.ssn} was used to verify insurance continuity. "
        f"Current status as of 2025-01-01: stable. Contact: {p.email}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "context_loss_temporal"}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "context_loss_temporal"}),
        (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "context_loss_temporal"}),
        (p.ssn, "US_SSN", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "context_loss_temporal"}),
    ])
    return text, labels


def _scenario_context_loss_negation(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Negation context: PII appears in a negated context. Naive anonymization might
    change meaning. E.g., 'The account does NOT belong to X' becomes ambiguous."""
    other_idx = p.idx + 18
    other_first = FIRST_NAMES["en"][other_idx % len(FIRST_NAMES["en"])]
    other_last = LAST_NAMES["en"][other_idx % len(LAST_NAMES["en"])]
    other_name = f"{other_first} {other_last}"
    text = (
        f"Investigation confirmed that account {p.bank_account} does NOT belong to "
        f"{other_name}. The actual account holder is {p.full_name} (SSN: {p.ssn}). "
        f"Correspondence addressed to {p.formal_name} at {p.email} confirmed this. "
        f"{other_name} was cleared of any involvement. The funds were traced to "
        f"card {p.credit_card_formatted} registered to {p.first} {p.last} at "
        f"{p.full_address}."
    )
    other_cluster = f"person-other-{other_idx:05d}"
    labels = _labels_for(text, [
        (p.bank_account, "BANK_ACCOUNT", {}),
        (other_name, "PERSON_NAME", {"cluster": other_cluster, "variant": "full_name", "context_group": "context_loss_negation"}),
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "context_loss_negation"}),
        (p.ssn, "US_SSN", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "context_loss_negation"}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "context_loss_negation"}),
        (p.credit_card_formatted, "CREDIT_CARD", {}),
        (p.full_address, "ADDRESS", {}),
    ])
    # Second other_name mention
    on2 = _label(text, other_name, "PERSON_NAME",
                  start_search=text.find(other_name) + len(other_name),
                  cluster=other_cluster, variant="full_name", context_group="context_loss_negation")
    if on2:
        labels.append(on2)
    # first + last as separate mention
    fl = _label(text, f"{p.first} {p.last} at", "PERSON_NAME",
                 cluster=p.cluster_id, variant="full_name", context_group="context_loss_negation")
    if fl:
        fl["end"] = fl["start"] + len(f"{p.first} {p.last}")
        labels.append(fl)
    return text, labels


def _scenario_embedded_pii_in_code(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """PII embedded in code/config snippets — common in log analysis pipelines."""
    text = (
        f'config = {{\n'
        f'    "db_user": "{p.username}",\n'
        f'    "admin_email": "{p.email}",\n'
        f'    "api_key": "sk-{_det_hash(p.idx, "apikey")}",\n'
        f'    "ssh_host": "{p.ipv4}",\n'
        f'    "owner": "{p.full_name}",\n'
        f'    "ssn_backup": "{p.ssn}",\n'
        f'}}\n'
        f'# Contact {p.formal_name} at {p.phone_us} for access.\n'
        f'# MAC whitelist: {p.mac_address}'
    )
    labels = _labels_for(text, [
        (p.username, "USERNAME", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "code_embedded"}),
        (p.ipv4, "IP_ADDRESS", {}),
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "code_embedded"}),
        (p.ssn, "US_SSN", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "code_embedded"}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.mac_address, "MAC_ADDRESS", {}),
    ])
    return text, labels


def _scenario_mixed_language(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Mixed-language text block where PII appears in multiple language contexts."""
    text = (
        f"Dear {p.full_name},\n\n"
        f"Estimado/a {p.formal_name}, le confirmamos su reserva. "
        f"Votre reference est #{80000 + p.idx}.\n\n"
        f"Passport: {p.passport}\n"
        f"Fecha de nacimiento / Date of birth: {p.dob}\n"
        f"Telefono / Phone: {p.phone_us}\n"
        f"Correo / Email: {p.email}\n"
        f"Direccion / Address: {p.full_address}\n\n"
        f"Merci, {p.first}. Gracias."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "mixed_language"}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "mixed_language"}),
        (p.passport, "PASSPORT", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "mixed_language"}),
        (p.full_address, "ADDRESS", {}),
    ])
    # First name at end
    fn = _label(text, f"Merci, {p.first}.", "PERSON_NAME",
                 cluster=p.cluster_id, variant="first_name", context_group="mixed_language")
    if fn:
        fn["start"] += len("Merci, ")
        fn["end"] = fn["start"] + len(p.first)
        labels.append(fn)
    return text, labels


# ---------------------------------------------------------------------------
# Multilingual baseline templates
# ---------------------------------------------------------------------------

def _multilingual_baseline(p: _PersonaFactory, language: str) -> tuple[str, list[dict[str, Any]]]:
    """Generate a medium-sized baseline record in the target language."""
    templates = {
        "es": (
            f"Registro de empleado: {p.full_name}, correo: {p.email}, "
            f"telefono: {p.phone_us}. Numero de seguro social: {p.ssn}. "
            f"Direccion: {p.full_address}. Empresa: {p.company}."
        ),
        "fr": (
            f"Fiche employe: {p.full_name}, courriel: {p.email}, "
            f"telephone: {p.phone_us}. NSS: {p.ssn}. "
            f"Adresse: {p.full_address}. Entreprise: {p.company}."
        ),
        "de": (
            f"Mitarbeiterdaten: {p.full_name}, E-Mail: {p.email}, "
            f"Telefon: {p.phone_us}. Sozialversicherungsnummer: {p.ssn}. "
            f"Adresse: {p.full_address}. Unternehmen: {p.company}."
        ),
        "it": (
            f"Scheda dipendente: {p.full_name}, email: {p.email}, "
            f"telefono: {p.phone_us}. Codice fiscale: {p.ssn}. "
            f"Indirizzo: {p.full_address}. Azienda: {p.company}."
        ),
        "pt": (
            f"Registo de funcionario: {p.full_name}, email: {p.email}, "
            f"telefone: {p.phone_us}. NIF: {p.ssn}. "
            f"Morada: {p.full_address}. Empresa: {p.company}."
        ),
        "nl": (
            f"Werknemergegevens: {p.full_name}, e-mail: {p.email}, "
            f"telefoon: {p.phone_us}. BSN: {p.ssn}. "
            f"Adres: {p.full_address}. Bedrijf: {p.company}."
        ),
        "ja": (
            f"Employee record: {p.full_name}, email: {p.email}, "
            f"phone: {p.phone_us}. ID: {p.ssn}. "
            f"Address: {p.full_address}. Company: {p.company}."
        ),
        "ar": (
            f"Employee record: {p.full_name}, email: {p.email}, "
            f"phone: {p.phone_us}. Tax ID: {p.ssn}. "
            f"Address: {p.full_address}. Company: {p.company}."
        ),
        "hi": (
            f"Karmachari record: {p.full_name}, email: {p.email}, "
            f"phone: {p.phone_us}. Tax ID: {p.ssn}. "
            f"Address: {p.full_address}. Company: {p.company}."
        ),
        "zh": (
            f"Employee profile: {p.full_name}, email: {p.email}, "
            f"phone: {p.phone_us}. Tax ID: {p.ssn}. "
            f"Address: {p.full_address}. Company: {p.company}."
        ),
        "ko": (
            f"Employee profile: {p.full_name}, email: {p.email}, "
            f"phone: {p.phone_us}. Tax ID: {p.ssn}. "
            f"Address: {p.full_address}. Company: {p.company}."
        ),
    }
    text = templates.get(language, templates["es"])
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.ssn, "US_SSN", {}),
        (p.full_address, "ADDRESS", {}),
        (p.company, "ORGANIZATION", {}),
    ])
    return text, labels


# ---------------------------------------------------------------------------
# CORE DATASET GENERATION
# ---------------------------------------------------------------------------

# Distribution plan for 10,200 records:
# - English: 5,000 (synthetic: 3,500, curated_public: 1,500)
# - Spanish: 1,300 (synthetic: 910, curated_public: 390)
# - French: 900 (synthetic: 630, curated_public: 270)
# - German: 550 (synthetic: 385, curated_public: 165)
# - Italian: 350 (synthetic: 245, curated_public: 105)
# - Portuguese: 250 (synthetic: 175, curated_public: 75)
# - Dutch: 150 (synthetic: 105, curated_public: 45)
# - Japanese: 150 (synthetic: 105, curated_public: 45)
# - Arabic: 400 (synthetic: 280, curated_public: 120)
# - Hindi: 400 (synthetic: 280, curated_public: 120)
# - Chinese: 400 (synthetic: 280, curated_public: 120)
# - Korean: 350 (synthetic: 245, curated_public: 105)

LANG_DISTRIBUTION = {
    "en": {"synthetic": 12250, "curated_public": 5250},
    "es": {"synthetic": 3185, "curated_public": 1365},
    "fr": {"synthetic": 2205, "curated_public": 945},
    "de": {"synthetic": 1348, "curated_public": 577},
    "it": {"synthetic": 858, "curated_public": 367},
    "pt": {"synthetic": 613, "curated_public": 262},
    "nl": {"synthetic": 368, "curated_public": 157},
    "ja": {"synthetic": 368, "curated_public": 157},
    "ar": {"synthetic": 980, "curated_public": 420},
    "hi": {"synthetic": 980, "curated_public": 420},
    "zh": {"synthetic": 980, "curated_public": 420},
    "ko": {"synthetic": 858, "curated_public": 367},
}

# English template weights (indexes into dispatch table)
EN_TEMPLATE_DISPATCH = [
    # Small templates (25%)
    ("small", _small_simple_contact),
    ("small", _small_phone_ssn),
    ("small", _small_id_only),
    ("small", _small_financial_alert),
    ("small", _small_log_line),
    ("small", _small_dob_address),
    # Medium templates (35%)
    ("medium", _medium_patient_note),
    ("medium", _medium_employee_record),
    ("medium", _medium_financial_transaction),
    ("medium", _medium_legal_notice),
    ("medium", _medium_structured_form),
    ("medium", _medium_insurance_claim),
    # Large templates (25%)
    ("large", _large_medical_discharge),
    ("large", _large_legal_deposition),
    ("large", _large_customer_support_thread),
    ("large", _large_hr_investigation),
    # Context-loss scenarios
    ("context_loss", _scenario_context_loss_pronoun),
    ("context_loss", _scenario_context_loss_temporal),
    ("context_loss", _scenario_context_loss_negation),
    ("context_loss", _scenario_embedded_pii_in_code),
    ("context_loss", _scenario_mixed_language),
]

# Very large templates need rng passed separately
EN_VLARGE_DISPATCH = [
    _vlarge_multi_party_medical,
    _vlarge_financial_audit,
    _vlarge_immigration_case,
]


def _generate_en_record(
    rng: random.Random, index: int, source_type: str,
) -> tuple[str, list[dict[str, Any]], str, str, str]:
    """Generate an English record. Returns (text, labels, scenario_id, size_tier, context_group)."""
    p = _PersonaFactory(rng, "en", index)

    # Deterministic size-tier assignment based on index
    tier_roll = index % 20
    if tier_roll < 4:  # 20% small
        tmpl_idx = index % 6  # 6 small templates
        size_tier, fn = EN_TEMPLATE_DISPATCH[tmpl_idx]
        text, labels = fn(p)
        scenario = "baseline"
        ctx = "baseline"
    elif tier_roll < 10:  # 30% medium
        tmpl_idx = 6 + (index % 6)  # 6 medium templates
        size_tier, fn = EN_TEMPLATE_DISPATCH[tmpl_idx]
        text, labels = fn(p)
        scenario = "baseline"
        ctx = "baseline"
    elif tier_roll < 15:  # 25% large
        tmpl_idx = 12 + (index % 4)  # 4 large templates
        size_tier, fn = EN_TEMPLATE_DISPATCH[tmpl_idx]
        text, labels = fn(p)
        scenario = "baseline"
        ctx = labels[0].get("context_group", "baseline") if labels else "baseline"
    elif tier_roll < 16:  # 5% very large
        vl_idx = index % len(EN_VLARGE_DISPATCH)
        text, labels = EN_VLARGE_DISPATCH[vl_idx](p, rng)
        size_tier = "very_large"
        scenario = "baseline"
        ctx = labels[0].get("context_group", "baseline") if labels else "baseline"
    else:  # 20% context-loss
        # Use index // 20 to vary template selection across context-loss rows.
        cl_idx = 16 + ((index // 20) % 5)  # 5 context-loss templates
        size_tier, fn = EN_TEMPLATE_DISPATCH[cl_idx]
        # _scenario_context_loss_relational needs rng as second arg
        try:
            text, labels = fn(p, rng)  # type: ignore[call-arg]
        except TypeError:
            text, labels = fn(p)
        scenario = "context_loss"
        ctx = labels[0].get("context_group", "context_loss") if labels else "context_loss"

    return text, labels, scenario, size_tier, ctx


def _multilingual_context_loss(
    p: _PersonaFactory,
    language: str,
    *,
    alternate_last_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate multilingual context-loss examples with alias ambiguity pressure."""
    alt_full = f"{p.first} {alternate_last_name}"
    alt_cluster = f"{p.cluster_id}-alt"

    templates = {
        "es": (
            f"En el expediente, {p.full_name} aparece como titular. "
            f"Mas tarde solo se menciona {p.first} sin apellido. "
            f"El correo {p.email} pertenece a {p.formal_name}; "
            f"no confundir con {alt_full}."
        ),
        "fr": (
            f"Dans le dossier, {p.full_name} est le sujet principal. "
            f"Plus tard, seule la forme {p.first} apparait. "
            f"L'adresse {p.email} appartient a {p.formal_name}; "
            f"ne pas confondre avec {alt_full}."
        ),
        "de": (
            f"Im Vorgang ist {p.full_name} die Hauptperson. "
            f"Spaeter wird nur {p.first} genannt. "
            f"Die Adresse {p.email} gehoert zu {p.formal_name}; "
            f"nicht mit {alt_full} verwechseln."
        ),
        "it": (
            f"Nel fascicolo {p.full_name} e il soggetto principale. "
            f"Piu avanti compare solo {p.first}. "
            f"L'email {p.email} appartiene a {p.formal_name}; "
            f"non confondere con {alt_full}."
        ),
        "pt": (
            f"No processo, {p.full_name} e o titular principal. "
            f"Depois surge apenas {p.first}. "
            f"O email {p.email} pertence a {p.formal_name}; "
            f"nao confundir com {alt_full}."
        ),
        "nl": (
            f"In het dossier is {p.full_name} de hoofdpersoon. "
            f"Later staat alleen {p.first}. "
            f"Het adres {p.email} hoort bij {p.formal_name}; "
            f"niet verwarren met {alt_full}."
        ),
        "ja": (
            f"Record states {p.full_name} as primary subject. "
            f"Later only {p.first} is referenced. "
            f"Email {p.email} belongs to {p.formal_name}; "
            f"do not confuse with {alt_full}."
        ),
        "ar": (
            f"File lists {p.full_name} as primary subject. "
            f"Later only {p.first} is referenced. "
            f"Email {p.email} belongs to {p.formal_name}; "
            f"do not confuse with {alt_full}."
        ),
        "hi": (
            f"Record me {p.full_name} primary subject hai. "
            f"Later sirf {p.first} mention hota hai. "
            f"Email {p.email} {p.formal_name} ka hai; "
            f"{alt_full} se confuse na karein."
        ),
        "zh": (
            f"Record marks {p.full_name} as primary subject. "
            f"Later only {p.first} appears. "
            f"Email {p.email} belongs to {p.formal_name}; "
            f"do not mix with {alt_full}."
        ),
        "ko": (
            f"Record names {p.full_name} as primary subject. "
            f"Later only {p.first} is referenced. "
            f"Email {p.email} belongs to {p.formal_name}; "
            f"do not confuse with {alt_full}."
        ),
    }
    text = templates.get(language, templates["es"])
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "context_loss_multilingual"}),
        (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "context_loss_multilingual"}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "context_loss_multilingual"}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "context_loss_multilingual"}),
        (alt_full, "PERSON_NAME", {"cluster": alt_cluster, "variant": "full_name", "context_group": "context_loss_multilingual"}),
    ])
    return text, labels


def _datatype_and_difficulty(
    *,
    index: int,
    language: str,
    scenario: str,
    size_tier: str,
    text: str,
    labels: list[dict[str, Any]],
) -> tuple[str, str, str, list[dict[str, Any]]]:
    working_text = text
    working_labels = list(labels)
    lower = working_text.lower()

    datatype_group = "general_pii"
    if index % 41 == 0:
        token = _det_hash(index, "wallet", language)
        working_text = f"{working_text} Wallet reference: 0x{token}{token}{token[:16]}."
        datatype_group = "crypto_wallet"
    elif index % 37 == 0:
        token = _det_hash(index, "apikey", language)
        working_text = f'{working_text} Sample config field api_key="sk-live-{token}" (dummy test token).'
        datatype_group = "api_key_like_negative"
    elif index % 29 == 0:
        marker = f"NH-{900000 + index:06d}"
        start = len(working_text) + 1
        working_text = f"{working_text} National health id: {marker}."
        working_labels.append(
            {
                "entity_type": "MEDICAL_RECORD_NUMBER",
                "start": start + len("National health id: "),
                "end": start + len("National health id: ") + len(marker),
                "entity_cluster_id": "none",
                "mention_variant": "none",
                "context_group": "health_registry",
            }
        )
        datatype_group = "national_health_id"
    elif index % 23 == 0:
        marker = f"TAX-{700000000 + index:09d}"
        start = len(working_text) + 1
        working_text = f"{working_text} International tax id: {marker}."
        working_labels.append(
            {
                "entity_type": "NATIONAL_ID",
                "start": start + len("International tax id: "),
                "end": start + len("International tax id: ") + len(marker),
                "entity_cluster_id": "none",
                "mention_variant": "none",
                "context_group": "tax_registry",
            }
        )
        datatype_group = "tax_id_intl"
    elif index % 19 == 0:
        marker = ":".join(f"{(index * p) % 256:02x}" for p in [2, 3, 5, 7, 11, 13])
        start = len(working_text) + 1
        working_text = f"{working_text} Device id MAC: {marker}."
        working_labels.append(
            {
                "entity_type": "MAC_ADDRESS",
                "start": start + len("Device id MAC: "),
                "end": start + len("Device id MAC: ") + len(marker),
                "entity_cluster_id": "none",
                "mention_variant": "none",
                "context_group": "device_inventory",
            }
        )
        datatype_group = "device_id"
    elif any(lbl.get("entity_type") == "PASSPORT" for lbl in working_labels):
        datatype_group = "passport_intl"
    elif scenario == "context_loss":
        datatype_group = "alias_context_loss"
    elif language != "en" and any(lbl.get("entity_type") == "US_SSN" for lbl in working_labels):
        datatype_group = "tax_id_intl"
    elif "api_key" in lower or "sk-" in lower:
        datatype_group = "api_key_like_negative"
    elif "wallet" in lower:
        datatype_group = "crypto_wallet"

    if size_tier in {"very_large", "context_loss"} or scenario == "context_loss":
        difficulty = "hard"
    elif datatype_group in {"api_key_like_negative", "crypto_wallet"}:
        difficulty = "challenging"
    elif size_tier == "large":
        difficulty = "moderate"
    else:
        difficulty = "easy"

    return datatype_group, difficulty, working_text, working_labels


def _generate_core_rows(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    counter = 0

    for language, splits in LANG_DISTRIBUTION.items():
        for source_type, count in splits.items():
            for i in range(count):
                counter += 1
                index = counter

                if language == "en":
                    text, labels, scenario, size_tier, ctx = _generate_en_record(rng, index, source_type)
                else:
                    p = _PersonaFactory(rng, language, index)
                    if i % 5 == 4:  # 20% context-loss for non-English rows
                        alt_names = LAST_NAMES.get(language, LAST_NAMES["en"])
                        alt_last = alt_names[(index + 17) % len(alt_names)]
                        text, labels = _multilingual_context_loss(
                            p,
                            language,
                            alternate_last_name=alt_last,
                        )
                        size_tier = "context_loss"
                        scenario = "context_loss"
                        ctx = "context_loss_multilingual"
                    else:
                        text, labels = _multilingual_baseline(p, language)
                        size_tier = "medium"
                        scenario = "baseline"
                        ctx = "baseline"

                datatype_group, difficulty_level, text, labels = _datatype_and_difficulty(
                    index=index,
                    language=language,
                    scenario=scenario,
                    size_tier=size_tier,
                    text=text,
                    labels=labels,
                )

                # Determine entity cluster info
                cluster_id = "none"
                mention_variant = "none"
                for lbl in labels:
                    if lbl.get("entity_cluster_id", "none") != "none":
                        cluster_id = lbl["entity_cluster_id"]
                        mention_variant = "multi_variant"
                        break

                row = {
                    "id": f"{source_type[:3].upper()}-{counter:06d}",
                    "text": text,
                    "language": language,
                    "labels": labels,
                    "source_type": source_type,
                    "source_id": (
                        f"synthetic://v2/template-{index % 21}"
                        if source_type == "synthetic"
                        else f"curated://public-template-{index % 21}"
                    ),
                    "license": "CC0-1.0" if source_type == "synthetic" else "CC-BY-4.0",
                    "scenario_id": scenario,
                    "entity_cluster_id": cluster_id,
                    "mention_variant": mention_variant,
                    "context_group": ctx,
                    "size_tier": size_tier,
                    "datatype_group": datatype_group,
                    "difficulty_level": difficulty_level,
                }
                rows.append(row)

    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# TRACKING DATASET GENERATION (2,800 records)
# ---------------------------------------------------------------------------

def _build_tracking_row_v2(
    rng: random.Random,
    index: int,
    *,
    ambiguous: bool,
    language: str,
) -> dict[str, Any]:
    p = _PersonaFactory(rng, language, index)

    if not ambiguous:
        # Single-entity continuity with rich alias set
        canonical_templates = {
            "en": (
                f"Document begins with reference to {p.full_name} as the primary subject. "
                f"In subsequent paragraphs, the subject is referred to as {p.formal_name}, "
                f"and informally as {p.first}. Email correspondence was directed to {p.email} "
                f"and the corporate address {p.corporate_email}. "
                f"Records identify {p.first_last_initial} with SSN {p.ssn}. "
                f"Phone contact: {p.phone_us}. Home address: {p.full_address}."
            ),
            "es": (
                f"El documento identifica a {p.full_name} como sujeto principal. "
                f"Despues aparece como {p.formal_name} y tambien como {p.first}. "
                f"El correo {p.email} y {p.corporate_email} pertenecen al mismo sujeto. "
                f"El expediente marca {p.first_last_initial} con SSN {p.ssn} y telefono {p.phone_us}."
            ),
            "fr": (
                f"Le document reference {p.full_name} comme sujet principal. "
                f"Ensuite, le sujet apparait comme {p.formal_name} et {p.first}. "
                f"Les adresses {p.email} et {p.corporate_email} appartiennent au meme sujet. "
                f"Le dossier note {p.first_last_initial} avec SSN {p.ssn} et telephone {p.phone_us}."
            ),
            "de": (
                f"Im Dokument ist {p.full_name} die Hauptperson. "
                f"Spater erscheint dieselbe Person als {p.formal_name} und {p.first}. "
                f"Die E-Mails {p.email} und {p.corporate_email} gehoeren zur gleichen Person. "
                f"Die Akte enthaelt {p.first_last_initial} mit SSN {p.ssn} und Telefon {p.phone_us}."
            ),
            "ja": (
                f"Record lists {p.full_name} as main subject. "
                f"Later mentions include {p.formal_name} and {p.first}. "
                f"Emails {p.email} and {p.corporate_email} refer to the same subject. "
                f"File includes {p.first_last_initial}, SSN {p.ssn}, and phone {p.phone_us}."
            ),
        }
        text = canonical_templates.get(language, canonical_templates["en"])
        labels = _labels_for(text, [
            (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "tracking"}),
            (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "tracking"}),
            (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "tracking"}),
            (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "personal_email", "context_group": "tracking"}),
            (p.corporate_email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "work_email", "context_group": "tracking"}),
            (p.first_last_initial, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_last_initial", "context_group": "tracking"}),
            (p.ssn, "US_SSN", {}),
            (p.phone_us, "PHONE_NUMBER", {}),
            (p.full_address, "ADDRESS", {}),
        ])
        return {
            "id": f"TRK-{index:05d}",
            "text": text,
            "language": language,
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"tracking://v2-canonical-{index}",
            "license": "CC0-1.0",
            "scenario_id": "continuity_tracking",
            "entity_cluster_id": p.cluster_id,
            "mention_variant": "full_formal_first_initial_emails",
            "context_group": "tracking",
            "size_tier": "medium",
            "datatype_group": "long_context_alias",
            "difficulty_level": "hard",
        }
    else:
        # Ambiguous: two people sharing a first name
        p2 = _PersonaFactory(rng, language, index + 500)
        # Force shared first name
        shared_first = p.first
        p2_formal = f"{HONORIFICS.get(language, HONORIFICS['en'])[1]} {p2.last}"
        ambiguous_templates = {
            "en": (
                f"Case file references both {p.full_name} and {shared_first} {p2.last}. "
                f"In meeting notes, the name {shared_first} appears without disambiguation. "
                f"Email {p.email} belongs to {p.formal_name}, while {p2.email} belongs to "
                f"{p2_formal}. "
                f"Phone records show {p.phone_us} for {p.last} and {p2.phone_us} for {p2.last}. "
                f"SSN {p.ssn} is associated with {p.full_name}; SSN {p2.ssn} with "
                f"{shared_first} {p2.last}."
            ),
            "es": (
                f"El caso incluye a {p.full_name} y a {shared_first} {p2.last}. "
                f"En notas de reunion, {shared_first} aparece sin apellido. "
                f"Correo {p.email} pertenece a {p.formal_name}, mientras {p2.email} pertenece a {p2_formal}. "
                f"Telefonos: {p.phone_us} para {p.last} y {p2.phone_us} para {p2.last}. "
                f"SSN {p.ssn} corresponde a {p.full_name}; SSN {p2.ssn} a {shared_first} {p2.last}."
            ),
            "fr": (
                f"Le dossier contient {p.full_name} et {shared_first} {p2.last}. "
                f"Dans les notes, {shared_first} apparait sans nom de famille. "
                f"L'email {p.email} correspond a {p.formal_name}, alors que {p2.email} correspond a {p2_formal}. "
                f"Telephones: {p.phone_us} pour {p.last} et {p2.phone_us} pour {p2.last}. "
                f"SSN {p.ssn} est lie a {p.full_name}; SSN {p2.ssn} a {shared_first} {p2.last}."
            ),
            "de": (
                f"Die Akte nennt {p.full_name} und {shared_first} {p2.last}. "
                f"In Besprechungsnotizen steht nur {shared_first}. "
                f"Email {p.email} gehoert zu {p.formal_name}, waehrend {p2.email} zu {p2_formal} gehoert. "
                f"Telefon: {p.phone_us} fuer {p.last}, {p2.phone_us} fuer {p2.last}. "
                f"SSN {p.ssn} gehoert zu {p.full_name}; SSN {p2.ssn} zu {shared_first} {p2.last}."
            ),
            "ja": (
                f"Case lists {p.full_name} and {shared_first} {p2.last}. "
                f"Notes use only {shared_first} without surname. "
                f"Email {p.email} belongs to {p.formal_name}; {p2.email} belongs to {p2_formal}. "
                f"Phone {p.phone_us} maps to {p.last}; {p2.phone_us} maps to {p2.last}. "
                f"SSN {p.ssn} is for {p.full_name}; SSN {p2.ssn} for {shared_first} {p2.last}."
            ),
        }
        text = ambiguous_templates.get(language, ambiguous_templates["en"])
        p2_full = f"{shared_first} {p2.last}"
        labels = _labels_for(text, [
            (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "ambiguous"}),
            (p2_full, "PERSON_NAME", {"cluster": p2.cluster_id, "variant": "full_name", "context_group": "ambiguous"}),
            (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "ambiguous"}),
            (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "ambiguous"}),
            (p2.email, "EMAIL_ADDRESS", {"cluster": p2.cluster_id, "variant": "email", "context_group": "ambiguous"}),
            (p2_formal, "PERSON_NAME", {"cluster": p2.cluster_id, "variant": "formal", "context_group": "ambiguous"}),
            (p.phone_us, "PHONE_NUMBER", {}),
            (p2.phone_us, "PHONE_NUMBER", {}),
            (p.ssn, "US_SSN", {}),
            (p2.ssn, "US_SSN", {}),
        ])
        # Shared first name (ambiguous mention)
        shared_lbl = _label(text, f"name {shared_first} appears", "PERSON_NAME",
                             cluster="ambiguous", variant="shared_first_name", context_group="ambiguous")
        if shared_lbl:
            shared_lbl["start"] += len("name ")
            shared_lbl["end"] = shared_lbl["start"] + len(shared_first)
            labels.append(shared_lbl)
        # Last name mentions
        for person, cluster in [(p, p.cluster_id), (p2, p2.cluster_id)]:
            ln = _label(text, f"for {person.last} and", "PERSON_NAME",
                         cluster=cluster, variant="last_name", context_group="ambiguous")
            if ln:
                ln["start"] += len("for ")
                ln["end"] = ln["start"] + len(person.last)
                labels.append(ln)
            ln2 = _label(text, f"for {person.last}.", "PERSON_NAME",
                          cluster=cluster, variant="last_name", context_group="ambiguous")
            if ln2:
                ln2["start"] += len("for ")
                ln2["end"] = ln2["start"] + len(person.last)
                labels.append(ln2)
        # Second full name mentions
        fn2 = _label(text, p.full_name, "PERSON_NAME",
                      start_search=text.find(p.full_name) + len(p.full_name),
                      cluster=p.cluster_id, variant="full_name", context_group="ambiguous")
        if fn2:
            labels.append(fn2)
        p2fn2 = _label(text, p2_full, "PERSON_NAME",
                         start_search=text.find(p2_full) + len(p2_full),
                         cluster=p2.cluster_id, variant="full_name", context_group="ambiguous")
        if p2fn2:
            labels.append(p2fn2)
        return {
            "id": f"TRK-{index:05d}",
            "text": text,
            "language": language,
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"tracking://v2-ambiguous-{index}",
            "license": "CC0-1.0",
            "scenario_id": "continuity_ambiguous",
            "entity_cluster_id": p.cluster_id,
            "mention_variant": "mixed",
            "context_group": "tracking",
            "size_tier": "medium",
            "datatype_group": "long_context_alias",
            "difficulty_level": "hard",
        }


def _generate_tracking_rows(seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed + 11)
    rows: list[dict[str, Any]] = []
    tracking_language_distribution = {
        "en": 1400,
        "es": 420,
        "fr": 350,
        "de": 350,
        "ja": 280,
    }
    language_pool: list[str] = []
    for language, count in tracking_language_distribution.items():
        language_pool.extend([language] * count)
    rng.shuffle(language_pool)

    # 1750 canonical + 1050 ambiguous records, with multilingual mix.
    for idx, language in enumerate(language_pool[:1750], start=1):
        rows.append(_build_tracking_row_v2(rng, idx, ambiguous=False, language=language))
    for idx, language in enumerate(language_pool[1750:], start=1751):
        rows.append(_build_tracking_row_v2(rng, idx, ambiguous=True, language=language))
    rng.shuffle(rows)
    return rows


# ---------------------------------------------------------------------------
# EVALUATION DIMENSION TEMPLATES — 7 research-grade dimensions
# ---------------------------------------------------------------------------
# Informed by: PII-Rate-Elo composite framework, TAB coreference evaluation,
# PII-Bench query-aware detection, RAT-Bench re-identification risk,
# TAU-EVAL utility-privacy trade-off, PrivaCI-Bench contextual integrity,
# ai4privacy multilingual methodology, PANORAMA memorization evaluation.

# ---- DIMENSION 1: ENTITY CONSISTENCY (weight 20%) ----
# Tests whether anonymization preserves consistent entity identity across
# multiple mentions, aliases, and coreference chains in lengthy documents.

def _dim_entity_consistency_long_narrative(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """Very long narrative (1000+ chars) with 8+ mentions of same entity in different forms."""
    colleague = _PersonaFactory(rng, "en", p.idx + 70)
    supervisor = _PersonaFactory(rng, "en", p.idx + 80)
    text = (
        f"PERFORMANCE REVIEW — Fiscal Year 2025\n"
        f"{'=' * 45}\n\n"
        f"Employee: {p.full_name} (ID: {p.employee_id})\n"
        f"Department: Engineering | Manager: {supervisor.full_name}\n\n"
        f"SUMMARY:\n"
        f"{p.formal_name} has demonstrated exceptional growth this year. "
        f"Since joining {p.company}, {p.first} has consistently exceeded expectations. "
        f"Colleagues, including {colleague.full_name}, praise {p.first_last_initial}'s "
        f"collaborative approach.\n\n"
        f"Q1 REVIEW:\n"
        f"During Q1, {p.full_name} led the migration project. {p.formal_name} coordinated "
        f"with three cross-functional teams. The project was delivered ahead of schedule, "
        f"and {p.first} received commendation from the CTO.\n\n"
        f"Q2-Q3 REVIEW:\n"
        f"{p.first_last_initial} took on mentorship of two junior engineers. "
        f"{supervisor.formal_name} noted that {p.last}'s mentorship style significantly "
        f"improved team velocity. Email correspondence between {p.email} and "
        f"{supervisor.email} confirms the positive trajectory.\n\n"
        f"RECOMMENDATION:\n"
        f"We recommend promotion for {p.full_name}. Contact {p.phone_us} or "
        f"{p.corporate_email} for scheduling the review meeting.\n"
        f"Signed: {supervisor.full_name}, {supervisor.phone_us}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "entity_consistency"}),
        (p.employee_id, "EMPLOYEE_ID", {}),
        (supervisor.full_name, "PERSON_NAME", {"cluster": supervisor.cluster_id, "variant": "full_name", "context_group": "entity_consistency"}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "entity_consistency"}),
        (p.company, "ORGANIZATION", {}),
        (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "entity_consistency"}),
        (colleague.full_name, "PERSON_NAME", {}),
        (p.first_last_initial, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_last_initial", "context_group": "entity_consistency"}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "personal_email", "context_group": "entity_consistency"}),
        (supervisor.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.corporate_email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "work_email", "context_group": "entity_consistency"}),
        (supervisor.phone_us, "PHONE_NUMBER", {}),
    ])
    # Repeated mentions
    for needle, variant in [(p.full_name, "full_name"), (p.formal_name, "formal"),
                            (p.first_last_initial, "first_last_initial")]:
        pos = text.find(needle)
        if pos >= 0:
            second = _label(text, needle, "PERSON_NAME", start_search=pos + len(needle),
                            cluster=p.cluster_id, variant=variant, context_group="entity_consistency")
            if second:
                labels.append(second)
    # Last name mention
    ln = _label(text, f"{p.last}'s mentorship", "PERSON_NAME",
                cluster=p.cluster_id, variant="last_name", context_group="entity_consistency")
    if ln:
        ln["end"] = ln["start"] + len(p.last)
        labels.append(ln)
    # Second supervisor full name
    sv2 = _label(text, supervisor.full_name, "PERSON_NAME",
                 start_search=text.find(supervisor.full_name) + len(supervisor.full_name),
                 cluster=supervisor.cluster_id, variant="full_name", context_group="entity_consistency")
    if sv2:
        labels.append(sv2)
    return text, labels


# ---- DIMENSION 3: CONTEXT PRESERVATION (weight 20%) ----
# Tests whether anonymization preserves conversational context, dialogue flow,
# and semantic relationships between speakers.

def _dim_context_conversation_chat(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """Multi-turn chat conversation with PII exchanged between participants."""
    other = _PersonaFactory(rng, "en", p.idx + 90)
    text = (
        f"[10:02 AM] {p.full_name}: Hey {other.first}, can you send me the client file?\n"
        f"[10:03 AM] {other.full_name}: Sure! The client is at {p.full_address}.\n"
        f"[10:04 AM] {p.first}: Thanks. Their contact is {rng.randint(200,999)}-{rng.randint(200,999)}-{rng.randint(1000,9999)}... "
        f"wait, let me check. It's {p.phone_us}.\n"
        f"[10:05 AM] {other.first}: Got it. I'll email the docs to {p.email}.\n"
        f"[10:06 AM] {p.full_name}: Perfect. CC {other.email} too.\n"
        f"[10:07 AM] {other.full_name}: Will do. The account number is {p.bank_account}.\n"
        f"[10:08 AM] {p.first}: And the SSN on file is {p.ssn}, right?\n"
        f"[10:09 AM] {other.first}: Confirmed. DOB: {p.dob_us}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "conversation"}),
        (other.full_name, "PERSON_NAME", {"cluster": other.cluster_id, "variant": "full_name", "context_group": "conversation"}),
        (p.full_address, "ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "conversation"}),
        (other.email, "EMAIL_ADDRESS", {"cluster": other.cluster_id, "variant": "email", "context_group": "conversation"}),
        (p.bank_account, "BANK_ACCOUNT", {}),
        (p.ssn, "US_SSN", {}),
        (p.dob_us, "DATE_OF_BIRTH", {}),
    ])
    # Repeated first names
    for person in [p, other]:
        search_from = 0
        found = 0
        while found < 4:
            lbl = _label(text, f"] {person.first}:", "PERSON_NAME", start_search=search_from,
                         cluster=person.cluster_id, variant="first_name", context_group="conversation")
            if lbl is None:
                break
            lbl["start"] += 2  # skip "] "
            lbl["end"] -= 1    # remove ":"
            labels.append(lbl)
            search_from = lbl["end"] + 1
            found += 1
    # Second full name mentions
    for person in [p, other]:
        fn2 = _label(text, person.full_name, "PERSON_NAME",
                     start_search=text.find(person.full_name) + len(person.full_name),
                     cluster=person.cluster_id, variant="full_name", context_group="conversation")
        if fn2:
            labels.append(fn2)
    return text, labels


def _dim_context_email_thread(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """Email thread with forwarded context and reply chains."""
    sender = _PersonaFactory(rng, "en", p.idx + 95)
    text = (
        f"From: {p.full_name} <{p.email}>\n"
        f"To: {sender.full_name} <{sender.email}>\n"
        f"Subject: Re: Account Verification for {p.full_name}\n\n"
        f"Hi {sender.first},\n\n"
        f"Per your request, here are my verification details:\n"
        f"- Full name: {p.full_name}\n"
        f"- SSN: {p.ssn}\n"
        f"- DOB: {p.dob}\n"
        f"- Address: {p.full_address}\n"
        f"- Phone: {p.phone_us}\n\n"
        f"Best regards,\n{p.first}\n\n"
        f"--- Original Message ---\n"
        f"From: {sender.full_name} <{sender.email}>\n"
        f"To: {p.email}\n\n"
        f"Dear {p.formal_name},\n\n"
        f"We need to verify your identity for account {p.bank_account}. "
        f"Please reply with your details.\n\n"
        f"Regards,\n{sender.formal_name}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "email_thread"}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "email_thread"}),
        (sender.full_name, "PERSON_NAME", {"cluster": sender.cluster_id, "variant": "full_name", "context_group": "email_thread"}),
        (sender.email, "EMAIL_ADDRESS", {"cluster": sender.cluster_id, "variant": "email", "context_group": "email_thread"}),
        (p.ssn, "US_SSN", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.full_address, "ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.bank_account, "BANK_ACCOUNT", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "email_thread"}),
        (sender.formal_name, "PERSON_NAME", {"cluster": sender.cluster_id, "variant": "formal", "context_group": "email_thread"}),
    ])
    # Capture repeated mentions
    for person, cluster in [(p, p.cluster_id), (sender, sender.cluster_id)]:
        pos = text.find(person.full_name)
        while pos >= 0:
            next_pos = text.find(person.full_name, pos + len(person.full_name))
            if next_pos >= 0:
                labels.append({"entity_type": "PERSON_NAME", "start": next_pos,
                                "end": next_pos + len(person.full_name),
                                "entity_cluster_id": cluster, "mention_variant": "full_name",
                                "context_group": "email_thread"})
            pos = next_pos
    return text, labels


# ---- DIMENSION 5: EDGE CASES (weight 10%) ----
# Tests boundary conditions: overlapping entities, false positive triggers,
# very dense PII, Unicode contexts, and ambiguous patterns.

def _dim_edge_overlapping_entities(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Entities that overlap or are adjacent, testing span boundary handling."""
    text = (
        f"Address: {p.street}, {p.city}, {p.state} {p.zip_code} "
        f"(near {p.city} General Hospital). "
        f"Patient {p.full_name}'s SSN{p.ssn} was entered without a space. "
        f"Phone{p.phone_us} also no space. Email:{p.email}."
    )
    labels = _labels_for(text, [
        (p.street, "ADDRESS", {}),
        (p.city, "LOCATION", {}),
        (p.full_name, "PERSON_NAME", {}),
        (p.ssn, "US_SSN", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {}),
    ])
    # Second city mention
    c2 = _label(text, p.city, "LOCATION", start_search=text.find(p.city) + len(p.city))
    if c2:
        labels.append(c2)
    return text, labels


def _dim_edge_false_positive_triggers(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Text with patterns that look like PII but aren't — tests false positive resilience."""
    text = (
        f"Order #123-45-6789 was shipped (not an SSN). "
        f"Product SKU: 4532-1234-5678-9012 (not a credit card). "
        f"Conference room 10.0.0.1 (room number, not IP). "
        f"Temperature reading: 98.6F for patient {p.full_name}. "
        f"Actual SSN: {p.ssn}. Actual email: {p.email}. "
        f"Actual phone: {p.phone_us}. Ref code: ABC-12-3456 (not an ID)."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.ssn, "US_SSN", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
    ])
    return text, labels


def _dim_edge_dense_pii(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Very short text with extremely dense PII — every token matters."""
    text = (
        f"{p.full_name}|{p.ssn}|{p.dob}|{p.email}|{p.phone_us}|"
        f"{p.full_address}|{p.credit_card_formatted}|{p.bank_account}|"
        f"{p.passport}|{p.drivers_license}|{p.mrn}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.ssn, "US_SSN", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.full_address, "ADDRESS", {}),
        (p.credit_card_formatted, "CREDIT_CARD", {}),
        (p.bank_account, "BANK_ACCOUNT", {}),
        (p.passport, "PASSPORT", {}),
        (p.drivers_license, "DRIVERS_LICENSE", {}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
    ])
    return text, labels


def _dim_edge_unicode_context(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """PII embedded in text with special Unicode characters and formatting."""
    text = (
        f"\u2014 Client Profile \u2014\n"
        f"\u2022 Name: {p.full_name}\n"
        f"\u2022 Email: {p.email}\n"
        f"\u2022 Phone: {p.phone_us}\n"
        f"\u2022 Address: {p.full_address}\n"
        f"\u2022 ID\u00a0Number: {p.ssn}\n"
        f"\u00a9 2025 {p.company}. Confidential \u2013 Do not distribute."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.full_address, "ADDRESS", {}),
        (p.ssn, "US_SSN", {}),
        (p.company, "ORGANIZATION", {}),
    ])
    return text, labels


def _dim_edge_pii_in_url(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """PII leaked through URL parameters and query strings."""
    text = (
        f"User accessed: https://portal.example.com/profile?user={p.username}"
        f"&email={p.email}&ssn={p.ssn}&name={p.full_name.replace(' ', '+')}\n"
        f"Redirect to: https://api.example.com/v2/records/{p.mrn}\n"
        f"Callback URL contains phone: https://hooks.example.com/notify?phone={p.phone_us}"
    )
    labels = _labels_for(text, [
        (p.username, "USERNAME", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.ssn, "US_SSN", {}),
        (p.full_name.replace(' ', '+'), "PERSON_NAME", {}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
    ])
    return text, labels


# ---- DIMENSION 6: FORMAT VARIATIONS (weight 10%) ----
# Tests PII detection across different data formats: JSON, CSV, XML,
# email headers, and structured tables.

def _dim_format_json_record(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """PII embedded in JSON structure."""
    text = (
        f'{{"patient": {{"name": "{p.full_name}", "dob": "{p.dob}", '
        f'"ssn": "{p.ssn}", "email": "{p.email}", '
        f'"phone": "{p.phone_us}", "address": "{p.full_address}", '
        f'"mrn": "{p.mrn}", "insurance_id": "{p.national_id}"}}, '
        f'"provider": {{"name": "{_pick(p.rng, FIRST_NAMES["en"])} {_pick(p.rng, LAST_NAMES["en"])}", '
        f'"employee_id": "{p.employee_id}"}}}}'
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.ssn, "US_SSN", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.full_address, "ADDRESS", {}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.national_id, "NATIONAL_ID", {}),
        (p.employee_id, "EMPLOYEE_ID", {}),
    ])
    return text, labels


def _dim_format_csv_rows(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """PII in CSV format with header row."""
    p2 = _PersonaFactory(rng, "en", p.idx + 100)
    p3 = _PersonaFactory(rng, "en", p.idx + 200)
    text = (
        f"name,email,ssn,phone,address,dob\n"
        f"{p.full_name},{p.email},{p.ssn},{p.phone_us},\"{p.full_address}\",{p.dob}\n"
        f"{p2.full_name},{p2.email},{p2.ssn},{p2.phone_us},\"{p2.full_address}\",{p2.dob}\n"
        f"{p3.full_name},{p3.email},{p3.ssn},{p3.phone_us},\"{p3.full_address}\",{p3.dob}"
    )
    labels = []
    for person in [p, p2, p3]:
        labels.extend(_labels_for(text, [
            (person.full_name, "PERSON_NAME", {}),
            (person.email, "EMAIL_ADDRESS", {}),
            (person.ssn, "US_SSN", {}),
            (person.phone_us, "PHONE_NUMBER", {}),
            (person.full_address, "ADDRESS", {}),
            (person.dob, "DATE_OF_BIRTH", {}),
        ]))
    return text, labels


def _dim_format_xml_fragment(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """PII in XML structure."""
    text = (
        f"<record>\n"
        f"  <patient name=\"{p.full_name}\" dob=\"{p.dob}\">\n"
        f"    <ssn>{p.ssn}</ssn>\n"
        f"    <contact email=\"{p.email}\" phone=\"{p.phone_us}\"/>\n"
        f"    <address>{p.full_address}</address>\n"
        f"    <medical mrn=\"{p.mrn}\"/>\n"
        f"  </patient>\n"
        f"  <employer name=\"{p.company}\" empid=\"{p.employee_id}\"/>\n"
        f"</record>"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.ssn, "US_SSN", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.full_address, "ADDRESS", {}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.company, "ORGANIZATION", {}),
        (p.employee_id, "EMPLOYEE_ID", {}),
    ])
    return text, labels


def _dim_format_table(p: _PersonaFactory, rng: random.Random) -> tuple[str, list[dict[str, Any]]]:
    """PII in ASCII table format."""
    p2 = _PersonaFactory(rng, "en", p.idx + 110)
    text = (
        f"+{'='*25}+{'='*30}+{'='*15}+{'='*20}+\n"
        f"| {'Name':<23} | {'Email':<28} | {'SSN':<13} | {'Phone':<18} |\n"
        f"+{'-'*25}+{'-'*30}+{'-'*15}+{'-'*20}+\n"
        f"| {p.full_name:<23} | {p.email:<28} | {p.ssn:<13} | {p.phone_us:<18} |\n"
        f"| {p2.full_name:<23} | {p2.email:<28} | {p2.ssn:<13} | {p2.phone_us:<18} |\n"
        f"+{'='*25}+{'='*30}+{'='*15}+{'='*20}+"
    )
    labels = []
    for person in [p, p2]:
        labels.extend(_labels_for(text, [
            (person.full_name, "PERSON_NAME", {}),
            (person.email, "EMAIL_ADDRESS", {}),
            (person.ssn, "US_SSN", {}),
            (person.phone_us, "PHONE_NUMBER", {}),
        ]))
    return text, labels


# ---- DIMENSION 7: TEMPORAL CONSISTENCY (weight 5%) ----
# Tests whether anonymization preserves temporal coherence across
# time-series data and longitudinal records.

def _dim_temporal_medical_timeline(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Longitudinal medical record with dates that must remain temporally coherent."""
    cond = MEDICAL_CONDITIONS[p.idx % len(MEDICAL_CONDITIONS)]
    med1 = MEDICATIONS[p.idx % len(MEDICATIONS)]
    med2 = MEDICATIONS[(p.idx + 3) % len(MEDICATIONS)]
    base_day = 1 + p.idx % 28
    text = (
        f"LONGITUDINAL PATIENT RECORD — {p.full_name} (MRN: {p.mrn})\n\n"
        f"2022-03-{base_day:02d}: Initial consultation. {p.formal_name} (DOB: {p.dob}) "
        f"presented with early symptoms of {cond}. SSN: {p.ssn}.\n\n"
        f"2022-06-{base_day:02d}: Follow-up. {p.first} reports worsening symptoms. "
        f"Started on {med1}. Phone: {p.phone_us}.\n\n"
        f"2023-01-15: 6-month review. {p.first_last_initial} showing improvement. "
        f"Labs within normal range. Added {med2}.\n\n"
        f"2023-07-{base_day:02d}: Annual review. {p.full_name} in remission. "
        f"Medication reduced. Email: {p.email}.\n\n"
        f"2024-03-{base_day:02d}: Relapse detected. {p.formal_name} readmitted. "
        f"Address on file: {p.full_address}.\n\n"
        f"2025-01-{base_day:02d}: Recovery confirmed. {p.first} discharged. "
        f"Insurance: {p.national_id}."
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "full_name", "context_group": "temporal_medical"}),
        (p.mrn, "MEDICAL_RECORD_NUMBER", {}),
        (p.formal_name, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "formal", "context_group": "temporal_medical"}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.ssn, "US_SSN", {}),
        (p.first, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_name", "context_group": "temporal_medical"}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.first_last_initial, "PERSON_NAME", {"cluster": p.cluster_id, "variant": "first_last_initial", "context_group": "temporal_medical"}),
        (p.email, "EMAIL_ADDRESS", {"cluster": p.cluster_id, "variant": "email", "context_group": "temporal_medical"}),
        (p.full_address, "ADDRESS", {}),
        (p.national_id, "NATIONAL_ID", {}),
    ])
    # Repeated full name and formal mentions
    for needle, variant in [(p.full_name, "full_name"), (p.formal_name, "formal"),
                            (p.first, "first_name")]:
        pos = text.find(needle)
        if pos >= 0:
            second = _label(text, needle, "PERSON_NAME", start_search=pos + len(needle),
                            cluster=p.cluster_id, variant=variant, context_group="temporal_medical")
            if second:
                labels.append(second)
    return text, labels


def _dim_temporal_financial_history(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Financial transaction history with temporal ordering that must be preserved."""
    text = (
        f"TRANSACTION HISTORY — Account Holder: {p.full_name}\n"
        f"Account: {p.bank_account} | Card: {p.credit_card_formatted}\n\n"
        f"2024-01-15: Deposit $5,000.00 — Employer: {p.company} (EMP: {p.employee_id})\n"
        f"2024-02-01: Rent payment $1,850.00 to {p.full_address}\n"
        f"2024-03-15: Direct deposit — SSN verified: {p.ssn}\n"
        f"2024-06-01: IBAN transfer to {p.iban} — ref: IMM-{70000 + p.idx}\n"
        f"2024-09-22: Insurance premium — Policy linked to DOB: {p.dob}\n"
        f"2025-01-15: Year-end statement sent to {p.email}\n"
        f"Contact: {p.phone_us} | Notifications: {p.corporate_email}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.bank_account, "BANK_ACCOUNT", {}),
        (p.credit_card_formatted, "CREDIT_CARD", {}),
        (p.company, "ORGANIZATION", {}),
        (p.employee_id, "EMPLOYEE_ID", {}),
        (p.full_address, "ADDRESS", {}),
        (p.ssn, "US_SSN", {}),
        (p.iban, "IBAN", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.corporate_email, "EMAIL_ADDRESS", {}),
    ])
    return text, labels


# ---- DIMENSION 4 BOOST: PII TYPE COVERAGE ----
# Additional templates to boost thin entity types (LICENSE_PLATE, LOCATION,
# MAC_ADDRESS, PASSPORT, NATIONAL_ID, IBAN, ROUTING_NUMBER).

def _dim_pii_coverage_travel_record(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """Travel record with passport, national ID, and international identifiers."""
    text = (
        f"Travel Record for {p.full_name}\n"
        f"Passport: {p.passport} | National ID: {p.national_id}\n"
        f"License Plate (rental): {p.license_plate}\n"
        f"DOB: {p.dob} | Phone: {p.phone_intl}\n"
        f"Hotel WiFi MAC: {p.mac_address}\n"
        f"IBAN for travel expenses: {p.iban}\n"
        f"Routing: {p.routing_number}\n"
        f"Email: {p.email} | Address: {p.full_address}"
    )
    labels = _labels_for(text, [
        (p.full_name, "PERSON_NAME", {}),
        (p.passport, "PASSPORT", {}),
        (p.national_id, "NATIONAL_ID", {}),
        (p.license_plate, "LICENSE_PLATE", {}),
        (p.dob, "DATE_OF_BIRTH", {}),
        (p.phone_intl, "PHONE_NUMBER", {}),
        (p.mac_address, "MAC_ADDRESS", {}),
        (p.iban, "IBAN", {}),
        (p.routing_number, "ROUTING_NUMBER", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.full_address, "ADDRESS", {}),
    ])
    return text, labels


def _dim_pii_coverage_iot_device_log(p: _PersonaFactory) -> tuple[str, list[dict[str, Any]]]:
    """IoT device log with MAC addresses, IPs, and device identifiers."""
    mac2 = ":".join(f"{(p.idx * q) % 256:02x}" for q in [19, 23, 29, 31, 37, 41])
    text = (
        f"[DEVICE LOG] User: {p.username} ({p.full_name})\n"
        f"Primary device MAC: {p.mac_address}\n"
        f"Secondary device MAC: {mac2}\n"
        f"Network IP: {p.ipv4}\n"
        f"Registration email: {p.email}\n"
        f"Owner phone: {p.phone_us}\n"
        f"License plate (geofence): {p.license_plate}\n"
        f"Home address: {p.full_address}"
    )
    labels = _labels_for(text, [
        (p.username, "USERNAME", {}),
        (p.full_name, "PERSON_NAME", {}),
        (p.mac_address, "MAC_ADDRESS", {}),
        (mac2, "MAC_ADDRESS", {}),
        (p.ipv4, "IP_ADDRESS", {}),
        (p.email, "EMAIL_ADDRESS", {}),
        (p.phone_us, "PHONE_NUMBER", {}),
        (p.license_plate, "LICENSE_PLATE", {}),
        (p.full_address, "ADDRESS", {}),
    ])
    return text, labels


# ---------------------------------------------------------------------------
# DIMENSION RECORD GENERATION — 11,500 additional records
# ---------------------------------------------------------------------------

DIMENSION_DISTRIBUTION = {
    # (dimension, count)
    "entity_consistency": 2000,
    "context_preservation": 2000,
    "edge_cases": 2000,
    "format_variations": 1500,
    "temporal_consistency": 1500,
    "pii_type_coverage": 2500,
}
# Total: 11,500 new records
# Note: "multilingual" is covered by generating records in multiple languages.
# All existing records also get tagged with their primary evaluation dimension.


def _generate_dimension_rows(seed: int) -> list[dict[str, Any]]:
    """Generate records targeting specific evaluation dimensions."""
    rng = random.Random(seed + 42)
    rows: list[dict[str, Any]] = []
    counter = 20000  # Start IDs above existing ranges

    # --- Entity Consistency (2000 records) ---
    ec_count = DIMENSION_DISTRIBUTION["entity_consistency"]
    ec_en = int(ec_count * 0.70)
    for i in range(ec_count):
        counter += 1
        lang = "en" if i < ec_en else ["es", "fr", "de", "ja", "ko"][(i - ec_en) % 5]
        p = _PersonaFactory(rng, lang, counter)
        text, labels = _dim_entity_consistency_long_narrative(p, rng)
        rows.append({
            "id": f"DIM-EC-{i:05d}",
            "text": text,
            "language": lang,
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"dimension://entity_consistency-{i}",
            "license": "CC0-1.0",
            "scenario_id": "entity_consistency",
            "entity_cluster_id": p.cluster_id,
            "mention_variant": "multi_variant",
            "context_group": "entity_consistency",
            "size_tier": "very_large",
            "datatype_group": "general_pii",
            "difficulty_level": "hard",
            "evaluation_dimension": "entity_consistency",
        })

    # --- Context Preservation (2000 records) ---
    cp_count = DIMENSION_DISTRIBUTION["context_preservation"]
    cp_en = int(cp_count * 0.70)
    for i in range(cp_count):
        counter += 1
        lang = "en" if i < cp_en else ["es", "fr", "de", "ja"][(i - cp_en) % 4]
        p = _PersonaFactory(rng, lang, counter)
        if i % 2 == 0:
            text, labels = _dim_context_conversation_chat(p, rng)
            ctx = "conversation"
        else:
            text, labels = _dim_context_email_thread(p, rng)
            ctx = "email_thread"
        rows.append({
            "id": f"DIM-CP-{i:05d}",
            "text": text,
            "language": lang,
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"dimension://context_preservation-{i}",
            "license": "CC0-1.0",
            "scenario_id": "context_preservation",
            "entity_cluster_id": p.cluster_id,
            "mention_variant": "multi_variant",
            "context_group": ctx,
            "size_tier": "large",
            "datatype_group": "general_pii",
            "difficulty_level": "hard",
            "evaluation_dimension": "context_preservation",
        })

    # --- Edge Cases (2000 records) ---
    edge_templates = [
        ("overlapping", lambda p, _: _dim_edge_overlapping_entities(p)),
        ("false_positive", lambda p, _: _dim_edge_false_positive_triggers(p)),
        ("dense_pii", lambda p, _: _dim_edge_dense_pii(p)),
        ("unicode", lambda p, _: _dim_edge_unicode_context(p)),
        ("pii_in_url", lambda p, _: _dim_edge_pii_in_url(p)),
        ("code_embedded", lambda p, _: _scenario_embedded_pii_in_code(p)),
    ]
    eg_count = DIMENSION_DISTRIBUTION["edge_cases"]
    for i in range(eg_count):
        counter += 1
        p = _PersonaFactory(rng, "en", counter)
        tmpl_name, tmpl_fn = edge_templates[i % len(edge_templates)]
        text, labels = tmpl_fn(p, rng)
        rows.append({
            "id": f"DIM-EG-{i:05d}",
            "text": text,
            "language": "en",
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"dimension://edge_case-{tmpl_name}-{i}",
            "license": "CC0-1.0",
            "scenario_id": f"edge_case_{tmpl_name}",
            "entity_cluster_id": "none",
            "mention_variant": "none",
            "context_group": f"edge_{tmpl_name}",
            "size_tier": "small" if tmpl_name == "dense_pii" else "medium",
            "datatype_group": "general_pii",
            "difficulty_level": "challenging",
            "evaluation_dimension": "edge_cases",
        })

    # --- Format Variations (1500 records) ---
    format_templates = [
        ("json", lambda p, r: _dim_format_json_record(p)),
        ("csv", lambda p, r: _dim_format_csv_rows(p, r)),
        ("xml", lambda p, r: _dim_format_xml_fragment(p)),
        ("table", lambda p, r: _dim_format_table(p, r)),
    ]
    fv_count = DIMENSION_DISTRIBUTION["format_variations"]
    for i in range(fv_count):
        counter += 1
        p = _PersonaFactory(rng, "en", counter)
        tmpl_name, tmpl_fn = format_templates[i % len(format_templates)]
        text, labels = tmpl_fn(p, rng)
        rows.append({
            "id": f"DIM-FV-{i:05d}",
            "text": text,
            "language": "en",
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"dimension://format_variation-{tmpl_name}-{i}",
            "license": "CC0-1.0",
            "scenario_id": f"format_{tmpl_name}",
            "entity_cluster_id": "none",
            "mention_variant": "none",
            "context_group": f"format_{tmpl_name}",
            "size_tier": "medium",
            "datatype_group": "general_pii",
            "difficulty_level": "moderate",
            "evaluation_dimension": "format_variations",
        })

    # --- Temporal Consistency (1500 records) ---
    tc_count = DIMENSION_DISTRIBUTION["temporal_consistency"]
    tc_en = int(tc_count * 0.75)
    for i in range(tc_count):
        counter += 1
        lang = "en" if i < tc_en else ["es", "fr", "de", "ja"][(i - tc_en) % 4]
        p = _PersonaFactory(rng, lang, counter)
        if i % 2 == 0:
            text, labels = _dim_temporal_medical_timeline(p)
        else:
            text, labels = _dim_temporal_financial_history(p)
        rows.append({
            "id": f"DIM-TC-{i:05d}",
            "text": text,
            "language": lang,
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"dimension://temporal_consistency-{i}",
            "license": "CC0-1.0",
            "scenario_id": "temporal_consistency",
            "entity_cluster_id": p.cluster_id,
            "mention_variant": "multi_variant",
            "context_group": "temporal",
            "size_tier": "large",
            "datatype_group": "general_pii",
            "difficulty_level": "hard",
            "evaluation_dimension": "temporal_consistency",
        })

    # --- PII Type Coverage Boost (2500 records) ---
    coverage_templates = [
        _dim_pii_coverage_travel_record,
        _dim_pii_coverage_iot_device_log,
    ]
    pc_count = DIMENSION_DISTRIBUTION["pii_type_coverage"]
    pc_en = int(pc_count * 0.67)
    for i in range(pc_count):
        counter += 1
        lang = "en" if i < pc_en else ["es", "fr", "de", "it", "pt"][(i - pc_en) % 5]
        p = _PersonaFactory(rng, lang, counter)
        tmpl_fn = coverage_templates[i % len(coverage_templates)]
        text, labels = tmpl_fn(p)
        rows.append({
            "id": f"DIM-PC-{i:05d}",
            "text": text,
            "language": lang,
            "labels": labels,
            "source_type": "synthetic",
            "source_id": f"dimension://pii_type_coverage-{i}",
            "license": "CC0-1.0",
            "scenario_id": "baseline",
            "entity_cluster_id": "none",
            "mention_variant": "none",
            "context_group": "pii_coverage",
            "size_tier": "medium",
            "datatype_group": "general_pii",
            "difficulty_level": "moderate",
            "evaluation_dimension": "pii_type_coverage",
        })

    rng.shuffle(rows)
    return rows


def _assign_evaluation_dimensions(rows: list[dict[str, Any]]) -> None:
    """Assign evaluation_dimension to records that don't already have one.

    Uses heuristics based on existing record metadata to classify each record
    into its primary evaluation dimension.
    """
    for row in rows:
        if row.get("evaluation_dimension"):
            continue

        scenario = row.get("scenario_id", "baseline")
        cluster = row.get("entity_cluster_id", "none")
        size_tier = row.get("size_tier", "medium")
        language = row.get("language", "en")
        ctx = row.get("context_group", "baseline")
        datatype = row.get("datatype_group", "general_pii")
        difficulty = row.get("difficulty_level", "easy")

        # Temporal scenarios
        if "temporal" in scenario or "temporal" in ctx:
            row["evaluation_dimension"] = "temporal_consistency"
        # Context preservation
        elif scenario == "context_loss" or "context_loss" in ctx:
            row["evaluation_dimension"] = "context_preservation"
        # Entity consistency (multi-variant mentions in large docs)
        elif cluster != "none" and size_tier in {"large", "very_large"}:
            row["evaluation_dimension"] = "entity_consistency"
        # Edge cases
        elif "code" in ctx or "mixed_language" in ctx or "negation" in ctx:
            row["evaluation_dimension"] = "edge_cases"
        elif datatype in {"api_key_like_negative", "crypto_wallet"}:
            row["evaluation_dimension"] = "edge_cases"
        elif difficulty == "challenging":
            row["evaluation_dimension"] = "edge_cases"
        # Multilingual
        elif language != "en":
            row["evaluation_dimension"] = "multilingual"
        # Entity tracking
        elif scenario.startswith("continuity_"):
            row["evaluation_dimension"] = "entity_consistency"
        # Format (structured forms)
        elif "structured" in ctx or "form" in str(row.get("source_id", "")):
            row["evaluation_dimension"] = "format_variations"
        # PII type coverage (default for baseline English)
        else:
            row["evaluation_dimension"] = "pii_type_coverage"


# ---------------------------------------------------------------------------
# METADATA & I/O
# ---------------------------------------------------------------------------

def _metadata(rows: list[dict[str, Any]], *, seed: int, dataset: str, version: str) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    by_language: dict[str, int] = {}
    by_scenario: dict[str, int] = {}
    by_size_tier: dict[str, int] = {}
    by_datatype_group: dict[str, int] = {}
    by_difficulty_level: dict[str, int] = {}
    by_eval_dimension: dict[str, int] = {}
    entity_types: dict[str, int] = {}

    for row in rows:
        by_source[row["source_type"]] = by_source.get(row["source_type"], 0) + 1
        by_language[row["language"]] = by_language.get(row["language"], 0) + 1
        by_scenario[row.get("scenario_id", "baseline")] = by_scenario.get(row.get("scenario_id", "baseline"), 0) + 1
        tier = row.get("size_tier", "medium")
        by_size_tier[tier] = by_size_tier.get(tier, 0) + 1
        datatype = str(row.get("datatype_group", "general_pii"))
        by_datatype_group[datatype] = by_datatype_group.get(datatype, 0) + 1
        difficulty = str(row.get("difficulty_level", "moderate"))
        by_difficulty_level[difficulty] = by_difficulty_level.get(difficulty, 0) + 1
        dim = str(row.get("evaluation_dimension", "unclassified"))
        by_eval_dimension[dim] = by_eval_dimension.get(dim, 0) + 1
        for lbl in row.get("labels", []):
            et = lbl.get("entity_type", "UNKNOWN")
            entity_types[et] = entity_types.get(et, 0) + 1

    return {
        "dataset": dataset,
        "version": version,
        "record_count": len(rows),
        "seed": seed,
        "schema": [
            "id", "text", "language", "labels", "source_type", "source_id",
            "license", "scenario_id", "entity_cluster_id", "mention_variant",
            "context_group", "size_tier", "datatype_group", "difficulty_level",
            "evaluation_dimension",
        ],
        "source_distribution": dict(sorted(by_source.items())),
        "language_distribution": dict(sorted(by_language.items())),
        "scenario_distribution": dict(sorted(by_scenario.items())),
        "size_tier_distribution": dict(sorted(by_size_tier.items())),
        "datatype_group_distribution": dict(sorted(by_datatype_group.items())),
        "difficulty_level_distribution": dict(sorted(by_difficulty_level.items())),
        "evaluation_dimension_distribution": dict(sorted(by_eval_dimension.items())),
        "evaluation_dimension_weights": {
            "entity_consistency": 0.20,
            "multilingual": 0.15,
            "context_preservation": 0.20,
            "pii_type_coverage": 0.20,
            "edge_cases": 0.10,
            "format_variations": 0.10,
            "temporal_consistency": 0.05,
        },
        "entity_type_distribution": dict(sorted(entity_types.items())),
        "licenses": {
            "synthetic": "CC0-1.0",
            "curated_public": "CC-BY-4.0",
        },
        "comparable_benchmarks": {
            "ai4privacy_pii_masking_300k": {"records": 300000, "entity_types": 8, "languages": 8},
            "microsoft_pii_bench": {"records": 2842, "entity_types": 55, "languages": 1},
            "tab_anonymization_benchmark": {"records": 1268, "entity_types": "semantic", "languages": 1},
            "nvidia_nemotron_pii": {"records": 100000, "entity_types": 55, "languages": 1},
            "piilo_educational": {"records": 22000, "entity_types": 7, "languages": 1},
            "spy_synthetic": {"records": 8688, "entity_types": 7, "languages": 1},
            "panorama_memorization": {"records": 384789, "entity_types": "diverse", "languages": 1},
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic benchmark datasets (v2)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Generating core dataset (35,700 records)...")
    core_rows = _generate_core_rows(seed=args.seed)
    print(f"  Generated {len(core_rows)} core rows")

    print("Generating tracking dataset (2,800 records)...")
    tracking_rows = _generate_tracking_rows(seed=args.seed)
    print(f"  Generated {len(tracking_rows)} tracking rows")

    print("Generating evaluation dimension records (11,500 records)...")
    dimension_rows = _generate_dimension_rows(seed=args.seed)
    print(f"  Generated {len(dimension_rows)} dimension rows")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Merge core + tracking + dimension into a single unified dataset.
    all_rows = core_rows + tracking_rows + dimension_rows

    # Assign evaluation_dimension to all records that don't have one yet.
    _assign_evaluation_dimensions(all_rows)

    print(f"Writing unified dataset ({len(all_rows)} records)...")
    _write_jsonl(UNIFIED_DATASET_FILE, all_rows)
    unified_meta = _metadata(all_rows, seed=args.seed, dataset="pii_anon_benchmark_v1", version=VERSION)
    UNIFIED_METADATA_FILE.write_text(
        json.dumps(unified_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"\n{'=' * 60}")
    print(f"UNIFIED DATASET: {UNIFIED_DATASET_FILE}")
    print(f"  Records: {len(all_rows)} (core={len(core_rows)}, tracking={len(tracking_rows)}, dimension={len(dimension_rows)})")
    print(f"  Languages: {unified_meta['language_distribution']}")
    print(f"  Sources: {unified_meta['source_distribution']}")
    print(f"  Scenarios: {unified_meta['scenario_distribution']}")
    print(f"  Entity types: {len(unified_meta['entity_type_distribution'])} distinct types")
    for et, count in sorted(unified_meta['entity_type_distribution'].items(), key=lambda x: -x[1]):
        print(f"    {et}: {count}")
    print(f"  Evaluation dimensions: {unified_meta.get('evaluation_dimension_distribution', {})}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
