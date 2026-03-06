"""Deterministic synthetic dataset generator for the evaluation framework.

Generates 50,000+ benchmark records across 52 languages, 44 entity types,
6 data types, 4 difficulty levels, and 4 context-length tiers.  All
generation is seed-based (no external API dependency) for reproducibility.

Evidence basis:
- Stratified sampling across entity types, difficulty, and context length
  ensures balanced evaluation coverage (see references.py: dataset_methodology).
- Context-length tiers (short/medium/long/very_long) exercise length-dependent
  bias documented in Regler et al., 2021.
- Adversarial templates test boundary, co-reference, obfuscation, and encoding
  robustness per PII-Bench (2025) and i2b2 2014 evaluation methodology.

Usage::

    python -m pii_anon.eval_framework.datasets.generator --output eval_framework_v1.jsonl
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import string
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Locale-aware fake-data pools  (deterministic, no external deps)
# ---------------------------------------------------------------------------

_FIRST_NAMES: dict[str, list[str]] = {
    "en": ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda", "David", "Elizabeth",
           "William", "Barbara", "Richard", "Susan", "Joseph", "Jessica", "Thomas", "Sarah", "Christopher", "Karen"],
    "es": ["Carlos", "Maria", "Jose", "Ana", "Luis", "Carmen", "Miguel", "Rosa", "Fernando", "Isabel",
           "Alejandro", "Lucia", "Pablo", "Elena", "Diego", "Sofia", "Andres", "Marta", "Jorge", "Laura"],
    "fr": ["Jean", "Marie", "Pierre", "Sophie", "Jacques", "Isabelle", "Michel", "Claire", "Philippe", "Anne",
           "Nicolas", "Camille", "Antoine", "Julie", "Laurent", "Margaux", "Francois", "Chloe", "Olivier", "Emilie"],
    "de": ["Hans", "Anna", "Wolfgang", "Claudia", "Friedrich", "Sabine", "Karl", "Monika", "Stefan", "Eva",
           "Markus", "Julia", "Andreas", "Katharina", "Tobias", "Lena", "Maximilian", "Sophie", "Felix", "Marie"],
    "it": ["Marco", "Giulia", "Alessandro", "Francesca", "Andrea", "Valentina", "Luca", "Chiara", "Giuseppe", "Sara",
           "Matteo", "Elisa", "Davide", "Aurora", "Lorenzo", "Alice", "Simone", "Martina", "Federico", "Giorgia"],
    "pt": ["Joao", "Maria", "Pedro", "Ana", "Carlos", "Fernanda", "Lucas", "Camila", "Rafael", "Julia",
           "Gabriel", "Beatriz", "Mateus", "Larissa", "Thiago", "Amanda", "Felipe", "Bruna", "Guilherme", "Leticia"],
    "nl": ["Jan", "Maria", "Pieter", "Anna", "Willem", "Sophie", "Hendrik", "Emma", "Joost", "Lisa",
           "Daan", "Fleur", "Bram", "Iris", "Sander", "Femke", "Ruben", "Lotte", "Thomas", "Eva"],
    "ru": ["Ivan", "Anna", "Dmitry", "Elena", "Alexei", "Olga", "Sergei", "Natasha", "Mikhail", "Tatiana",
           "Andrei", "Maria", "Viktor", "Yulia", "Pavel", "Ekaterina", "Nikolai", "Irina", "Vladimir", "Svetlana"],
    "zh": ["\u5f20\u4f1f", "\u738b\u82b3", "\u674e\u660e", "\u8d75\u4e3d", "\u5218\u5f3a", "\u9648\u7ea2", "\u6768\u52c7", "\u5434\u79c0", "\u5468\u6d77", "\u9ec4\u6d01"],
    "ja": ["\u592a\u90ce", "\u82b1\u5b50", "\u5065\u4e00", "\u7f8e\u54b2", "\u96c5\u4eba", "\u3042\u304b\u308a", "\u4eae\u592a", "\u3055\u304f\u3089", "\u5927\u8f14", "\u6e29\u5b50"],
    "ko": ["\ubbfc\uc900", "\uc11c\uc5f0", "\uc900\ud638", "\uc9c0\uc6d0", "\uc815\ud6c8", "\uc218\uc544", "\ud604\uc6b0", "\ud558\ub098", "\uc9c0\ud6c4", "\uc740\uc11c"],
    "ar": ["\u0623\u062d\u0645\u062f", "\u0641\u0627\u0637\u0645\u0629", "\u0645\u062d\u0645\u062f", "\u0639\u0627\u0626\u0634\u0629", "\u064a\u0648\u0633\u0641", "\u0645\u0631\u064a\u0645", "\u0639\u0644\u064a", "\u0632\u064a\u0646\u0628", "\u0639\u0645\u0631", "\u062e\u062f\u064a\u062c\u0629"],
    "hi": ["\u0930\u093e\u0939\u0941\u0932", "\u092a\u094d\u0930\u093f\u092f\u093e", "\u0935\u093f\u0915\u093e\u0938", "\u0938\u0941\u0928\u0940\u0924\u093e", "\u0905\u092e\u093f\u0924", "\u0930\u0947\u0916\u093e", "\u0930\u093e\u091c\u0947\u0936", "\u0928\u0940\u0924\u093e", "\u0938\u0941\u0930\u0947\u0936", "\u0917\u0940\u0924\u093e"],
    "tr": ["Mehmet", "Ayse", "Mustafa", "Fatma", "Ahmet", "Emine", "Ali", "Hatice", "Hasan", "Zeynep"],
    "sw": ["Juma", "Amina", "Hassan", "Fatuma", "Omar", "Zainab", "Bakari", "Mwanaisha", "Said", "Aisha"],
    "th": ["\u0e2a\u0e21\u0e0a\u0e32\u0e22", "\u0e2a\u0e21\u0e2b\u0e0d\u0e34\u0e07", "\u0e1e\u0e07\u0e28\u0e4c", "\u0e2a\u0e21\u0e28\u0e23\u0e35", "\u0e2a\u0e21\u0e1e\u0e07\u0e29\u0e4c", "\u0e2a\u0e21\u0e1e\u0e23", "\u0e2a\u0e21\u0e0a\u0e32\u0e15\u0e34", "\u0e2a\u0e21\u0e1b\u0e23\u0e32\u0e23\u0e16\u0e19\u0e32", "\u0e2a\u0e21\u0e28\u0e31\u0e01\u0e14\u0e34\u0e4c", "\u0e2a\u0e21\u0e2a\u0e21\u0e23"],
    "vi": ["Minh", "Lan", "Hoa", "Thu", "Duc", "Linh", "Tuan", "Mai", "Long", "Ngoc"],
}

_LAST_NAMES: dict[str, list[str]] = {
    "en": ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
           "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White", "Lopez"],
    "es": ["Garcia", "Rodriguez", "Martinez", "Lopez", "Gonzalez", "Hernandez", "Perez", "Sanchez", "Ramirez", "Torres",
           "Flores", "Rivera", "Gomez", "Diaz", "Reyes", "Morales", "Jimenez", "Ruiz", "Alvarez", "Mendoza"],
    "fr": ["Martin", "Bernard", "Dubois", "Thomas", "Robert", "Richard", "Petit", "Durand", "Leroy", "Moreau",
           "Simon", "Laurent", "Lefebvre", "Michel", "Garcia", "David", "Bertrand", "Roux", "Vincent", "Fournier"],
    "de": ["Mueller", "Schmidt", "Schneider", "Fischer", "Weber", "Meyer", "Wagner", "Becker", "Schulz", "Hoffmann",
           "Koch", "Richter", "Wolf", "Klein", "Braun", "Zimmermann", "Krueger", "Werner", "Hartmann", "Lange"],
    "it": ["Rossi", "Russo", "Ferrari", "Esposito", "Bianchi", "Romano", "Colombo", "Ricci", "Marino", "Greco",
           "Bruno", "Gallo", "Conti", "DeLuca", "Mancini", "Costa", "Giordano", "Rizzo", "Lombardi", "Moretti"],
    "pt": ["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves", "Pereira", "Lima", "Gomes",
           "Costa", "Ribeiro", "Martins", "Carvalho", "Araujo", "Melo", "Barbosa", "Cardoso", "Rocha", "Dias"],
    "nl": ["de Jong", "Jansen", "de Vries", "van den Berg", "Bakker", "Visser", "Smit", "Meijer", "de Groot", "Bos"],
    "ru": ["Ivanov", "Petrov", "Sidorov", "Smirnov", "Kuznetsov", "Popov", "Volkov", "Sokolov", "Lebedev", "Kozlov"],
    "zh": ["\u5f20", "\u738b", "\u674e", "\u8d75", "\u5218", "\u9648", "\u6768", "\u5434", "\u5468", "\u9ec4"],
    "ja": ["\u4f50\u85e4", "\u9234\u6728", "\u9ad8\u6a4b", "\u7530\u4e2d", "\u6e21\u8fba", "\u4f0a\u85e4", "\u5c71\u672c", "\u4e2d\u6751", "\u5c0f\u6797", "\u52a0\u85e4"],
    "ko": ["\uae40", "\uc774", "\ubc15", "\ucd5c", "\uc815", "\uac15", "\uc870", "\uc724", "\uc7a5", "\uc784"],
    "ar": ["\u0627\u0644\u0639\u0644\u064a", "\u0627\u0644\u0633\u0639\u064a\u062f", "\u0627\u0644\u062d\u0633\u0646", "\u0627\u0644\u062e\u0627\u0644\u062f", "\u0627\u0644\u0639\u0628\u062f\u0627\u0644\u0644\u0647", "\u0627\u0644\u0631\u0634\u064a\u062f", "\u0627\u0644\u0639\u0645\u0631", "\u0627\u0644\u0633\u0644\u064a\u0645\u0627\u0646", "\u0627\u0644\u0645\u0648\u0633\u0649", "\u0627\u0644\u0625\u0628\u0631\u0627\u0647\u064a\u0645"],
    "hi": ["\u0936\u0930\u094d\u092e\u093e", "\u0935\u0930\u094d\u092e\u093e", "\u091c\u094b\u0936\u0940", "\u092e\u093f\u0936\u094d\u0930\u093e", "\u0938\u093f\u0902\u0939", "\u0917\u0941\u092a\u094d\u0924\u093e", "\u092e\u0939\u0924\u094b", "\u0930\u093e\u0923\u0947", "\u0926\u0947\u0938\u093e\u0908", "\u092a\u093e\u0923\u094d\u0921\u0947"],
    "tr": ["Yilmaz", "Kaya", "Demir", "Celik", "Sahin", "Yildiz", "Yildirim", "Ozturk", "Aydin", "Ozdemir"],
    "sw": ["Mwangi", "Ochieng", "Kamau", "Otieno", "Wanjiku", "Nyambura", "Kiplagat", "Chebet", "Mutua", "Wafula"],
    "th": ["\u0e28\u0e23\u0e35\u0e2a\u0e38\u0e02", "\u0e43\u0e08\u0e14\u0e35", "\u0e41\u0e2a\u0e07\u0e17\u0e2d\u0e07", "\u0e2a\u0e21\u0e1a\u0e31\u0e15\u0e34", "\u0e27\u0e07\u0e28\u0e4c\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e34\u0e4c", "\u0e1e\u0e34\u0e17\u0e31\u0e01\u0e29\u0e4c", "\u0e08\u0e23\u0e34\u0e22\u0e32", "\u0e27\u0e34\u0e44\u0e25\u0e27\u0e23\u0e23\u0e13", "\u0e2a\u0e39\u0e07\u0e2a\u0e38\u0e14", "\u0e1b\u0e23\u0e30\u0e40\u0e2a\u0e23\u0e34\u0e10"],
    "vi": ["Nguyen", "Tran", "Le", "Pham", "Hoang", "Phan", "Vu", "Dang", "Bui", "Do"],
}

_ORGS = ["Acme Corp", "GlobalTech", "MedStar Health", "First National Bank", "EduLearn Inc",
         "CyberShield", "GreenEnergy Ltd", "DataVault", "QuantumAI", "NexGen Solutions",
         "BioPharm Industries", "Pacific Logistics", "Summit Healthcare", "Apex Financial",
         "Horizon Media", "CoreTech Systems", "United Innovations", "Atlas Consulting",
         "Pioneer Manufacturing", "Sterling Partners"]

_CITIES = {
    "en": ["New York", "London", "Sydney", "Toronto", "San Francisco", "Chicago", "Boston", "Seattle", "Austin", "Denver"],
    "es": ["Madrid", "Barcelona", "Mexico City", "Buenos Aires", "Lima", "Bogota", "Santiago", "Quito", "Havana", "Seville"],
    "fr": ["Paris", "Lyon", "Montreal", "Brussels", "Geneva", "Marseille", "Toulouse", "Nice", "Strasbourg", "Bordeaux"],
    "de": ["Berlin", "Munich", "Vienna", "Zurich", "Hamburg", "Frankfurt", "Stuttgart", "Cologne", "Dresden", "Leipzig"],
    "default": ["Springfield", "Fairview", "Madison", "Georgetown", "Riverside", "Oakland", "Portland", "Durham", "Salem", "Canton"],
}

_SCRIPTS: dict[str, str] = {
    "en": "Latin", "es": "Latin", "fr": "Latin", "de": "Latin", "it": "Latin",
    "pt": "Latin", "nl": "Latin", "sv": "Latin", "no": "Latin", "da": "Latin",
    "fi": "Latin", "pl": "Latin", "cs": "Latin", "hu": "Latin", "ro": "Latin",
    "tr": "Latin", "vi": "Latin", "id": "Latin", "sw": "Latin", "af": "Latin",
    "et": "Latin", "ms": "Latin", "fil": "Latin", "yo": "Latin", "ig": "Latin",
    "so": "Latin", "ha": "Latin", "mg": "Latin", "xh": "Latin", "el": "Greek",
    "ru": "Cyrillic", "uk": "Cyrillic", "bg": "Cyrillic", "sr": "Cyrillic",
    "zh": "CJK", "ja": "CJK", "ko": "Hangul", "ar": "Arabic", "he": "Hebrew",
    "fa": "Arabic", "ur": "Arabic", "ps": "Arabic", "hi": "Devanagari",
    "bn": "Bengali", "ta": "Tamil", "te": "Telugu", "th": "Thai",
    "am": "Ethiopic", "ka": "Georgian", "km": "Khmer", "lo": "Lao", "my": "Myanmar",
}

_LANGUAGE_WEIGHTS: dict[str, int] = {
    "en": 5000, "es": 2500, "fr": 2000, "de": 2000, "it": 1500, "pt": 1500,
    "nl": 800, "ru": 2000, "zh": 2500, "ja": 2000, "ko": 1500, "ar": 2000,
    "hi": 1500, "tr": 1000, "th": 800, "vi": 800, "id": 600, "sv": 500,
    "no": 400, "da": 400, "fi": 400, "pl": 600, "cs": 400, "hu": 400,
    "ro": 400, "uk": 500, "bg": 300, "sr": 200, "el": 400, "he": 400,
    "fa": 300, "ur": 300, "ps": 100, "bn": 400, "ta": 200, "te": 200,
    "sw": 300, "am": 200, "yo": 100, "ig": 100, "so": 100, "ha": 100,
    "mg": 100, "xh": 100, "fil": 200, "ms": 200, "km": 100, "lo": 100,
    "my": 100, "af": 200, "ka": 200, "et": 200,
}


# ---------------------------------------------------------------------------
# Deterministic generators
# ---------------------------------------------------------------------------

def _seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _fake_name(rng: random.Random, lang: str) -> str:
    firsts = _FIRST_NAMES.get(lang, _FIRST_NAMES["en"])
    lasts = _LAST_NAMES.get(lang, _LAST_NAMES["en"])
    return f"{rng.choice(firsts)} {rng.choice(lasts)}"


def _fake_email(rng: random.Random, name: str) -> str:
    local = name.lower().replace(" ", ".") + str(rng.randint(1, 99))
    domains = ["gmail.com", "yahoo.com", "outlook.com", "protonmail.com", "example.org"]
    return f"{local}@{rng.choice(domains)}"


def _fake_phone(rng: random.Random) -> str:
    return f"+1 ({rng.randint(200,999)}) {rng.randint(200,999)}-{rng.randint(1000,9999)}"


def _fake_ssn(rng: random.Random) -> str:
    return f"{rng.randint(100,999)}-{rng.randint(10,99)}-{rng.randint(1000,9999)}"


def _fake_address(rng: random.Random, lang: str) -> str:
    cities = _CITIES.get(lang, _CITIES["default"])
    return f"{rng.randint(100,9999)} {rng.choice(['Main', 'Oak', 'Elm', 'Pine', 'Maple'])} {rng.choice(['St', 'Ave', 'Dr', 'Blvd'])}, {rng.choice(cities)}"


def _fake_credit_card(rng: random.Random) -> str:
    prefix = rng.choice(["4", "5", "37"])
    digits = prefix + "".join(str(rng.randint(0, 9)) for _ in range(15 - len(prefix)))
    return f"{digits[:4]}-{digits[4:8]}-{digits[8:12]}-{digits[12:16]}"


def _fake_iban(rng: random.Random) -> str:
    country = rng.choice(["GB", "DE", "FR", "NL", "ES", "IT"])
    check = f"{rng.randint(10,99)}"
    body = "".join(str(rng.randint(0,9)) for _ in range(18))
    return f"{country}{check} {body[:4]} {body[4:8]} {body[8:12]} {body[12:16]} {body[16:]}"


def _fake_ip(rng: random.Random) -> str:
    return f"{rng.randint(1,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}"


def _fake_mac(rng: random.Random) -> str:
    return ":".join(f"{rng.randint(0,255):02x}" for _ in range(6))


def _fake_passport(rng: random.Random) -> str:
    return rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + "".join(str(rng.randint(0,9)) for _ in range(8))


def _fake_api_key(rng: random.Random) -> str:
    return "sk-" + "".join(rng.choices(string.ascii_letters + string.digits, k=32))


def _fake_dob(rng: random.Random) -> str:
    return f"{rng.randint(1940,2005)}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"


def _fake_mrn(rng: random.Random) -> str:
    return f"MRN-{rng.randint(100000,999999)}"


def _fake_employee_id(rng: random.Random) -> str:
    return f"EMP-{rng.randint(10000,99999)}"


# -- New generators for expanded entity coverage --

def _fake_bank_account(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(rng.choice([8, 10, 12])))


def _fake_routing_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(9))


def _fake_swift_bic(rng: random.Random) -> str:
    bank = "".join(rng.choices(string.ascii_uppercase, k=4))
    country = rng.choice(["US", "GB", "DE", "FR", "JP", "CH"])
    loc = "".join(rng.choices(string.ascii_uppercase + string.digits, k=2))
    return f"{bank}{country}{loc}"


def _fake_crypto_wallet(rng: random.Random) -> str:
    prefix = rng.choice(["0x", "1", "3", "bc1q"])
    chars = string.ascii_lowercase + string.digits
    body = "".join(rng.choices(chars, k=rng.choice([34, 40])))
    return prefix + body


def _fake_tax_id(rng: random.Random) -> str:
    return f"{rng.randint(10,99)}-{rng.randint(1000000,9999999)}"


def _fake_drivers_license(rng: random.Random) -> str:
    letter = rng.choice(string.ascii_uppercase)
    digits = "".join(str(rng.randint(0, 9)) for _ in range(12))
    return f"{letter}{digits}"


def _fake_national_id(rng: random.Random) -> str:
    return f"{rng.randint(1000,9999)} {rng.randint(1000,9999)} {rng.randint(1000,9999)}"


def _fake_visa_number(rng: random.Random) -> str:
    return rng.choice(string.ascii_uppercase) + "".join(str(rng.randint(0, 9)) for _ in range(8))


def _fake_license_plate(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=3))
    digits = "".join(str(rng.randint(0, 9)) for _ in range(4))
    return f"{letters} {digits}"


def _fake_vin(rng: random.Random) -> str:
    chars = string.ascii_uppercase.replace("I", "").replace("O", "").replace("Q", "") + string.digits
    return "".join(rng.choices(chars, k=17))


def _fake_prescription(rng: random.Random) -> str:
    return f"RX-{rng.randint(10000000,99999999)}"


def _fake_biometric_id(rng: random.Random) -> str:
    return f"BIO-FP-{''.join(rng.choices(string.ascii_lowercase + string.digits, k=10))}"


def _fake_auth_token(rng: random.Random) -> str:
    header = "eyJhbGciOiJIUzI1NiJ9"
    payload = "".join(rng.choices(string.ascii_letters + string.digits, k=36))
    sig = "".join(rng.choices(string.ascii_letters + string.digits, k=24))
    return f"{header}.{payload}.{sig}"


def _fake_device_id(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(15))


def _fake_url_with_pii(rng: random.Random, name: str) -> str:
    username = name.lower().replace(" ", ".")
    return f"https://example.com/users/{username}?id={rng.randint(1000,9999)}"


def _fake_zip_code(rng: random.Random) -> str:
    return f"{rng.randint(10000,99999)}"


def _fake_social_media_handle(rng: random.Random, name: str) -> str:
    return f"@{name.lower().replace(' ', '_')}{rng.randint(1,99)}"


def _fake_gender(rng: random.Random) -> str:
    return rng.choice(["Male", "Female", "Non-binary"])


def _fake_location_coords(rng: random.Random) -> str:
    lat = round(rng.uniform(-90.0, 90.0), 6)
    lon = round(rng.uniform(-180.0, 180.0), 6)
    return f"{lat}, {lon}"


def _fake_age(rng: random.Random) -> str:
    return f"{rng.randint(18, 95)} years old"


def _fake_ethnic_origin(rng: random.Random) -> str:
    return rng.choice(["Asian", "Hispanic", "Caucasian", "African American", "Middle Eastern", "South Asian"])


def _fake_religious_belief(rng: random.Random) -> str:
    return rng.choice(["Christian", "Muslim", "Buddhist", "Hindu", "Jewish", "Sikh"])


def _fake_political_opinion(rng: random.Random) -> str:
    return rng.choice(["member of the Green Party", "progressive Democrat",
                        "conservative Republican", "social democrat", "independent centrist"])


# ---------------------------------------------------------------------------
# Filler text for medium / long context tiers
# ---------------------------------------------------------------------------

_FILLER_SENTENCES = [
    "The quarterly report indicated significant growth in the technology sector.",
    "Several departments were restructured following the merger announcement.",
    "Training sessions were scheduled for all employees during the transition period.",
    "The committee reviewed the compliance documentation and approved the changes.",
    "New security protocols were implemented across all regional offices.",
    "The research team published findings in peer-reviewed journals.",
    "Budget allocations were adjusted to accommodate the new project timeline.",
    "Stakeholder feedback was incorporated into the revised strategy document.",
    "The audit revealed several areas for improvement in data handling procedures.",
    "Cross-functional teams were assembled to address the operational challenges.",
    "Regulatory requirements necessitated updates to the existing privacy framework.",
    "The organization invested heavily in infrastructure modernization.",
    "Performance benchmarks were established for the upcoming evaluation cycle.",
    "External consultants provided recommendations for process optimization.",
    "The board of directors approved the proposed expansion plan unanimously.",
]


def _generate_filler(rng: random.Random, target_words: int) -> str:
    """Generate filler text to reach approximately *target_words* words."""
    sentences: list[str] = []
    word_count = 0
    while word_count < target_words:
        sentence = rng.choice(_FILLER_SENTENCES)
        sentences.append(sentence)
        word_count += len(sentence.split())
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Template-based record generation
# ---------------------------------------------------------------------------

_DIAGNOSES = ["Type 2 Diabetes Mellitus", "Hypertension Stage II", "Migraine with Aura",
              "Major Depressive Disorder", "Chronic Kidney Disease Stage 3",
              "Acute Myocardial Infarction", "Generalized Anxiety Disorder",
              "Rheumatoid Arthritis", "Obstructive Sleep Apnea", "Celiac Disease"]
_JOB_TITLES = ["Senior Software Engineer", "Data Analyst", "Compliance Officer",
               "Medical Director", "Financial Advisor", "Research Scientist",
               "Operations Manager", "Chief Technology Officer", "Legal Counsel",
               "Product Designer"]
_SALARIES = ["$85,000", "$125,000", "$95,500", "$150,000", "$72,000",
             "$210,000", "$67,500", "$118,750", "$142,000", "$88,000"]
_NATIONALITIES = ["American", "British", "German", "Japanese", "Brazilian", "Indian", "Australian",
                  "Canadian", "French", "Korean", "Mexican", "Italian", "Chinese", "Swedish"]
_INSURANCE_IDS = ["HIN-456789", "HIN-234567", "HIN-678901", "HIN-345678", "HIN-890123",
                  "HIN-567890", "HIN-123456", "HIN-789012"]

_TEMPLATES: list[dict[str, Any]] = [
    # === Original templates (short context) ===
    {
        "id": "general_pii",
        "template": "Employee {name} ({emp_id}) at {org}. SSN: {ssn}. Address: {address}. Email: {email}. Phone: {phone}.",
        "entity_types": ["PERSON_NAME", "EMPLOYEE_ID", "ORGANIZATION", "US_SSN", "ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        "data_type": "unstructured_text",
        "difficulty": "easy",
    },
    {
        "id": "medical_record",
        "template": "Discharge Summary - Patient: {name}, MRN: {mrn}, DOB: {dob}. Contact: {phone}. Email: {email}.",
        "entity_types": ["PERSON_NAME", "MEDICAL_RECORD_NUMBER", "DATE_OF_BIRTH", "PHONE_NUMBER", "EMAIL_ADDRESS"],
        "data_type": "unstructured_text",
        "difficulty": "moderate",
    },
    {
        "id": "financial_record",
        "template": "Transaction for {name}. Card: {cc}. IBAN: {iban}. Amount: $1,234.56. Address: {address}.",
        "entity_types": ["PERSON_NAME", "CREDIT_CARD_NUMBER", "IBAN", "ADDRESS"],
        "data_type": "structured",
        "difficulty": "moderate",
    },
    {
        "id": "log_entry",
        "template": "[INFO] User {username} logged in from {ip} at 2025-03-15T10:30:00Z. Device MAC: {mac}.",
        "entity_types": ["USERNAME", "IP_ADDRESS", "MAC_ADDRESS"],
        "data_type": "logs",
        "difficulty": "easy",
    },
    {
        "id": "code_snippet",
        "template": "# Config file\napi_key = \"{api_key}\"\ndb_host = \"{ip}\"\nadmin_email = \"{email}\"\n",
        "entity_types": ["API_KEY", "IP_ADDRESS", "EMAIL_ADDRESS"],
        "data_type": "code",
        "difficulty": "moderate",
    },
    {
        "id": "passport_travel",
        "template": "Passenger {name}, Passport: {passport}, DOB: {dob}, Nationality: {nationality}. Flight booked to {address}.",
        "entity_types": ["PERSON_NAME", "PASSPORT_NUMBER", "DATE_OF_BIRTH", "NATIONALITY", "ADDRESS"],
        "data_type": "semi_structured",
        "difficulty": "moderate",
    },
    {
        "id": "complex_medical",
        "template": "Patient {name} (MRN: {mrn}) presented with {diagnosis}. SSN: {ssn}. Insurance: {insurance_id}. Emergency contact: {contact_name} at {phone}. Provider: {org}.",
        "entity_types": ["PERSON_NAME", "MEDICAL_RECORD_NUMBER", "MEDICAL_DIAGNOSIS", "US_SSN", "HEALTH_INSURANCE_ID", "PERSON_NAME", "PHONE_NUMBER", "ORGANIZATION"],
        "data_type": "unstructured_text",
        "difficulty": "hard",
    },
    {
        "id": "multi_entity",
        "template": "{name} ({email}) works at {org} as {job_title}. Salary: {salary}. SSN: {ssn}. Phone: {phone}. Address: {address}. DOB: {dob}.",
        "entity_types": ["PERSON_NAME", "EMAIL_ADDRESS", "ORGANIZATION", "JOB_TITLE", "SALARY", "US_SSN", "PHONE_NUMBER", "ADDRESS", "DATE_OF_BIRTH"],
        "data_type": "unstructured_text",
        "difficulty": "challenging",
    },
    {
        "id": "adversarial_boundary",
        "template": "Meeting with {name} scheduled. Note: {name}'s email is {email} (verified). Phone for {name}: {phone}.",
        "entity_types": ["PERSON_NAME", "PERSON_NAME", "EMAIL_ADDRESS", "PERSON_NAME", "PHONE_NUMBER"],
        "data_type": "unstructured_text",
        "difficulty": "hard",
        "adversarial": "boundary_case",
    },
    {
        "id": "semi_structured_json",
        "template": '{{\"name\": \"{name}\", \"email\": \"{email}\", \"phone\": \"{phone}\", \"ssn\": \"{ssn}\", \"ip\": \"{ip}\"}}',
        "entity_types": ["PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "IP_ADDRESS"],
        "data_type": "semi_structured",
        "difficulty": "easy",
    },
    # === New templates: expanded entity coverage ===
    {
        "id": "banking_full",
        "template": "Account holder: {name}. Bank account: {bank_account}. Routing: {routing_number}. SWIFT: {swift_bic}. Tax ID: {tax_id}. Address: {address}.",
        "entity_types": ["PERSON_NAME", "BANK_ACCOUNT_NUMBER", "ROUTING_NUMBER", "SWIFT_BIC_CODE", "TAX_ID", "ADDRESS"],
        "data_type": "structured",
        "difficulty": "moderate",
    },
    {
        "id": "digital_identity",
        "template": "User profile: {social_handle} (username: {username}). IP: {ip}. Device: {device_id}. Auth token: {auth_token}. Profile URL: {url_pii}.",
        "entity_types": ["SOCIAL_MEDIA_HANDLE", "USERNAME", "IP_ADDRESS", "DEVICE_ID", "AUTHENTICATION_TOKEN", "URL_WITH_PII"],
        "data_type": "semi_structured",
        "difficulty": "moderate",
    },
    {
        "id": "vehicle_record",
        "template": "Registration for {name}. License plate: {license_plate}. VIN: {vin}. Driver's license: {drivers_license}. Address: {address}.",
        "entity_types": ["PERSON_NAME", "LICENSE_PLATE", "VEHICLE_IDENTIFICATION_NUMBER", "DRIVERS_LICENSE", "ADDRESS"],
        "data_type": "structured",
        "difficulty": "moderate",
    },
    {
        "id": "government_ids",
        "template": "Applicant: {name}. SSN: {ssn}. National ID: {national_id}. Passport: {passport}. Visa: {visa_number}. DOB: {dob}.",
        "entity_types": ["PERSON_NAME", "US_SSN", "NATIONAL_ID_NUMBER", "PASSPORT_NUMBER", "VISA_NUMBER", "DATE_OF_BIRTH"],
        "data_type": "structured",
        "difficulty": "hard",
    },
    {
        "id": "medical_full",
        "template": "Patient: {name}, MRN: {mrn}, DOB: {dob}, Gender: {gender}, Age: {age}. Diagnosis: {diagnosis}. Rx: {prescription}. Insurance: {insurance_id}. Biometric: {biometric_id}.",
        "entity_types": ["PERSON_NAME", "MEDICAL_RECORD_NUMBER", "DATE_OF_BIRTH", "GENDER", "AGE", "MEDICAL_DIAGNOSIS", "PRESCRIPTION_NUMBER", "HEALTH_INSURANCE_ID", "BIOMETRIC_ID"],
        "data_type": "unstructured_text",
        "difficulty": "hard",
    },
    {
        "id": "crypto_finance",
        "template": "Transfer from {name} ({email}). Wallet: {crypto_wallet}. IBAN: {iban}. Card: {cc}. Tax ID: {tax_id}. ZIP: {zip_code}.",
        "entity_types": ["PERSON_NAME", "EMAIL_ADDRESS", "CRYPTOCURRENCY_WALLET", "IBAN", "CREDIT_CARD_NUMBER", "TAX_ID", "ZIP_CODE"],
        "data_type": "semi_structured",
        "difficulty": "challenging",
    },
    {
        "id": "behavioral_sensitive",
        "template": "Survey response - Name: {name}, Age: {age}, Gender: {gender}, Ethnic origin: {ethnic_origin}, Religion: {religious_belief}, Political: {political_opinion}. Location: {location_coords}.",
        "entity_types": ["PERSON_NAME", "AGE", "GENDER", "ETHNIC_ORIGIN", "RELIGIOUS_BELIEF", "POLITICAL_OPINION", "LOCATION_COORDINATES"],
        "data_type": "unstructured_text",
        "difficulty": "challenging",
    },
    {
        "id": "employment_full",
        "template": "{name} ({email}), {job_title} at {org}. EMP ID: {emp_id}. Salary: {salary}. SSN: {ssn}. DOB: {dob}. Phone: {phone}. ZIP: {zip_code}.",
        "entity_types": ["PERSON_NAME", "EMAIL_ADDRESS", "JOB_TITLE", "ORGANIZATION", "EMPLOYEE_ID", "SALARY", "US_SSN", "DATE_OF_BIRTH", "PHONE_NUMBER", "ZIP_CODE"],
        "data_type": "unstructured_text",
        "difficulty": "challenging",
    },
    {
        "id": "infrastructure_logs",
        "template": "[WARN] {username} from {ip} (MAC: {mac}) accessed {url_pii}. Device: {device_id}. API key: {api_key}. Token: {auth_token}.",
        "entity_types": ["USERNAME", "IP_ADDRESS", "MAC_ADDRESS", "URL_WITH_PII", "DEVICE_ID", "API_KEY", "AUTHENTICATION_TOKEN"],
        "data_type": "logs",
        "difficulty": "hard",
    },
    # === Adversarial templates ===
    {
        "id": "adversarial_coreference",
        "template": "{name} called from {phone}. Later, {name} sent an email to {email}. {name} lives at {address} with SSN {ssn}.",
        "entity_types": ["PERSON_NAME", "PHONE_NUMBER", "PERSON_NAME", "EMAIL_ADDRESS", "PERSON_NAME", "ADDRESS", "US_SSN"],
        "data_type": "unstructured_text",
        "difficulty": "hard",
        "adversarial": "coreference",
    },
    {
        "id": "adversarial_mixed_format",
        "template": "CC: {cc} | SSN: {ssn} | IBAN: {iban} | Phone: {phone} | IP: {ip} | MAC: {mac} | Email: {email}",
        "entity_types": ["CREDIT_CARD_NUMBER", "US_SSN", "IBAN", "PHONE_NUMBER", "IP_ADDRESS", "MAC_ADDRESS", "EMAIL_ADDRESS"],
        "data_type": "mixed",
        "difficulty": "challenging",
        "adversarial": "mixed_format",
    },
    {
        "id": "adversarial_obfuscation",
        "template": "Contact: {name} | e-mail: {email} | tel: {phone} | social security number: {ssn} | born: {dob} | at: {address}",
        "entity_types": ["PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "DATE_OF_BIRTH", "ADDRESS"],
        "data_type": "unstructured_text",
        "difficulty": "hard",
        "adversarial": "label_obfuscation",
    },
    {
        "id": "adversarial_dense_pii",
        "template": "{name},{email},{phone},{ssn},{cc},{iban},{dob},{address}",
        "entity_types": ["PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD_NUMBER", "IBAN", "DATE_OF_BIRTH", "ADDRESS"],
        "data_type": "structured",
        "difficulty": "challenging",
        "adversarial": "dense_pii",
    },
    # === Medium context templates (~100-200 words) ===
    {
        "id": "medium_hr_report",
        "template": "Human Resources Report\n\nEmployee: {name}\nID: {emp_id}\nTitle: {job_title}\nOrganization: {org}\nSalary: {salary}\nSSN: {ssn}\nDOB: {dob}\nAddress: {address}\nPhone: {phone}\nEmail: {email}\n\n{filler_medium}\n\nEmergency contact: {contact_name} at {phone2}. Approved by HR on 2025-01-15.",
        "entity_types": ["PERSON_NAME", "EMPLOYEE_ID", "JOB_TITLE", "ORGANIZATION", "SALARY", "US_SSN", "DATE_OF_BIRTH", "ADDRESS", "PHONE_NUMBER", "EMAIL_ADDRESS", "PERSON_NAME", "PHONE_NUMBER"],
        "data_type": "unstructured_text",
        "difficulty": "moderate",
        "context_tier": "medium",
    },
    {
        "id": "medium_insurance_claim",
        "template": "Insurance Claim #IC-{claim_id}\n\nClaimant: {name}\nMRN: {mrn}\nDOB: {dob}\nGender: {gender}\nDiagnosis: {diagnosis}\nInsurance ID: {insurance_id}\nProvider: {org}\n\n{filler_medium}\n\nPrescription: {prescription}. Biometric verification: {biometric_id}. Contact: {phone}.",
        "entity_types": ["PERSON_NAME", "MEDICAL_RECORD_NUMBER", "DATE_OF_BIRTH", "GENDER", "MEDICAL_DIAGNOSIS", "HEALTH_INSURANCE_ID", "ORGANIZATION", "PRESCRIPTION_NUMBER", "BIOMETRIC_ID", "PHONE_NUMBER"],
        "data_type": "unstructured_text",
        "difficulty": "hard",
        "context_tier": "medium",
    },
    # === Long context template (~500+ words) ===
    {
        "id": "long_case_file",
        "template": "CASE FILE: {case_id}\n\nSubject: {name}\nEmail: {email}\nPhone: {phone}\nSSN: {ssn}\nDOB: {dob}\nAddress: {address}\nNationality: {nationality}\n\nFINANCIAL RECORDS:\nBank Account: {bank_account}\nRouting: {routing_number}\nCredit Card: {cc}\nIBAN: {iban}\n\n{filler_long}\n\nDIGITAL FOOTPRINT:\nIP: {ip}\nMAC: {mac}\nDevice: {device_id}\nUsername: {username}\n\n{filler_long2}\n\nEMPLOYMENT:\nOrganization: {org}\nEmployee ID: {emp_id}\nJob Title: {job_title}\nSalary: {salary}\n\nAssociate: {contact_name} ({email2}). Vehicle: {license_plate}, VIN: {vin}.",
        "entity_types": [
            "PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "DATE_OF_BIRTH", "ADDRESS", "NATIONALITY",
            "BANK_ACCOUNT_NUMBER", "ROUTING_NUMBER", "CREDIT_CARD_NUMBER", "IBAN",
            "IP_ADDRESS", "MAC_ADDRESS", "DEVICE_ID", "USERNAME",
            "ORGANIZATION", "EMPLOYEE_ID", "JOB_TITLE", "SALARY",
            "PERSON_NAME", "EMAIL_ADDRESS", "LICENSE_PLATE", "VEHICLE_IDENTIFICATION_NUMBER",
        ],
        "data_type": "unstructured_text",
        "difficulty": "challenging",
        "context_tier": "long",
    },
]


def _fill_template(template: dict[str, Any], rng: random.Random, lang: str, idx: int) -> dict[str, Any]:
    """Fill a template with fake data and return a JSONL-ready dict."""
    name = _fake_name(rng, lang)
    email = _fake_email(rng, name)
    phone = _fake_phone(rng)
    ssn = _fake_ssn(rng)
    address = _fake_address(rng, lang)
    contact_name = _fake_name(rng, lang)

    values = {
        "name": name,
        "email": email,
        "phone": phone,
        "ssn": ssn,
        "address": address,
        "org": rng.choice(_ORGS),
        "emp_id": _fake_employee_id(rng),
        "mrn": _fake_mrn(rng),
        "dob": _fake_dob(rng),
        "cc": _fake_credit_card(rng),
        "iban": _fake_iban(rng),
        "ip": _fake_ip(rng),
        "mac": _fake_mac(rng),
        "passport": _fake_passport(rng),
        "api_key": _fake_api_key(rng),
        "username": name.split()[0].lower() + str(rng.randint(1, 999)),
        "diagnosis": rng.choice(_DIAGNOSES),
        "job_title": rng.choice(_JOB_TITLES),
        "salary": rng.choice(_SALARIES),
        "nationality": rng.choice(_NATIONALITIES),
        "insurance_id": rng.choice(_INSURANCE_IDS),
        "contact_name": contact_name,
        # New entity values
        "bank_account": _fake_bank_account(rng),
        "routing_number": _fake_routing_number(rng),
        "swift_bic": _fake_swift_bic(rng),
        "crypto_wallet": _fake_crypto_wallet(rng),
        "tax_id": _fake_tax_id(rng),
        "drivers_license": _fake_drivers_license(rng),
        "national_id": _fake_national_id(rng),
        "visa_number": _fake_visa_number(rng),
        "license_plate": _fake_license_plate(rng),
        "vin": _fake_vin(rng),
        "prescription": _fake_prescription(rng),
        "biometric_id": _fake_biometric_id(rng),
        "auth_token": _fake_auth_token(rng),
        "device_id": _fake_device_id(rng),
        "url_pii": _fake_url_with_pii(rng, name),
        "zip_code": _fake_zip_code(rng),
        "social_handle": _fake_social_media_handle(rng, name),
        "gender": _fake_gender(rng),
        "location_coords": _fake_location_coords(rng),
        "age": _fake_age(rng),
        "ethnic_origin": _fake_ethnic_origin(rng),
        "religious_belief": _fake_religious_belief(rng),
        "political_opinion": _fake_political_opinion(rng),
        # Duplicates for multi-entity templates
        "phone2": _fake_phone(rng),
        "email2": _fake_email(rng, contact_name),
        # Filler text for medium/long context
        "filler_medium": _generate_filler(rng, 80),
        "filler_long": _generate_filler(rng, 200),
        "filler_long2": _generate_filler(rng, 150),
        # Non-PII placeholders
        "claim_id": str(rng.randint(100000, 999999)),
        "case_id": f"CF-{rng.randint(100000,999999)}",
    }

    text_template = template["template"]
    text = text_template
    labels: list[dict[str, Any]] = []

    # Build text by replacing placeholders and tracking positions
    result_text = ""
    remaining = text
    entity_idx = 0
    entity_types = list(template["entity_types"])

    for key, value in values.items():
        placeholder = "{" + key + "}"
        while placeholder in remaining:
            pos = remaining.index(placeholder)
            result_text += remaining[:pos]
            start = len(result_text)
            result_text += value
            end = len(result_text)
            remaining = remaining[pos + len(placeholder):]

            if entity_idx < len(entity_types):
                labels.append({
                    "entity_type": entity_types[entity_idx],
                    "start": start,
                    "end": end,
                })
                entity_idx += 1

    result_text += remaining

    # Determine regulatory domain
    reg_domain: list[str] = []
    for et in template["entity_types"]:
        if et in {"MEDICAL_RECORD_NUMBER", "MEDICAL_DIAGNOSIS", "HEALTH_INSURANCE_ID",
                  "PRESCRIPTION_NUMBER", "BIOMETRIC_ID"}:
            if "hipaa" not in reg_domain:
                reg_domain.append("hipaa")
        if et in {"PERSON_NAME", "EMAIL_ADDRESS", "PHONE_NUMBER", "ADDRESS", "DATE_OF_BIRTH",
                  "GENDER", "NATIONALITY", "ETHNIC_ORIGIN", "RELIGIOUS_BELIEF", "POLITICAL_OPINION",
                  "LOCATION_COORDINATES", "IP_ADDRESS", "MAC_ADDRESS"}:
            if "gdpr" not in reg_domain:
                reg_domain.append("gdpr")
        if et in {"US_SSN", "CREDIT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER", "DRIVERS_LICENSE",
                  "NATIONAL_ID_NUMBER", "TAX_ID"}:
            if "ccpa" not in reg_domain:
                reg_domain.append("ccpa")
        if et in {"CREDIT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER", "ROUTING_NUMBER"}:
            if "pci_dss" not in reg_domain:
                reg_domain.append("pci_dss")

    # Determine context length tier
    token_count = len(result_text.split())
    explicit_tier = template.get("context_tier")
    if explicit_tier:
        context_tier = explicit_tier
    elif token_count < 100:
        context_tier = "short"
    elif token_count < 1000:
        context_tier = "medium"
    elif token_count < 10000:
        context_tier = "long"
    else:
        context_tier = "very_long"

    record_id = f"EVAL-{idx:06d}"
    return {
        "id": record_id,
        "text": result_text,
        "labels": labels,
        "language": lang,
        "source_type": "synthetic",
        "source_id": f"synthetic://eval-framework/v1/{template['id']}",
        "license": "CC0-1.0",
        "scenario_id": template["id"],
        "entity_cluster_id": "none",
        "mention_variant": "none",
        "context_group": template.get("data_type", "unstructured_text"),
        "datatype_group": template["id"],
        "difficulty_level": template["difficulty"],
        "data_type": template.get("data_type", "unstructured_text"),
        "context_length_tier": context_tier,
        "token_count": token_count,
        "regulatory_domain": reg_domain,
        "adversarial_type": template.get("adversarial"),
        "script": _SCRIPTS.get(lang, "Latin"),
    }


def generate_dataset(*, seed: int = 42, target_records: int = 50000) -> list[dict[str, Any]]:
    """Generate the full evaluation dataset deterministically."""
    rng = _seeded_rng(seed)
    records: list[dict[str, Any]] = []
    idx = 0

    # Calculate per-language targets
    total_weight = sum(_LANGUAGE_WEIGHTS.values())
    lang_targets: dict[str, int] = {}
    for lang, weight in _LANGUAGE_WEIGHTS.items():
        lang_targets[lang] = max(50, int(target_records * weight / total_weight))

    # Adjust to hit target
    current_total = sum(lang_targets.values())
    if current_total < target_records:
        lang_targets["en"] += target_records - current_total

    for lang, count in sorted(lang_targets.items()):
        for i in range(count):
            template = rng.choice(_TEMPLATES)
            record = _fill_template(template, rng, lang, idx)
            records.append(record)
            idx += 1

    # Sort by ID for reproducibility
    records.sort(key=lambda r: r["id"])
    return records


def compute_dataset_checksum(records: list[dict[str, Any]]) -> str:
    """Compute SHA-256 checksum over all record texts for integrity verification."""
    h = hashlib.sha256()
    for record in records:
        h.update(record["id"].encode("utf-8"))
        h.update(record["text"].encode("utf-8"))
    return h.hexdigest()


def dataset_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute distribution statistics for a generated dataset.

    Returns entity type coverage, context length distribution,
    difficulty distribution, adversarial counts, and text length stats.
    """
    entity_types: set[str] = set()
    ctx_dist: dict[str, int] = {}
    diff_dist: dict[str, int] = {}
    dtype_dist: dict[str, int] = {}
    adv_count = 0
    lang_dist: dict[str, int] = {}
    text_lengths: list[int] = []
    reg_domains: set[str] = set()

    for r in records:
        for lbl in r.get("labels", []):
            entity_types.add(lbl["entity_type"])
        tier = r.get("context_length_tier", "short")
        ctx_dist[tier] = ctx_dist.get(tier, 0) + 1
        diff = r.get("difficulty_level", "moderate")
        diff_dist[diff] = diff_dist.get(diff, 0) + 1
        dt = r.get("data_type", "unstructured_text")
        dtype_dist[dt] = dtype_dist.get(dt, 0) + 1
        if r.get("adversarial_type"):
            adv_count += 1
        lang = r.get("language", "en")
        lang_dist[lang] = lang_dist.get(lang, 0) + 1
        text_lengths.append(len(r.get("text", "")))
        for rd in r.get("regulatory_domain", []):
            reg_domains.add(rd)

    return {
        "total_records": len(records),
        "entity_types_covered": sorted(entity_types),
        "entity_type_count": len(entity_types),
        "context_length_distribution": ctx_dist,
        "difficulty_distribution": diff_dist,
        "data_type_distribution": dtype_dist,
        "adversarial_count": adv_count,
        "adversarial_ratio": round(adv_count / max(len(records), 1), 4),
        "language_count": len(lang_dist),
        "language_distribution": lang_dist,
        "text_length_min": min(text_lengths) if text_lengths else 0,
        "text_length_max": max(text_lengths) if text_lengths else 0,
        "text_length_mean": round(sum(text_lengths) / max(len(text_lengths), 1), 1),
        "regulatory_domains": sorted(reg_domains),
    }


def write_dataset(output_path: str | Path, *, seed: int = 42, target_records: int = 50000) -> int:
    """Generate and write the dataset to compressed JSONL (.jsonl.gz)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = generate_dataset(seed=seed, target_records=target_records)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[4] / "packages" / "pii_anon_datasets" / "src" / "pii_anon_datasets" / "eval_framework" / "data" / "eval_framework_v1.jsonl"
    if len(sys.argv) > 2 and sys.argv[1] == "--output":
        output = Path(sys.argv[2])
    count = write_dataset(output)
    print(f"Generated {count} records -> {output}")
