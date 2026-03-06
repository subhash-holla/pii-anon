"""Pure-function validators for PII pattern candidates.

Each validator implements a well-known checksum or format rule to distinguish
genuine PII from coincidental regex matches.  Validated matches receive higher
confidence scores, while format-only matches receive medium confidence — giving
downstream consumers the information they need for precision/recall tuning.

Algorithms implemented
----------------------
- **Luhn mod-10** — ISO/IEC 7812-1 check digit for credit cards and Canadian SIN.
- **IBAN mod-97** — ISO 13616-1 checksum for International Bank Account Numbers.
- **ABA routing** — Weighted-sum mod-10 (weights 3-7-1) per Federal Reserve spec.
- **SSN area rules** — USCIS area-number restrictions (000, 666, 900+ invalid).
- **VIN check digit** — NHTSA position-9 transliteration algorithm (49 CFR 565).
- **Verhoeff** — Dihedral group D₅ anti-symmetric checksum for Indian Aadhaar UIDs.
- **IPv4 octet range** — 0–255 per octet.
- **Credit-card issuer prefix** — Visa (4), MC (51-55/2221-2720), Amex (34/37),
  Discover (6011/65/644-649), Diners (30/36/38).
- **NPI Luhn** — National Provider Identifier uses Luhn with prefix "80840".

References
----------
.. [1] ISO/IEC 7812-1:2017 — Identification cards, Numbering system.
.. [2] ISO 13616-1:2020 — Financial services, IBAN.
.. [3] 49 CFR § 565.15 — Vehicle Identification Number requirements.
.. [4] Verhoeff, J. (1969). *Error Detecting Decimal Codes*.
.. [5] CMS NPI Standard — 45 CFR 162.408.
"""

from __future__ import annotations

import re

# Street-address suffixes used by ``looks_like_address_phrase``.
ADDRESS_SUFFIXES: frozenset[str] = frozenset({
    "street", "st", "avenue", "ave", "road", "rd", "way",
    "boulevard", "blvd", "lane", "ln", "circle", "terrace", "drive", "dr",
})


# ---------------------------------------------------------------------------
# IPv4
# ---------------------------------------------------------------------------


def is_valid_ipv4(candidate: str) -> bool:
    """Return *True* if *candidate* is a syntactically valid IPv4 address.

    Each dot-separated octet must be a decimal integer in [0, 255].
    """
    parts = candidate.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        val = int(part)
        if val < 0 or val > 255:
            return False
    return True


# ---------------------------------------------------------------------------
# Luhn (ISO/IEC 7812-1)
# ---------------------------------------------------------------------------


def luhn_checksum(digits: str) -> bool:
    """Validate a digit string using the Luhn mod-10 algorithm.

    Used for credit-card numbers, Canadian SIN, and NPI validation.
    Returns *True* when ``sum % 10 == 0``.
    """
    total = 0
    reverse_digits = digits[::-1]
    for index, char in enumerate(reverse_digits):
        val = int(char)
        if index % 2 == 1:
            val *= 2
            if val > 9:
                val -= 9
        total += val
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Credit-card issuer prefixes
# ---------------------------------------------------------------------------


def is_cc_format(digits: str) -> bool:
    """Check if *digits* match a known credit-card issuer prefix format.

    Supports Visa (4), Mastercard (51-55, 2221-2720), Amex (34, 37),
    Discover (6011, 65, 644-649), and Diners Club (30, 36, 38).
    """
    if len(digits) < 13 or len(digits) > 19:
        return False
    first = digits[0]
    first_two = digits[:2]
    first_four = digits[:4]
    if first == "4":
        return True
    if first_two in ("34", "37"):
        return True
    if first_two in ("51", "52", "53", "54", "55"):
        return True
    if 2221 <= int(first_four) <= 2720:
        return True
    if first_four == "6011" or first_two == "65":
        return True
    first_three = digits[:3]
    if 644 <= int(first_three) <= 649:
        return True
    if first == "3" and first_two in ("30", "36", "38"):
        return True
    return False


def is_valid_credit_card(candidate: str) -> bool:
    """Validate via Luhn checksum **or** issuer-prefix format match."""
    digits = "".join(ch for ch in candidate if ch.isdigit())
    if len(digits) < 13 or len(digits) > 19:
        return False
    if luhn_checksum(digits):
        return True
    return is_cc_format(digits)


# ---------------------------------------------------------------------------
# IBAN (ISO 13616-1)
# ---------------------------------------------------------------------------


def is_valid_iban_strict(candidate: str) -> bool:
    """Validate IBAN using the mod-97 checksum (strict)."""
    compact = "".join(ch for ch in candidate if ch.isalnum()).upper()
    if len(compact) < 15 or len(compact) > 34:
        return False
    if not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    rearranged = compact[4:] + compact[:4]
    converted = []
    for char in rearranged:
        if char.isdigit():
            converted.append(char)
        elif char.isalpha():
            converted.append(str(ord(char) - 55))
        else:
            return False
    try:
        return int("".join(converted)) % 97 == 1
    except Exception:
        return False


def is_valid_iban_format(candidate: str) -> bool:
    """Validate IBAN format only (no checksum) — fallback for synthetic data."""
    compact = "".join(ch for ch in candidate if ch.isalnum()).upper()
    if len(compact) < 15 or len(compact) > 34:
        return False
    if not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    return all(ch.isalnum() for ch in compact[4:])


def is_valid_iban(candidate: str) -> bool:
    """Validate via mod-97 checksum **or** format-only match."""
    if is_valid_iban_strict(candidate):
        return True
    return is_valid_iban_format(candidate)


# ---------------------------------------------------------------------------
# US Social Security Number area rules
# ---------------------------------------------------------------------------


def is_valid_ssn_digits(candidate: str) -> bool:
    """Validate a 9-digit SSN (no separators) using USCIS area/group/serial rules.

    Invalid areas: 000, 666, 900-999.  Group and serial cannot be all zeros.
    """
    if len(candidate) != 9 or not candidate.isdigit():
        return False
    area = int(candidate[:3])
    if area == 0 or area == 666 or area >= 900:
        return False
    group = int(candidate[3:5])
    serial = int(candidate[5:9])
    if group == 0 or serial == 0:
        return False
    return True


# ---------------------------------------------------------------------------
# ABA Routing Number
# ---------------------------------------------------------------------------


def is_valid_aba_routing(candidate: str) -> bool:
    """Validate ABA routing number via weighted-sum mod-10 checksum.

    Weights cycle as 3, 7, 1 across the 9 digits.  The Federal Reserve
    assigns routing numbers such that the weighted sum is divisible by 10.
    """
    if len(candidate) != 9 or not candidate.isdigit():
        return False
    weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
    total = sum(int(candidate[i]) * weights[i] for i in range(9))
    return total % 10 == 0


# ---------------------------------------------------------------------------
# VIN Check Digit (NHTSA — 49 CFR § 565)
# ---------------------------------------------------------------------------

# Transliteration table mapping VIN characters to numeric values.
_VIN_VALUES: dict[str, int] = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
    "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9,
    "S": 2, "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9,
}
for _d in "0123456789":
    _VIN_VALUES[_d] = int(_d)

# Positional weight factors (position 9 has weight 0 — it is the check digit).
_VIN_WEIGHTS: tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2)


def is_valid_vin_check_digit(candidate: str) -> bool:
    """Validate VIN check digit at position 9 per the NHTSA algorithm.

    Characters I, O, Q are excluded from valid VINs.  The check digit is
    ``'X'`` when the remainder is 10.
    """
    if len(candidate) != 17:
        return False
    try:
        total = sum(_VIN_VALUES[candidate[i]] * _VIN_WEIGHTS[i] for i in range(17))
        remainder = total % 11
        check = "X" if remainder == 10 else str(remainder)
        return candidate[8] == check
    except (KeyError, IndexError):
        return False


# ---------------------------------------------------------------------------
# Aadhaar Verhoeff Checksum
# ---------------------------------------------------------------------------

# Dihedral group D₅ multiplication table.
_VERHOEFF_D: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

# Permutation table.
_VERHOEFF_P: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)


def is_valid_aadhaar_verhoeff(digits: str) -> bool:
    """Validate a 12-digit Aadhaar number using the Verhoeff checksum.

    The Verhoeff algorithm uses dihedral group D₅ operations to detect
    all single-digit errors and all adjacent transposition errors.
    """
    if len(digits) != 12 or not digits.isdigit():
        return False
    try:
        c = 0
        reversed_digits = digits[::-1]
        for i, ch in enumerate(reversed_digits):
            c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
        return c == 0
    except (IndexError, ValueError):
        return False


# ---------------------------------------------------------------------------
# NPI (National Provider Identifier) — Enhancement 2F
# ---------------------------------------------------------------------------


def is_valid_npi(candidate: str) -> bool:
    """Validate a 10-digit NPI using Luhn with the "80840" prefix.

    Per CMS (45 CFR 162.408), prepend "80840" to the 10-digit NPI and
    validate the resulting 15-digit string with the Luhn algorithm.
    """
    if len(candidate) != 10 or not candidate.isdigit():
        return False
    return luhn_checksum("80840" + candidate)


# ---------------------------------------------------------------------------
# DEA Number checksum
# ---------------------------------------------------------------------------


def is_valid_dea_number(candidate: str) -> bool:
    """Validate a DEA registration number (2 letters + 7 digits).

    The check digit (last digit) equals ``(sum_odd + 2*sum_even) % 10``
    where odd/even refer to positions 1,3,5 and 2,4,6 of the 7-digit part.
    """
    if len(candidate) != 9:
        return False
    prefix = candidate[:2]
    digits_part = candidate[2:]
    if not prefix[0].isalpha() or not prefix[1].isalpha():
        return False
    if not digits_part.isdigit():
        return False
    d = [int(ch) for ch in digits_part]
    odd_sum = d[0] + d[2] + d[4]
    even_sum = d[1] + d[3] + d[5]
    check = (odd_sum + 2 * even_sum) % 10
    return check == d[6]


# ---------------------------------------------------------------------------
# Address phrase heuristic
# ---------------------------------------------------------------------------


def looks_like_address_phrase(
    text: str, start: int, end: int, address_suffixes: frozenset[str] = ADDRESS_SUFFIXES,
) -> bool:
    """Return *True* if the span ``text[start:end]`` looks like a street address.

    Checks for address suffix words (Street, Ave, Blvd, etc.) and numeric
    prefixes that suggest a physical address rather than a person name.
    """
    phrase = text[start:end]
    tokens = re.findall(r"[A-Za-z]+", phrase)
    if len(tokens) >= 2 and tokens[-1].lower() in address_suffixes:
        return True

    window_start = max(0, start - 20)
    prefix = text[window_start:start].lower()
    if "address" in prefix and start > 0 and text[start - 1].isdigit():
        return True

    if start > 0 and text[start - 1].isdigit():
        return True
    return False
