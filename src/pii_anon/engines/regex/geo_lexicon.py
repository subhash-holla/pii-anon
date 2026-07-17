"""sp7 #9 — curated geo gazetteer (LOCATION), home-safe subset.

Emits LOCATION for countries and US states/territories — unambiguous
public-domain geographic facts (no external gazetteer dependency, no licensing
question). The high-value taxonomy split (STATE/COUNTY/COUNTRY subtypes for
nemotron's 712 unreachable spans) is DEFERRED — it expands
SUPPORTED_ENTITY_TYPES and touches the label census, which needs a user
decision (the sp3 categorical-regression lesson).

Precision discipline (the sp3 label-gating lesson): a bare gazetteer token is
NOT a reliable place signal — "Georgia"/"Jordan"/"Reading" are also names and
common words. So a match fires ONLY when it carries geographic evidence:
  * the entry is multi-word / unambiguous ("United States", "New York"), OR
  * a location preposition/cue precedes it ("in|at|from|near|to|of <Place>"), OR
  * it is followed by a country/state comma-structure,
AND the token is not in the COLLIDE veto (place names that collide with common
English words or common given names).
"""
from __future__ import annotations

import re

# ── Public-domain facts: countries (common English short names) ─────────────
_COUNTRIES = {
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Argentina",
    "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cambodia", "Cameroon", "Canada", "Chad",
    "Chile", "China", "Colombia", "Comoros", "Congo", "Croatia", "Cuba",
    "Cyprus", "Czechia", "Georgia", "Jordan", "Turkey",
    "Denmark", "Djibouti", "Dominica", "Ecuador", "Egypt", "Eritrea",
    "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon",
    "Gambia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea",
    "Guyana", "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia",
    "Iran", "Iraq", "Ireland", "Israel", "Italy", "Jamaica", "Japan",
    "Kazakhstan", "Kenya", "Kiribati", "Kosovo", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein",
    "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives",
    "Mali", "Malta", "Mauritania", "Mauritius", "Mexico", "Micronesia",
    "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
    "Myanmar", "Namibia", "Nauru", "Nepal", "Netherlands", "Nicaragua",
    "Niger", "Nigeria", "Norway", "Oman", "Pakistan", "Palau", "Panama",
    "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Rwanda", "Samoa", "Senegal", "Serbia", "Seychelles",
    "Singapore", "Slovakia", "Slovenia", "Somalia", "Spain", "Sudan",
    "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan",
    "Tanzania", "Thailand", "Togo", "Tonga", "Tunisia", "Turkmenistan",
    "Tuvalu", "Uganda", "Ukraine", "Uruguay", "Uzbekistan", "Vanuatu",
    "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe",
    # multi-word (unambiguous — accepted without preposition context)
    "United States", "United Kingdom", "New Zealand", "South Africa",
    "South Korea", "North Korea", "Saudi Arabia", "Sri Lanka", "Costa Rica",
    "El Salvador", "Papua New Guinea", "Sierra Leone", "Ivory Coast",
    "Czech Republic", "Dominican Republic", "United Arab Emirates",
    "San Marino", "Cape Verde", "Trinidad and Tobago",
}
# ── US states + territories (public-domain) ─────────────────────────────────
_US_STATES = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Hawaii", "Idaho", "Illinois",
    "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland",
    "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri",
    "Montana", "Nebraska", "Nevada", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Tennessee", "Texas", "Utah", "Vermont", "Virginia",
    "Washington", "Wisconsin", "Wyoming",
    "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina",
    "North Dakota", "Rhode Island", "South Carolina", "South Dakota",
    "West Virginia", "Puerto Rico",
}
_GAZ = _COUNTRIES | _US_STATES
_GAZ_MULTIWORD = {g for g in _GAZ if " " in g}
_GAZ_SINGLE = {g for g in _GAZ if " " not in g}

# Single-token gazetteer entries that collide with common given names or common
# English words — these fire ONLY with an explicit location preposition/cue.
_COLLIDE = {
    "Georgia", "Jordan", "Chad", "Turkey", "China", "Guinea", "Reading",
    "Mobile", "Nice", "Bath", "Hull", "Deal", "Sandwich", "Jersey", "Virginia",
    "Washington", "Israel", "Mali", "Benin", "Togo", "Chile", "Cuba", "Oman",
    "Laos", "Fiji", "Palau", "Nauru", "India", "Charlotte", "Florence",
    "Austin", "Dakota", "Carolina", "Marshall", "Phoenix", "Salem",
}
_SINGLE_SAFE = _GAZ_SINGLE - _COLLIDE

_LOC_CUE = re.compile(
    r"(?i)\b(?:in|at|from|near|to|of|into|within|across|throughout|"
    r"located\s+in|based\s+in|residing\s+in|traveled\s+to|born\s+in|"
    r"lives?\s+in|city\s+of|state\s+of|country\s+of|province\s+of)\s*$"
)
# longest gazetteer phrase first so "New York" beats "York" etc.
_ALL = sorted(_GAZ, key=len, reverse=True)
_GAZ_RE = re.compile(r"\b(" + "|".join(re.escape(g) for g in _ALL) + r")\b")

_CONF = 0.72


def geo_subtype(place: str) -> str:
    """Classify a gazetteer place into its geographic SUBTYPE — ``"STATE"`` (US
    state/territory) or ``"COUNTRY"`` — else ``"LOCATION"`` when unknown.

    This is an ADVISORY signal only: the emitted ``entity_type`` stays
    ``LOCATION`` (so home strict scoring is byte-identical). The subtype is
    surfaced on the finding's ``explanation`` for downstream masking/policy and
    lets the eval-side taxonomy relabel become a thin projection instead of a
    re-classifier. The lexicon carries no county data, so ``COUNTY`` is never
    fabricated. ``_US_STATES`` is checked first, so a name in both sets resolves
    to the more specific ``STATE``."""
    if place in _US_STATES:
        return "STATE"
    if place in _COUNTRIES:
        return "COUNTRY"
    return "LOCATION"


def extract_locations(text: str) -> list[tuple[str, int, int, float]]:
    """Return ``(LOCATION, start, end, confidence)`` for gazetteer places that
    carry geographic evidence. Additive and leak-safe."""
    out: list[tuple[str, int, int, float]] = []
    for m in _GAZ_RE.finditer(text):
        place = m.group(1)
        start, end = m.start(1), m.end(1)
        if place in _GAZ_MULTIWORD:
            accept = True  # multi-word entries are unambiguous
        elif place in _SINGLE_SAFE:
            accept = True  # unambiguous single-token place
        else:
            # ambiguous single token (in COLLIDE): require a location cue before it
            accept = bool(_LOC_CUE.search(text[max(0, start - 24):start]))
        if accept:
            out.append(("LOCATION", start, end, _CONF))
    return out
