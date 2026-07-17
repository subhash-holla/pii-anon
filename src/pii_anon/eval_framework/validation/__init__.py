"""Public, contract-tested no-fabrication validators.

Historically these guards lived as private (underscore) functions inside
``eval_framework/evaluation/competitive_supremacy.py`` and were reused only by
that one gate. The PII Handling Assurance Report needs the *same* numeric-
integrity guarantees on every value it emits, so this module gives them a
PUBLIC, separately-tested home.

A contract test (``tests/assurance/test_no_fabrication_contract.py``) pins this
module's behaviour to the original private functions so the two copies can never
drift. The eventual DRY-refactor — having ``competitive_supremacy.py`` import
from here — is deliberately deferred (it would itself trigger that gate's
mandatory adversarial close); until then the contract test is the anti-drift
guarantee.

CARDINAL: well-formedness validation (what this module does) is NECESSARY but
NOT SUFFICIENT for an honest number. It proves a value is finite / in-range /
non-blank; it cannot prove the value is *scientifically* honest, nor that a
string is free of PII. Those are the claim-strength engine's and the PII-egress
gate's jobs respectively.
"""

from __future__ import annotations

from .no_fabrication import (
    detected_entity_names,
    finite_unit_score,
    is_finite_number,
    is_nonblank_str,
    safe_names,
    safe_repr,
)

__all__ = [
    "detected_entity_names",
    "finite_unit_score",
    "is_finite_number",
    "is_nonblank_str",
    "safe_names",
    "safe_repr",
]
