"""Per-dimension assessors (Port 2).

Each assessor declares the inputs it needs and self-marks ``NOT_ASSESSABLE`` (with
a reason) when they are absent — never a phantom 0.
"""

from __future__ import annotations

from .detection import assess_detection
from .leakage import assess_leakage
from .reidentification import assess_reidentification
from .utility_fairness import assess_compliance, assess_fairness, assess_utility

__all__ = [
    "assess_detection",
    "assess_leakage",
    "assess_reidentification",
    "assess_utility",
    "assess_fairness",
    "assess_compliance",
]
