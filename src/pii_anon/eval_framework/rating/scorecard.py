"""System scorecards for PII de-identification benchmark reporting.

A scorecard combines Tier 1 metrics (detection + efficiency), optional
Tier 2 metrics (privacy, utility, fairness), the composite score, and
Elo ratings into a single structured record per system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SystemScorecard:
    """Complete benchmark scorecard for one system.

    Combines all metric tiers, composite score, and Elo rating
    into a single auditable record.

    Attributes
    ----------
    system_name:
        Human-readable system identifier (e.g., "pii-anon", "presidio").
    available:
        Whether the system was available for benchmarking.

    Tier 1 — Detection quality:
        f1, precision, recall

    Tier 1 — Operational efficiency:
        latency_p50_ms, docs_per_hour

    Tier 2 — Privacy, utility, fairness:
        privacy_score, utility_score, fairness_score

    Composite:
        composite_score — single [0,1] value combining all tiers.

    Elo:
        elo_rating, elo_rd — pairwise rating and uncertainty.
    """

    system_name: str
    available: bool = True

    # Tier 1 — Detection
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    # Tier 1 — Efficiency
    latency_p50_ms: float = 0.0
    docs_per_hour: float = 0.0

    # Tier 2 — Privacy / Utility / Fairness
    privacy_score: float = 0.0
    utility_score: float = 0.0
    fairness_score: float = 0.0

    # Composite
    composite_score: float = 0.0

    # Elo
    elo_rating: float = 1500.0
    elo_rd: float = 350.0

    # Metadata
    samples: int = 0
    evaluation_track: str = "detect_only"
    license_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "system_name": self.system_name,
            "available": self.available,
            "f1": round(self.f1, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "latency_p50_ms": round(self.latency_p50_ms, 3),
            "docs_per_hour": round(self.docs_per_hour, 2),
            "privacy_score": round(self.privacy_score, 6),
            "utility_score": round(self.utility_score, 6),
            "fairness_score": round(self.fairness_score, 6),
            "composite_score": round(self.composite_score, 6),
            "elo_rating": round(self.elo_rating, 2),
            "elo_rd": round(self.elo_rd, 2),
            "samples": self.samples,
            "evaluation_track": self.evaluation_track,
            "license_name": self.license_name,
        }


@dataclass
class BenchmarkScorecard:
    """Collection of system scorecards from a single benchmark run.

    Attributes
    ----------
    benchmark_name:
        Identifier for the benchmark (e.g., "pii_anon_benchmark_v1_v1").
    dataset_name:
        Dataset used for evaluation.
    system_scorecards:
        Mapping of system name → scorecard.
    """

    benchmark_name: str
    dataset_name: str = ""
    system_scorecards: dict[str, SystemScorecard] = field(default_factory=dict)

    def add_system(self, scorecard: SystemScorecard) -> None:
        """Add or replace a system scorecard."""
        self.system_scorecards[scorecard.system_name] = scorecard

    def get_system(self, name: str) -> SystemScorecard | None:
        """Look up a system scorecard by name."""
        return self.system_scorecards.get(name)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "benchmark_name": self.benchmark_name,
            "dataset_name": self.dataset_name,
            "systems": {
                name: sc.to_dict()
                for name, sc in sorted(self.system_scorecards.items())
            },
        }
