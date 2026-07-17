"""Run configuration for the assurance report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .adapters import PipelineAdapter
from .claim_strength import PowerThresholds

ALL_DIMENSIONS = ("detection", "leakage", "reidentification", "utility_fairness_compliance")
PHASE1_DIMENSIONS = ("detection", "leakage")
ALL_OUTPUTS = ("json", "markdown", "html", "summary")


@dataclass
class AssuranceConfig:
    """Everything one assurance run needs.

    Provide a dataset as a path (``dataset``) or in-memory rows (``records``).
    Provide the user pipeline via ``pipeline`` (a :class:`PipelineAdapter`).
    """

    dataset: str | None = None
    records: Sequence[Mapping[str, object]] | None = None
    dataset_name: str | None = None
    pipeline: PipelineAdapter | None = None
    dimensions: tuple[str, ...] = PHASE1_DIMENSIONS
    sample_size: int | None = None
    seed: int = 20260617
    outputs: tuple[str, ...] = ("json", "markdown")
    out_dir: str = "./assurance-out"
    # power gate
    power_min_support: int = 50
    power_max_ci_halfwidth: float = 0.05
    power_min_clusters: int = 20
    require_significance: bool = False
    # safety / stats
    egress_min_k: int = 8
    n_resamples: int = 500
    # which pii-anon offering to compare against
    compare_swarm: bool = False
    baseline_name: str = "pii-anon"

    def __repr__(self) -> str:  # defensive: config holds the RAW records — never echo them
        n = len(self.records) if self.records is not None else None
        src = "records" if self.records is not None else ("file" if self.dataset else "none")
        return (
            f"AssuranceConfig(source={src}, n_records={n}, "
            f"pipeline={self.pipeline.name if self.pipeline else None!r}, "
            f"dimensions={self.dimensions}, outputs={self.outputs}, seed={self.seed})"
        )

    def thresholds(self) -> PowerThresholds:
        return PowerThresholds(
            min_support=self.power_min_support,
            max_ci_halfwidth=self.power_max_ci_halfwidth,
            require_significance=self.require_significance,
            min_clusters=self.power_min_clusters,
        )

    def validate(self) -> None:
        if self.dataset is None and self.records is None:
            raise ValueError("AssuranceConfig needs a dataset path or in-memory records")
        if self.pipeline is None:
            raise ValueError("AssuranceConfig needs a pipeline adapter")
        unknown = set(self.dimensions) - set(ALL_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown dimensions: {sorted(unknown)}")
        bad_out = set(self.outputs) - set(ALL_OUTPUTS)
        if bad_out:
            raise ValueError(f"unknown outputs: {sorted(bad_out)}")
        if self.n_resamples < 1:
            raise ValueError("n_resamples must be >= 1")
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("sample_size must be >= 1 when set")
        # a sane seed bound: a huge int would crash str(seed) (CPython int->str digit limit)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or abs(self.seed) >= 2**63:
            raise ValueError("seed must be an int within +/- 2**63")
        # Power thresholds are TIGHTEN-ONLY: a run may make the publishability bar STRICTER
        # than the committed defaults but never weaker, so a MEASURED label cannot be forged
        # by relaxing the gate to a no-op. (PowerThresholds() carries the committed floors.)
        _floor = PowerThresholds()
        if self.power_min_support < _floor.min_support:
            raise ValueError(f"power_min_support must be >= {_floor.min_support} (committed floor)")
        if not (0.0 < self.power_max_ci_halfwidth <= _floor.max_ci_halfwidth):
            raise ValueError(
                f"power_max_ci_halfwidth must be in (0, {_floor.max_ci_halfwidth}] (committed floor)"
            )
        if self.power_min_clusters < _floor.min_clusters:
            raise ValueError(f"power_min_clusters must be >= {_floor.min_clusters} (committed floor)")
