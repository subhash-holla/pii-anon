"""Bayesian Bradley-Terry rating engine — THIRD / CLAIM-GRADE tier (S3-03).

Fits a Bayesian Bradley-Terry model by **NUTS (NumPyro)** and produces a
**joint posterior** over per-system strengths, behind the structural
:class:`~pii_anon.eval_framework.rating.port.RatingEnginePort`. This is the
**only claim-grade tier** of the 3-tier ladder
(``glicko-legacy`` → ``bradley-terry-mle`` → ``bayes-bt``): every fit is run
through the hard NFR-001 convergence gate
(:mod:`pii_anon.eval_framework.rating.convergence`), which **fails loud** when
split-R̂ > 1.01 ∨ bulk-ESS < 400/param ∨ divergences > 0. Resolves SME
CATASTROPHIC eval-01 (frequentist tiers satisfy NFR-001 only by-substitution).

Environment fork (S3-03 §2) — import-safe WITHOUT jax
-----------------------------------------------------
numpyro / jax / arviz are heavy, platform-specific wheels and are **not** part
of the base install (they live in the ``bayes-eval`` extra). This module is
therefore **import-safe with numpyro absent**: numpyro/jax are imported
**lazily, inside the sampling method**, never at module top level. So:

* the engine stays *discoverable* — the registry lists ``bayes-bt`` even without
  the extra installed (only *sampling* needs jax);
* calling a sampling method (:meth:`run_round_robin` /
  :meth:`fit_paired_posterior`) without the extra raises a clear, loud
  :class:`MissingOptionalDependencyError` naming the ``bayes-eval`` extra —
  **never** a silent non-Bayesian fallback (which would re-introduce eval-01).

Model (Bayesian Bradley-Terry, claim-grade)
-------------------------------------------
Latent per-system strengths ``θ_i``::

    σ      ~ HalfNormal(1.0)                         # hierarchical scale
    θ_raw  ~ Normal(0, σ)                            # per system
    θ      = θ_raw − mean(θ_raw)                     # sum-to-zero (identifiable)
    wins_i ~ Binomial(n_ij, logits = θ_i − θ_j)     # per comparison record

Sampler: ``NUTS(model, target_accept_prob=0.9)``;
``MCMC(num_warmup, num_samples, num_chains>=4, chain_method='sequential')``
(sequential avoids jax pmap device issues in CI); seeded ``random.PRNGKey``.

Posterior → port types (Elo scale, shared with ``elo.py`` / ``bradley_terry.py``)::

    rating_i = 1500 + 400 · mean(θ_i) / ln(10)
    rd_i     = 400 · sd(θ_i)   / ln(10)             # a genuine posterior SD

# DAVIDSON(S3-04): ties are out of scope here (plain BT). The Davidson tie term
# and the ONE-joint-posterior significance coherence (P(rank=1 | joint posterior)
# for the SDO CompetitiveSupremacyGate J-meter) land in S3-04, which consumes the
# exact :class:`Posterior` this engine returns. See the seam in ``_bt_model``.

Evidence basis:
    - Bradley & Terry (1952) "Rank Analysis of Incomplete Block Designs".
    - Hoffman & Gelman (2014) "The No-U-Turn Sampler" (NUTS).
    - Phan, Pradhan & Jankowiak (2019) "Composable Effects for Flexible and
      Accelerated Probabilistic Programming in NumPyro".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .convergence import ConvergenceReport, assert_claim_grade, count_divergences
from .elo import EloRating, RatingUpdate

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np
    from numpy.typing import NDArray

__all__ = [
    "BayesBTEngine",
    "MissingOptionalDependencyError",
    "PairedCountsWithTies",
    "Posterior",
]

# Elo-scale reporting constants — kept in lock-step with ``elo.py`` /
# ``bradley_terry.py`` so deltas read like Elo points across all three tiers.
_ELO_ANCHOR = 1500.0
_ELO_SCALE = 400.0
_LN10 = math.log(10.0)

# The optional extra that carries the claim-grade sampler stack.
_BAYES_EVAL_EXTRA = "bayes-eval"

# Comparison-shape alias, identical to S3-02's ``fit_paired`` shape:
# ``{(sys_i, sys_j): (wins_i, wins_j, n)}``.
PairedCounts = dict[tuple[str, str], tuple[float, float, int]]

# DAVIDSON(S3-04): the PARALLEL 4-tuple shape carrying tie counts:
# ``{(sys_i, sys_j): (wins_i, wins_j, ties, n)}`` with ``wins_i+wins_j+ties=n``.
# Kept separate from ``PairedCounts`` (which ``fit_paired_posterior`` consumes
# with integer ``int(round(...))`` totals) so the tie path never perturbs the
# plain-BT counts. Produced by ``rating.paired_set.assemble_paired_set``.
PairedCountsWithTies = dict[tuple[str, str], tuple[float, float, float, int]]


class MissingOptionalDependencyError(ImportError):
    """Raised when a claim-grade sampling call needs the absent ``bayes-eval`` extra.

    Subclasses :class:`ImportError` so the codebase's existing
    ``except ImportError`` optional-dependency handlers keep working and it reads
    idiomatically as a missing-import condition. The message names the
    ``bayes-eval`` extra and the offending package so the fix is unambiguous.
    This is the NFR-026 *fail-loud* contract: the claim-grade tier NEVER silently
    degrades to a non-Bayesian answer (that would re-introduce eval-01).
    """

    @classmethod
    def for_package(cls, package: str) -> MissingOptionalDependencyError:
        """Build the standard message for a missing ``package`` from ``bayes-eval``."""
        return cls(
            f"{package!r} is required for the claim-grade 'bayes-bt' rating tier "
            f"but is not installed. Install the optional extra: "
            f"pip install 'pii-anon[{_BAYES_EVAL_EXTRA}]'  "
            f"(provides numpyro, jax, arviz). The 'bayes-bt' tier never falls "
            f"back to a non-Bayesian answer — claim-grade requires the real NUTS "
            f"posterior + the NFR-001 convergence gate."
        )


@dataclass(frozen=True)
class Posterior:
    """Joint posterior over per-system strengths from a claim-grade NUTS fit.

    This is the object S3-04 (Davidson ties + by-construction significance) and
    the SDO ``CompetitiveSupremacyGate`` J-meter
    (``J = P(rank(pii-anon)=1 | joint posterior)``) consume — it carries the
    *joint* draws (not just marginal summaries) so rank probabilities are
    computed coherently from one sample.

    Attributes
    ----------
    systems:
        System names in column order of ``theta_samples`` (sorted).
    theta_samples:
        ``(n_draws_total, n_systems)`` array of sum-to-zero log-strength draws
        (chains concatenated). Column ``k`` corresponds to ``systems[k]``.
    convergence:
        The NFR-001 :class:`ConvergenceReport` for this fit (already gate-checked
        by :meth:`BayesBTEngine.fit_paired_posterior`).
    ratings:
        Posterior-summary :class:`EloRating` per system (mean θ → Elo rating,
        sd θ → Elo-scale rd).
    nu_samples:
        DAVIDSON(S3-04): the joint draws of the Davidson tie parameter ``ν``,
        or ``None`` for a plain Bradley-Terry (no-ties) fit. When present the fit
        modeled ties; ``ν > 0`` is the identifiable tie propensity. Carried on the
        SAME posterior as ``theta_samples`` so significance and tie inference are
        read off ONE joint sample.
    """

    systems: list[str]
    theta_samples: "NDArray[np.float64]"
    convergence: ConvergenceReport
    ratings: dict[str, EloRating]
    nu_samples: "NDArray[np.float64] | None" = None


class BayesBTEngine:
    """Bayesian Bradley-Terry engine (NumPyro NUTS) — the claim-grade tier.

    Satisfies :class:`~pii_anon.eval_framework.rating.port.RatingEnginePort`
    structurally (``run_round_robin`` + ``get_rating``); the claim-grade
    :meth:`fit_paired_posterior` is engine-only (NOT on the port, like S3-02's
    ``fit_paired``) because it returns the rich :class:`Posterior` the downstream
    significance / J-meter consume.

    The module is import-safe without numpyro; the heavy stack is imported lazily
    inside :meth:`_run_nuts`. Construction, :meth:`get_rating`, and the pure
    posterior→Elo summary work with numpyro absent.

    Parameters
    ----------
    gamma:
        Logistic steepness for the composite→outcome map in the port path
        (default 10, matching ``elo`` / ``bradley_terry``).
    num_warmup:
        NUTS warmup (burn-in) iterations per chain (default 1000).
    num_samples:
        Retained NUTS draws per chain (default 1000).
    num_chains:
        Number of chains (default 4 — the NFR-001 split-R̂ needs ≥ 2; ≥ 4 is the
        claim-grade convention).
    target_accept_prob:
        NUTS target acceptance probability (default 0.9).
    seed:
        PRNG seed → ``jax.random.PRNGKey(seed)`` (AX-002 determinism).
    n_synthetic_per_pair:
        Binomial trial count synthesized per pair in the port path (default 100).
    """

    def __init__(
        self,
        *,
        gamma: float = 10.0,
        num_warmup: int = 1000,
        num_samples: int = 1000,
        num_chains: int = 4,
        target_accept_prob: float = 0.9,
        seed: int = 0,
        n_synthetic_per_pair: int = 100,
    ) -> None:
        if num_chains < 2:
            # split-R̂ is undefined with < 2 chains; the claim-grade gate needs it.
            raise ValueError("num_chains must be >= 2 for the NFR-001 split-R̂ gate")
        self._gamma = gamma
        self._num_warmup = num_warmup
        self._num_samples = num_samples
        self._num_chains = num_chains
        self._target_accept_prob = target_accept_prob
        self._seed = seed
        self._n_synthetic_per_pair = n_synthetic_per_pair

        self._ratings: dict[str, EloRating] = {}
        self._history: list[RatingUpdate] = []
        self._last_posterior: Posterior | None = None

    # -- Read-only configuration --------------------------------------------

    @property
    def num_chains(self) -> int:
        """Number of NUTS chains (>= 2, required by the split-R̂ gate)."""
        return self._num_chains

    @property
    def last_posterior(self) -> Posterior | None:
        """The most recent claim-grade :class:`Posterior`, or ``None``."""
        return self._last_posterior

    # -- Pure helpers (no numpyro) ------------------------------------------

    def _sigmoid(self, x: float) -> float:
        """Logistic map of a composite gap to a [0, 1] soft outcome.

        Mirrors :meth:`elo.PIIRateEloEngine._sigmoid` and
        :meth:`bradley_terry.BradleyTerryMLEEngine._sigmoid` (``σ(γ·x)``) so the
        bayes port path is a faithful drop-in for the other tiers' outcome model.
        """
        clamped = max(-20.0, min(20.0, self._gamma * x))
        return 1.0 / (1.0 + math.exp(-clamped))

    def _composites_to_counts(self, composites: dict[str, float]) -> PairedCounts:
        """Map point composites → a synthetic paired Binomial design.

        Reuses S3-02's sigmoid soft-outcome mapping so the port-compat path is
        consistent across tiers: for each pair ``s_ij = sigmoid(C_i − C_j)`` is
        the expected win fraction, realized as integer Binomial counts over
        ``n_synthetic_per_pair`` trials (rounded), so the claim-grade Binomial
        likelihood applies unchanged. Sorted iteration → determinism.
        """
        systems = sorted(composites)
        n = self._n_synthetic_per_pair
        counts: PairedCounts = {}
        for i in range(len(systems)):
            for j in range(i + 1, len(systems)):
                name_i, name_j = systems[i], systems[j]
                s_ij = self._sigmoid(composites[name_i] - composites[name_j])
                wins_i = float(round(s_ij * n))
                counts[(name_i, name_j)] = (wins_i, float(n) - wins_i, n)
        return counts

    @staticmethod
    def _normalize_comparisons(
        comparisons: PairedCounts,
    ) -> tuple[
        list[str],
        "NDArray[np.int_]",
        "NDArray[np.int_]",
        "NDArray[np.int_]",
        "NDArray[np.int_]",
    ]:
        """Flatten paired counts into sorted-name index arrays for the model.

        Returns ``(names, idx_i, idx_j, wins_i, totals)`` where ``names`` is the
        sorted system list and the arrays are parallel over comparison records
        (``idx_i``/``idx_j`` are column indices into ``names``; ``wins_i`` and
        ``totals`` are the Binomial successes and trial counts). numpy-only.
        """
        import numpy as np

        names: list[str] = sorted(
            {name for pair in comparisons for name in pair}
        )
        index = {name: k for k, name in enumerate(names)}
        idx_i: list[int] = []
        idx_j: list[int] = []
        wins: list[int] = []
        totals: list[int] = []
        for (sys_i, sys_j), (wins_i, _wins_j, n) in comparisons.items():
            idx_i.append(index[sys_i])
            idx_j.append(index[sys_j])
            wins.append(int(round(wins_i)))
            totals.append(int(n))
        return (
            names,
            np.asarray(idx_i, dtype=np.int_),
            np.asarray(idx_j, dtype=np.int_),
            np.asarray(wins, dtype=np.int_),
            np.asarray(totals, dtype=np.int_),
        )

    @staticmethod
    def _normalize_comparisons_with_ties(
        comparisons: "PairedCountsWithTies",
    ) -> tuple[
        list[str],
        "NDArray[np.int_]",
        "NDArray[np.int_]",
        "NDArray[np.int_]",
    ]:
        """Flatten 4-tuple tie counts into sorted-name index + outcome arrays.

        DAVIDSON(S3-04). Returns ``(names, idx_i, idx_j, outcomes)`` where
        ``outcomes`` is an ``(n_records_total, 3)`` integer array of
        ``(wins_i, ties, wins_j)`` per comparison record (the Davidson Multinomial
        category order is i-wins / tie / j-wins, matching logits
        ``[δ/2, ln ν, −δ/2]``). Each ``(wins_i, wins_j, ties, n)`` entry becomes
        ONE Multinomial record over ``n`` trials. numpy-only; integer counts (no
        half-splitting). Raises if a tally is inconsistent
        (``wins_i + wins_j + ties != n``).
        """
        import numpy as np

        names: list[str] = sorted({name for pair in comparisons for name in pair})
        index = {name: k for k, name in enumerate(names)}
        idx_i: list[int] = []
        idx_j: list[int] = []
        outcomes: list[tuple[int, int, int]] = []
        for (sys_i, sys_j), (wins_i, wins_j, ties, n) in comparisons.items():
            wi, wj, t = int(round(wins_i)), int(round(wins_j)), int(round(ties))
            if wi + wj + t != int(n):
                raise ValueError(
                    f"inconsistent tie tally for ({sys_i}, {sys_j}): "
                    f"wins_i={wi} + wins_j={wj} + ties={t} != n={int(n)}"
                )
            idx_i.append(index[sys_i])
            idx_j.append(index[sys_j])
            outcomes.append((wi, t, wj))  # category order: i-wins, tie, j-wins
        return (
            names,
            np.asarray(idx_i, dtype=np.int_),
            np.asarray(idx_j, dtype=np.int_),
            np.asarray(outcomes, dtype=np.int_),
        )

    def _posterior_to_ratings(
        self, names: list[str], theta_samples: "NDArray[np.float64]"
    ) -> dict[str, EloRating]:
        """Summarize joint θ draws → per-system :class:`EloRating` (pure-numpy).

        ``rating = 1500 + 400·mean(θ)/ln10``; ``rd = 400·sd(θ)/ln10`` — the
        posterior SD is a *genuine* uncertainty (unlike the legacy
        match-count-only RD). Pure function of the samples → deterministic
        (AX-002). ``theta_samples`` is ``(n_draws, n_systems)``.
        """
        import numpy as np

        ratings: dict[str, EloRating] = {}
        means = np.mean(theta_samples, axis=0)
        sds = np.std(theta_samples, axis=0, ddof=1) if theta_samples.shape[0] > 1 \
            else np.zeros(theta_samples.shape[1])
        for k, name in enumerate(names):
            rating = _ELO_ANCHOR + _ELO_SCALE * float(means[k]) / _LN10
            rd = _ELO_SCALE * float(sds[k]) / _LN10
            ratings[name] = EloRating(
                system_name=name,
                rating=rating,
                rd=rd,
                num_matches=self._ratings.get(name, EloRating(name)).num_matches,
            )
        return ratings

    # -- Claim-grade NUTS (lazy numpyro import lives HERE) ------------------

    def _bt_model(
        self,
        idx_i: "NDArray[np.int_]",
        idx_j: "NDArray[np.int_]",
        n_systems: int,
        wins_i: "NDArray[np.int_] | None" = None,
        totals: "NDArray[np.int_] | None" = None,
        outcomes: "NDArray[np.int_] | None" = None,
    ) -> None:
        """NumPyro Bayesian Bradley-Terry model (called only under the NUTS runs).

        ``σ ~ HalfNormal(1)``; ``θ_raw ~ Normal(0, σ)``; ``θ = θ_raw − mean`` for
        sum-to-zero identifiability. The likelihood branch depends on the data:

        * **Plain BT (default):** ``wins_i``/``totals`` given →
          ``wins_i ~ Binomial(n_ij, logits = θ_i − θ_j)`` per record.
        * **DAVIDSON(S3-04) tie branch:** ``outcomes`` (an ``(R, 3)`` array of
          ``(wins_i, ties, wins_j)`` per record) given → a tie parameter
          ``ν ~ HalfNormal(1)`` and a three-way Multinomial with logits
          ``[δ/2, ln ν, −δ/2]`` (δ = θ_i − θ_j), i.e. Davidson (1970):
          ``P(i>j) ∝ exp(θ_i)``, ``P(tie) ∝ ν·exp((θ_i+θ_j)/2)``,
          ``P(j>i) ∝ exp(θ_j)``. The tie term shares the SAME ``θ`` as the win
          term, so significance stays coherent off ONE joint posterior. ``ν = 0``
          would nest plain BT (the closed form in ``significance.py`` proves this
          equivalence; the in-tree unit tests pin the math without jax).
        """
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist

        sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))  # pragma: no cover
        with numpyro.plate("systems", n_systems):  # pragma: no cover
            theta_raw = numpyro.sample("theta_raw", dist.Normal(0.0, sigma))
        # Sum-to-zero anchor → identifiable strengths on the Elo scale.
        theta = numpyro.deterministic(  # pragma: no cover
            "theta", theta_raw - jnp.mean(theta_raw)
        )

        delta = theta[idx_i] - theta[idx_j]  # pragma: no cover

        if outcomes is not None:  # pragma: no cover
            # DAVIDSON(S3-04) tie branch. Multinomial logits [δ/2, ln ν, −δ/2]
            # over (i-wins, tie, j-wins). The common exp((θ_i+θ_j)/2) factor
            # cancels in the softmax, so these logits reproduce the Davidson
            # closed form exactly (see significance.davidson_*).
            nu = numpyro.sample("nu", dist.HalfNormal(1.0))
            log_nu = jnp.log(nu)
            tie_logits = jnp.broadcast_to(log_nu, delta.shape)
            logits3 = jnp.stack([delta / 2.0, tie_logits, -delta / 2.0], axis=-1)
            total_per_record = outcomes.sum(axis=-1)
            numpyro.sample(
                "outcomes",
                dist.Multinomial(total_count=total_per_record, logits=logits3),
                obs=outcomes,
            )
        else:
            # Plain Bradley-Terry (no ties): the legacy default path.
            numpyro.sample(  # pragma: no cover
                "wins",
                dist.BinomialLogits(logits=delta, total_count=totals),
                obs=wins_i,
            )

    def _run_nuts(
        self, comparisons: PairedCounts
    ) -> tuple[list[str], "NDArray[np.float64]", int]:
        """Run NUTS on ``comparisons``; return ``(names, theta_samples, n_div)``.

        This is the ONLY place numpyro/jax are imported. With the ``bayes-eval``
        extra absent the import fails and is re-raised as a loud
        :class:`MissingOptionalDependencyError` — never a silent fallback.
        ``theta_samples`` is ``(num_chains·num_samples, n_systems)``.
        """
        try:
            import jax
            import numpy as np
            import numpyro
            from numpyro.infer import MCMC, NUTS
        except ImportError as exc:  # bayes-eval extra absent → fail LOUD.
            package = getattr(exc, "name", None) or "numpyro"
            raise MissingOptionalDependencyError.for_package(package) from exc

        if not comparisons:
            raise ValueError("comparisons must be non-empty to fit a posterior")

        names, idx_i, idx_j, wins_i, totals = self._normalize_comparisons(comparisons)
        n_systems = len(names)

        # Sequential chains avoid jax pmap device issues in CI; host device count
        # is set defensively so multi-chain sampling is reproducible.
        numpyro.set_host_device_count(self._num_chains)
        kernel = NUTS(self._bt_model, target_accept_prob=self._target_accept_prob)
        mcmc = MCMC(
            kernel,
            num_warmup=self._num_warmup,
            num_samples=self._num_samples,
            num_chains=self._num_chains,
            chain_method="sequential",
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(self._seed)
        # ``extra_fields=("diverging",)`` is REQUIRED for the NFR-001 0-divergence
        # gate to have teeth: without it numpyro does not retain the per-sample
        # divergence flags and ``get_extra_fields()`` would always report none,
        # silently blinding the gate. Collect them explicitly.
        mcmc.run(
            rng_key,
            idx_i=idx_i,
            idx_j=idx_j,
            n_systems=n_systems,
            wins_i=wins_i,
            totals=totals,
            extra_fields=("diverging",),
        )

        # theta with chain dim retained → (num_chains, num_samples, n_systems)
        # for the convergence gate; also a flat (total, n_systems) for summaries.
        samples = mcmc.get_samples(group_by_chain=True)
        theta_by_chain = np.asarray(samples["theta"], dtype=np.float64)
        n_div = self._count_divergences(mcmc)
        return names, theta_by_chain, n_div

    def _run_nuts_with_ties(
        self, comparisons: "PairedCountsWithTies"
    ) -> tuple[list[str], "NDArray[np.float64]", "NDArray[np.float64]", int]:
        """Run the DAVIDSON(S3-04) tie-aware NUTS; return
        ``(names, theta_by_chain, nu_by_chain, n_div)``.

        Mirrors :meth:`_run_nuts` exactly (lazy numpyro import HERE; loud
        :class:`MissingOptionalDependencyError` when the ``bayes-eval`` extra is
        absent; sequential chains; explicit ``extra_fields=("diverging",)`` so the
        NFR-001 0-divergence arm keeps its teeth) but threads the
        ``(R, 3)`` Davidson outcome array and additionally retains the ``ν`` draws.
        ``theta_by_chain`` is ``(num_chains, num_samples, n_systems)``;
        ``nu_by_chain`` is ``(num_chains, num_samples)``.
        """
        try:
            import jax
            import numpy as np
            import numpyro
            from numpyro.infer import MCMC, NUTS
        except ImportError as exc:  # bayes-eval extra absent → fail LOUD.
            package = getattr(exc, "name", None) or "numpyro"
            raise MissingOptionalDependencyError.for_package(package) from exc

        if not comparisons:
            raise ValueError("comparisons must be non-empty to fit a posterior")

        names, idx_i, idx_j, outcomes = self._normalize_comparisons_with_ties(
            comparisons
        )
        n_systems = len(names)

        numpyro.set_host_device_count(self._num_chains)
        kernel = NUTS(self._bt_model, target_accept_prob=self._target_accept_prob)
        mcmc = MCMC(
            kernel,
            num_warmup=self._num_warmup,
            num_samples=self._num_samples,
            num_chains=self._num_chains,
            chain_method="sequential",
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(self._seed)
        mcmc.run(
            rng_key,
            idx_i=idx_i,
            idx_j=idx_j,
            n_systems=n_systems,
            outcomes=outcomes,
            extra_fields=("diverging",),
        )

        samples = mcmc.get_samples(group_by_chain=True)
        theta_by_chain = np.asarray(samples["theta"], dtype=np.float64)
        nu_by_chain = np.asarray(samples["nu"], dtype=np.float64)
        n_div = self._count_divergences(mcmc)
        return names, theta_by_chain, nu_by_chain, n_div

    @staticmethod
    def _count_divergences(mcmc: Any) -> int:
        """Total NUTS divergences from the sampler's ``diverging`` extra field.

        Delegates the actual counting to the gate's canonical
        :func:`~pii_anon.eval_framework.rating.convergence.count_divergences`
        (single source of truth), so the ``diverging`` flags are summed exactly
        the way the NFR-001 gate expects.

        **Fails loud** (does NOT swallow to 0): ``_run_nuts`` always requests
        ``extra_fields=("diverging",)``, so the field must be present. If it is
        absent (numpyro API drift, or a run that forgot the request) the
        0-divergence arm of the NFR-001 gate would be silently blinded — a
        fabricated "no divergences" could pass a non-claim-grade fit. We raise
        instead, so a real failure of the divergence arm is visible (the
        fail-loud contract that resolves eval-01). A genuine exception from
        ``get_extra_fields()`` is allowed to propagate for the same reason.
        """
        extra = mcmc.get_extra_fields()
        if not isinstance(extra, dict) or "diverging" not in extra:
            raise RuntimeError(
                "NUTS extra fields missing 'diverging'; cannot verify the "
                "NFR-001 0-divergence gate. Ensure the sampler runs with "
                "extra_fields=('diverging',). Refusing to report a fabricated "
                "0-divergence count (fail-loud, eval-01 contract)."
            )
        return count_divergences(extra["diverging"])

    # -- Public API: structural port (FR-003) -------------------------------

    def run_round_robin(self, composites: dict[str, float]) -> list[RatingUpdate]:
        """Run an all-pairs round-robin from ``name -> composite score`` via NUTS.

        Point composites are mapped to a synthetic Binomial paired design (S3-02
        sigmoid soft-outcome mapping), NUTS samples the joint posterior, the
        NFR-001 convergence gate is enforced, and one :class:`RatingUpdate` per
        system (old = prior rating, new = posterior-mean rating) is emitted for
        audit parity with the other tiers.

        **Requires the ``bayes-eval`` extra.** With numpyro absent this raises
        :class:`MissingOptionalDependencyError` (loud) — never a non-Bayesian
        fallback.
        """
        systems = sorted(composites)
        old_ratings = {
            name: (
                self._ratings[name].rating
                if name in self._ratings
                else _ELO_ANCHOR
            )
            for name in systems
        }
        counts = self._composites_to_counts(composites)
        posterior = self.fit_paired_posterior(counts)

        updates: list[RatingUpdate] = []
        for name in systems:
            new_rating = posterior.ratings[name].rating
            updates.append(
                RatingUpdate(
                    system_name=name,
                    old_rating=old_ratings[name],
                    new_rating=new_rating,
                    change=new_rating - old_ratings[name],
                    expected_score=0.5,
                    actual_score=0.5,
                    k_factor=0.0,
                )
            )
        self._history.extend(updates)
        return updates

    def get_rating(self, name: str) -> EloRating | None:
        """Return the current (posterior-summary) rating for ``name``, or ``None``.

        Reads stored state only — safe with numpyro absent (no fallback, no
        import).
        """
        return self._ratings.get(name)

    # -- Public API: claim-grade paired posterior (engine-only) -------------

    def fit_paired_posterior(self, comparisons: PairedCounts) -> Posterior:
        """Fit the claim-grade joint posterior from record-level paired *counts*.

        ``comparisons`` maps ``(sys_i, sys_j)`` → ``(wins_i, wins_j, n)`` (the
        real shape; real designs are not perfectly separable so no soft mapping
        is needed). Runs NUTS, builds the joint :class:`Posterior`, and ENFORCES
        the NFR-001 convergence gate via :func:`assert_claim_grade` — a
        non-converged fit raises :class:`ConvergenceError` (fails loud); the
        leaderboard never emits a non-claim-grade Bayesian rating.

        This is NOT part of the structural port (engine-only, like S3-02's
        ``fit_paired``): it returns the joint :class:`Posterior` that S3-04's
        by-construction significance and the SDO J-meter consume.

        **Requires the ``bayes-eval`` extra** → loud
        :class:`MissingOptionalDependencyError` when absent.
        """
        names, theta_by_chain, n_div = self._run_nuts(comparisons)

        # NFR-001 gate over the chain-structured draws (split-R̂ / bulk-ESS /
        # divergences). Build the report, store ratings, THEN assert claim-grade
        # so a failed gate still leaves an inspectable report on the exception.
        report = ConvergenceReport.from_samples(theta_by_chain, n_divergences=n_div)

        # Flatten chains → (total_draws, n_systems) for marginal summaries.
        n_chains, n_draws, n_systems = theta_by_chain.shape
        theta_flat = theta_by_chain.reshape(n_chains * n_draws, n_systems)

        # Track per-system match counts for audit parity, then summarize.
        for (sys_i, sys_j), (_wi, _wj, n) in comparisons.items():
            for name in (sys_i, sys_j):
                prior = self._ratings.get(name, EloRating(name))
                self._ratings[name] = EloRating(
                    system_name=name,
                    rating=prior.rating,
                    rd=prior.rd,
                    num_matches=prior.num_matches + n,
                )
        ratings = self._posterior_to_ratings(names, theta_flat)
        self._ratings.update(ratings)

        posterior = Posterior(
            systems=names,
            theta_samples=theta_flat,
            convergence=report,
            ratings=ratings,
        )
        self._last_posterior = posterior

        # Fail loud on a non-claim-grade fit (the NFR-001 teeth).
        assert_claim_grade(report)
        return posterior

    def fit_paired_posterior_with_ties(
        self, comparisons: "PairedCountsWithTies", *, _store_state: bool = True
    ) -> Posterior:
        """DAVIDSON(S3-04): fit the claim-grade joint posterior WITH ties.

        Sibling of :meth:`fit_paired_posterior` for the tie-aware path.
        ``comparisons`` maps ``(sys_i, sys_j)`` → ``(wins_i, wins_j, ties, n)``
        (the 4-tuple ``rating.paired_set.assemble_paired_set`` produces). Runs the
        Davidson NUTS model (a ``ν ~ HalfNormal`` tie parameter + a three-way
        Multinomial sharing the SAME ``θ``), ENFORCES the NFR-001 convergence gate
        over the θ draws, and returns a :class:`Posterior` whose ``nu_samples``
        carries the joint ``ν`` draws — so by-construction significance and the
        tie inference are read off ONE joint posterior.

        ``_store_state`` (internal) controls whether this fit mutates the engine's
        stored ratings / ``last_posterior``. The Tier-3 RRS path
        (:meth:`fit_rrs_posterior`) passes ``False`` so its DISTINCT posterior
        never bleeds into the de-id engine state (FR-010/AX-004).

        **Requires the ``bayes-eval`` extra** → loud
        :class:`MissingOptionalDependencyError` when absent.
        """
        names, theta_by_chain, nu_by_chain, n_div = self._run_nuts_with_ties(
            comparisons
        )

        # NFR-001 gate over the θ chains (the ν draws are an auxiliary parameter;
        # the claim-grade ordering/significance live in θ). Report first so a
        # failed gate still leaves an inspectable report on the exception.
        report = ConvergenceReport.from_samples(theta_by_chain, n_divergences=n_div)

        n_chains, n_draws, n_systems = theta_by_chain.shape
        theta_flat = theta_by_chain.reshape(n_chains * n_draws, n_systems)
        nu_flat = nu_by_chain.reshape(n_chains * n_draws)

        ratings = self._posterior_to_ratings(names, theta_flat)

        if _store_state:
            # Track per-system match counts for audit parity (full n incl. ties),
            # then summarize — mirrors the plain-BT path.
            for (sys_i, sys_j), (_wi, _wj, _t, n) in comparisons.items():
                for name in (sys_i, sys_j):
                    prior = self._ratings.get(name, EloRating(name))
                    self._ratings[name] = EloRating(
                        system_name=name,
                        rating=prior.rating,
                        rd=prior.rd,
                        num_matches=prior.num_matches + n,
                    )
            self._ratings.update(ratings)

        posterior = Posterior(
            systems=names,
            theta_samples=theta_flat,
            convergence=report,
            ratings=ratings,
            nu_samples=nu_flat,
        )
        if _store_state:
            self._last_posterior = posterior

        # Fail loud on a non-claim-grade fit (the NFR-001 teeth).
        assert_claim_grade(report)
        return posterior

    def fit_rrs_posterior(self, comparisons: "PairedCountsWithTies") -> Posterior:
        """FR-010/AX-004: fit the Tier-3 re-identification-resistance (RRS) posterior.

        A SEPARATE Davidson sub-model for the Tier-3 RRS metric, kept STRICTLY
        DISTINCT from the detection / de-identification significance: it is its
        own method, returns its OWN :class:`Posterior` object, and — critically —
        passes ``_store_state=False`` so it shares NO mutable state with the de-id
        engine (it never touches ``self._ratings`` or ``self._last_posterior``).
        The anon≠pseudo / RRS-never-merged-into-de-id axiom (AX-004) is therefore
        enforced structurally: there is no code path that folds an RRS fit into a
        de-id rating.

        ``comparisons`` is the same 4-tuple tie shape (RRS paired outcomes derive
        from re-identification attempts, not detection F1). Returns the RRS joint
        :class:`Posterior` (with ``nu_samples``) for the caller to consume in
        isolation.

        **Requires the ``bayes-eval`` extra** → loud
        :class:`MissingOptionalDependencyError` when absent.
        """
        return self.fit_paired_posterior_with_ties(comparisons, _store_state=False)
