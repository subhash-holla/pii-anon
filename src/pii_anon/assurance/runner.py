"""The assurance runner — orchestrates a full report end-to-end.

config -> load+fingerprint dataset -> build adapters -> per-record detect/transform
-> paired substrate (+ silver reference) -> assessors -> Holm multiplicity ->
run-level gates (provenance + harness-close demotion) -> AssuranceReport ->
numeric gate (in to_dict) -> render -> PII-egress gate on each artifact -> write.
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import PipelineAdapter, Span, pii_anon_adapter
from .adjudication import RecordEval, Substrate, agreement_report, union_reference
from .assessors import (
    assess_compliance,
    assess_detection,
    assess_fairness,
    assess_leakage,
    assess_reidentification,
    assess_utility,
)
from .claim_strength import ClaimStrength, MeasurementMode, demote_to_advisory, not_assessable
from .config import AssuranceConfig
from .dataset import AssuranceDataset, compute_fingerprint, load_jsonl, load_records
from .pii_egress import assert_safe, scrub_label
from . import provenance as _provenance
from .provenance import (
    build_provenance,
    output_hash,
    provenance_gate,
    transform_hash,
)
from .report import AssuranceReport
from .render import render_html, render_json_bundle, render_markdown, render_summary
from .stats import holm_bonferroni
from .synthetic_mirror import is_synthetic_mirror, mirror_to_rows, synthesize_mirror

_CS_ORDER = {
    ClaimStrength.MEASURED: 2,
    ClaimStrength.ADVISORY: 1,
    ClaimStrength.NOT_ASSESSABLE: 0,
}


def _weaker(a: ClaimStrength, b: ClaimStrength) -> ClaimStrength:
    """The weaker of two claim strengths (a comparison is only as strong as its
    weaker system)."""
    return a if _CS_ORDER[a] <= _CS_ORDER[b] else b


def _load_dataset(config: AssuranceConfig) -> AssuranceDataset:
    if config.records is not None:
        return load_records(config.records, name=config.dataset_name or "user-dataset")
    assert config.dataset is not None  # guaranteed by config.validate()
    return load_jsonl(config.dataset, name=config.dataset_name)


def _sample(ds: AssuranceDataset, config: AssuranceConfig) -> AssuranceDataset:
    if not config.sample_size or len(ds.records) <= config.sample_size:
        return ds
    rng = random.Random(config.seed)
    idxs = sorted(rng.sample(range(len(ds.records)), config.sample_size))
    records = tuple(ds.records[i] for i in idxs)
    return AssuranceDataset(records, ds.mode, compute_fingerprint(records), ds.name)


def _build_substrate(
    ds: AssuranceDataset, user: PipelineAdapter, pii: PipelineAdapter, *, seed: int
) -> tuple[Substrate, list[tuple[Span, ...]], list[str | None], bool]:
    record_evals: list[RecordEval] = []
    user_preds: list[tuple[Span, ...]] = []
    user_transforms: list[str | None] = []
    for rec in ds.records:
        pred_by: dict[str, tuple[Span, ...]] = {}
        trans_by: dict[str, str | None] = {}
        err_by: dict[str, str | None] = {}
        for adapter in (user, pii):
            spans, _derr = adapter.safe_detect(rec.text)
            pred_by[adapter.name] = tuple(spans)
            out, terr = adapter.safe_transform(rec.text)
            trans_by[adapter.name] = out
            err_by[adapter.name] = terr
        user_preds.append(pred_by[user.name])
        user_transforms.append(trans_by[user.name])
        if ds.mode is MeasurementMode.LABELED:
            reference = tuple((s.entity_type, s.start, s.end) for s in (rec.labels or ()))
        else:
            reference = union_reference(pred_by)
        record_evals.append(RecordEval(
            record_id=rec.record_id, original_text=rec.text, language=rec.language,
            reference=reference, pred_by_system=pred_by, transform_by_system=trans_by,
            transform_error_by_system=err_by, group=rec.group,
        ))
    substrate = Substrate(
        tuple(record_evals), (user.name, pii.name), ds.mode,
        "gold" if ds.mode is MeasurementMode.LABELED else "silver",
    )
    # Determinism PROBE (best-effort; do NOT trust the declared flag). Re-run BOTH the
    # detector and transform over SEVERAL varied call orders (reversed + pinned shuffles)
    # and require every pass to match the forward baseline. Multiple varied-order passes
    # raise the power against stateful / period-aligned non-determinism. A low-rate
    # per-call flake can still slip an in-process probe — that residual is documented in
    # the reproducibility claim and is the job of the EXTERNAL verification mode (an
    # independent re-run that must match the recorded output_hash + transform_output_hash).
    n = len(ds.records)
    base = (output_hash(user_preds), transform_hash(user_transforms))

    def _pass(order: list[int]) -> tuple[str, str]:
        preds: list[tuple[Span, ...]] = [()] * n
        trans: list[str | None] = [None] * n
        for idx in order:
            text = ds.records[idx].text
            preds[idx] = tuple(user.safe_detect(text)[0])
            trans[idx] = user.safe_transform(text)[0]
        return output_hash(preds), transform_hash(trans)

    # Consecutive-repeat probe: any per-call non-determinism (a period-2/N toggle, a
    # flapping cache, an A/B alternation) DIFFERS between two back-to-back calls on the
    # SAME input, regardless of period — which the order-varied passes alone miss when each
    # record consumes an even number of calls and returns to the same toggle phase.
    repeat_ok = True
    for rec in ds.records:
        if (
            tuple(user.safe_detect(rec.text)[0]) != tuple(user.safe_detect(rec.text)[0])
            or user.safe_transform(rec.text)[0] != user.safe_transform(rec.text)[0]
        ):
            repeat_ok = False
            break

    orders = [list(reversed(range(n)))]
    for offset in (1, 2):
        shuffled = list(range(n))
        random.Random(seed + offset).shuffle(shuffled)
        orders.append(shuffled)
    deterministic_observed = repeat_ok and all(_pass(order) == base for order in orders)
    return substrate, user_preds, user_transforms, deterministic_observed


def _methodology(
    ds: AssuranceDataset, substrate: Substrate, demote_reasons: list[str],
    *, n_resamples: int, alpha: float, family_size: int, seed: int,
) -> list[str]:
    m = [
        f"Measurement mode: {ds.mode.value} (reference = {substrate.reference_kind}).",
        "Detection: strict span match (exact boundaries + casefolded entity type) vs the "
        "reference; per-record (cluster) bootstrap CIs over pooled counts; paired bootstrap "
        "vs the pii-anon baseline; Holm-Bonferroni across the comparison family.",
        f"Statistics provenance: bootstrap B={n_resamples} resamples at seed={seed}, "
        f"alpha={alpha}, Holm-Bonferroni over a family of {family_size} comparison(s) — so "
        "every CI / p-value is exactly recomputable from this bundle.",
        "Residual leakage: a fail-closed LOWER BOUND — count of referenced PII surface strings "
        "that survive verbatim in the transformed output (independent of detector recall). "
        "Degenerate/empty/failed transforms are NOT_ASSESSABLE, never 0.",
        "Claim strength: MEASURED (gold + powered), ADVISORY (silver / under-powered / "
        "un-certified harness), NOT_ASSESSABLE (input absent). Strengths are never blended.",
        "Self-grading disclosure: pii-anon is one of the systems under comparison. In silver "
        "mode (where pii-anon helps build the reference) no publishable head-to-head winner is "
        "reported — only agreement, disagreement, and a union/intersection sensitivity check.",
        "Every emitted number is routed through the no-fabrication validators; every emitted "
        "artifact is routed through a separate, fail-closed PII-egress gate (containment scan vs "
        "the in-memory raw corpus + a detector pass) before being written.",
    ]
    if demote_reasons:
        m.append("Run-level demotion applied (see limitations): " + "; ".join(sorted(set(demote_reasons))))
    return m


def _limitations(ds: AssuranceDataset) -> list[str]:
    lim = [
        "Residual leakage is a LOWER BOUND (leakage of referenced entities only); leakage of "
        "entities the reference itself missed is not counted.",
        "Re-identification resistance is a deterministic representative-adversary stand-in over a "
        "closed world of the dataset's own records (ADVISORY only, never a trained-shadow attack); "
        "real LiRA/MIA attacks are NOT_ASSESSABLE without a candidate pool, persona-target links, "
        "and shadow models. A low measured re-id rate is NOT a guarantee of anonymity.",
        "Utility (non-PII preservation), fairness (worst-group recall gap), and compliance "
        "(standard-coverage capability) are reported as separate axes with their own claim "
        "strengths; compliance coverage depends on the system's type-name vocabulary and is never "
        "rendered as a head-to-head winner.",
        "Headline numbers are recomputable only by a holder of the original dataset (AX-001); a "
        "synthetic-mirror corpus is shipped so the methodology is reproducible on non-real data.",
        # honesty for the reproducibility round-trip: when the input is ITSELF the shipped
        # mirror, no further mirror is regenerated (the input already IS the non-real corpus),
        # so the bundle carries an empty synthetic-mirror.jsonl by design, not by failure.
        *(["This run's input is itself a synthetic mirror, so no further mirror is regenerated "
           "(the input is already the non-real reproduction corpus); the bundle's "
           "synthetic-mirror.jsonl is intentionally empty for this run."]
          if is_synthetic_mirror(ds) else []),
        "User-supplied labels (pipeline name/version, dataset name) are best-effort scrubbed for "
        "PII (structured tokens + detector-found names); use non-identifying labels — a lowercase "
        "personal name in a label may not be detected (design §10.2).",
    ]
    if ds.mode is MeasurementMode.SILVER:
        lim.append("Silver mode: the reference is detector-derived (includes pii-anon), so results "
                   "are ADVISORY and carry a self-reference bias; no winner is declared.")
    return lim


class AssuranceRunError(RuntimeError):
    """A run failed; the original (possibly PII-bearing) error chain is suppressed."""


def run_assurance_report(config: AssuranceConfig) -> AssuranceReport:
    """Run a full assurance report; write artifacts to ``config.out_dir``; return the report.

    Wraps the PII-touching work so NO uncaught exception (and no locals-capturing
    traceback consumer — pytest --showlocals, Sentry, cgitb, ...) can serialize raw
    records: any failure — including a ``BaseException`` such as ``KeyboardInterrupt`` /
    ``SystemExit`` a user pipeline may raise — is re-raised as a PII-free
    :class:`AssuranceRunError` with the original chain suppressed (``from None``), so the
    inner frames (and their raw-text locals) never reach a traceback consumer. An
    interrupt therefore surfaces as a PII-free AssuranceRunError rather than the raw
    signal. Config-level errors are raised BEFORE any record is read.
    """
    config.validate()
    etype = "error"
    try:
        return _run_inner(config)
    except BaseException:  # noqa: BLE001 - even BaseException must be scrubbed (no PII via message/traceback)
        etype = type(sys.exc_info()[1]).__name__
    # Raise OUTSIDE the except block so the new error has NO __context__ at all — `from None`
    # only sets __suppress_context__ (renderers honor it, but a consumer that walks
    # __context__ regardless could still recover raw-text locals from the inner frames).
    raise AssuranceRunError(f"assurance run failed ({etype}); details suppressed for PII-safety")


def _run_inner(config: AssuranceConfig) -> AssuranceReport:
    assert config.pipeline is not None  # guaranteed by config.validate()
    ds_full = _load_dataset(config)
    ds = _sample(ds_full, config)
    sampled_size = (
        config.sample_size
        if (config.sample_size and len(ds_full.records) > config.sample_size)
        else None
    )
    full_fingerprint = ds_full.fingerprint
    pii = pii_anon_adapter(swarm=config.compare_swarm)
    # Scrub PII-shaped tokens from the user-supplied pipeline name/version (they are
    # emitted as labels), and disambiguate from the baseline name. The scrubbed name is
    # used as the system key everywhere, so the report never carries a raw label.
    src = config.pipeline
    safe_name = scrub_label(src.name, pii.detect)
    if safe_name == pii.name:
        safe_name = f"{safe_name}-user"
    user = PipelineAdapter(
        name=safe_name, detect=src.detect, transform=src.transform,
        version=scrub_label(src.version, pii.detect), deterministic=src.deterministic,
    )
    baseline = pii.name
    ds_name_safe = scrub_label(ds.name, pii.detect)

    substrate, user_preds, user_transforms, deterministic_observed = _build_substrate(
        ds, user, pii, seed=config.seed
    )
    thresholds = config.thresholds()
    seed = config.seed

    dimensions: dict[str, dict[str, Any]] = {}
    detection_comparisons: dict[str, dict[str, Any]] = {}
    comparison_family_size = 0
    alpha = 0.05

    if "detection" in config.dimensions:
        det, comps = assess_detection(
            substrate, baseline_system=baseline, thresholds=thresholds,
            seed=seed, n_resamples=config.n_resamples,
        )
        dimensions["detection"] = det
        # Capability gating (mirrors leakage's transform-absence handling): a system with
        # NO detector cannot be assessed on detection — emit NOT_ASSESSABLE, never a
        # phantom-0 F1. Then render NO head-to-head when either side is not assessable
        # (the locked phantom-0 / G2 rule: a one-sided dimension has no winner).
        name_to_adapter = {user.name: user, baseline: pii}
        for s, adapter in name_to_adapter.items():
            if adapter.detect is None and s in det:
                det[s] = not_assessable("detection", s, "adapter exposes no detect(); detection not assessable")
        pvals = {s: comps[s].p_value for s in comps}
        holm = holm_bonferroni(pvals)
        comparison_family_size = len(holm)
        for s, c in comps.items():
            either_na = (
                det.get(s) and det[s].claim_strength is ClaimStrength.NOT_ASSESSABLE
            ) or (
                det.get(baseline) and det[baseline].claim_strength is ClaimStrength.NOT_ASSESSABLE
            )
            if either_na:
                continue  # no win rendered when a system cannot enter the dimension
            detection_comparisons[s] = {
                "delta": c.delta, "ci": c.ci, "p_value": c.p_value,
                "p_value_holm": None, "significant_holm": holm.get(s, False),
                "effect_size": c.effect_size, "n": c.n,
            }

    if "leakage" in config.dimensions:
        dimensions["leakage"] = assess_leakage(
            substrate, thresholds=thresholds, seed=seed, n_resamples=config.n_resamples,
        )

    if "reidentification" in config.dimensions:
        dimensions["reidentification"] = assess_reidentification(
            substrate, thresholds=thresholds, seed=seed, n_resamples=config.n_resamples,
        )

    # The user-facing "utility_fairness_compliance" dimension is expanded into three honestly
    # separated axes, each with its own claim strength (a single blended value would hide which
    # axis is publishable vs hedged vs not-assessable).
    if "utility_fairness_compliance" in config.dimensions:
        dimensions["utility"] = assess_utility(
            substrate, thresholds=thresholds, seed=seed, n_resamples=config.n_resamples,
        )
        dimensions["fairness"] = assess_fairness(substrate, min_support=config.power_min_support)
        dimensions["compliance"] = assess_compliance(substrate)

    # provenance + run-level gates
    prov = build_provenance(
        dataset_name=ds_name_safe, dataset_fingerprint=ds.fingerprint, seed=seed,
        pii_anon_version=pii.version, user_pipeline_name=user.name,
        user_pipeline_version=user.version,
        # the declared flag is only a HINT — the observed determinism probe must also pass
        user_pipeline_deterministic=user.deterministic and deterministic_observed,
        # a detect-less pipeline's output_hash would be a non-binding constant (empty
        # preds), so emit it only when there's a detector; same for transform.
        user_pipeline_output_hash=output_hash(user_preds) if user.detect is not None else None,
        user_pipeline_transform_hash=(
            transform_hash(user_transforms) if user.transform is not None else None
        ),
        full_dataset_fingerprint=full_fingerprint,
        sample_size=sampled_size,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    prov_ok, prov_reasons = provenance_gate(prov)
    demote_reasons: list[str] = list(prov_reasons)
    # read the flag via the module (single source of truth) so the gate and the recorded
    # provenance value can never desync from an import-time snapshot.
    if not _provenance.HARNESS_CLOSE_CERTIFIED:
        demote_reasons.append(
            "assurance harness not yet certified by a 0-upheld adversarial close"
        )
    if demote_reasons:
        for by_system in dimensions.values():
            for s, res in list(by_system.items()):
                if res.claim_strength is ClaimStrength.MEASURED:
                    by_system[s] = demote_to_advisory(res, *demote_reasons)

    # Tag the head-to-head comparison with a claim strength derived from BOTH systems
    # (a comparison is only as strong as its weaker system) — AFTER demotion, so a
    # comparison can never be MEASURED while either dimension is ADVISORY. This closes
    # the "unqualified significant win escapes every gate" hole.
    det_dims = dimensions.get("detection", {})
    for s in detection_comparisons:
        cs_user = det_dims[s].claim_strength if s in det_dims else ClaimStrength.NOT_ASSESSABLE
        cs_base = (
            det_dims[baseline].claim_strength if baseline in det_dims else ClaimStrength.NOT_ASSESSABLE
        )
        detection_comparisons[s]["claim_strength"] = _weaker(cs_user, cs_base).value

    agreement = (
        agreement_report(substrate.records, user.name, baseline)
        if ds.mode is MeasurementMode.SILVER else None
    )

    report = AssuranceReport(
        dataset_name=ds_name_safe, dataset_fingerprint=ds.fingerprint, mode=ds.mode.value,
        reference_kind=substrate.reference_kind, n_records=len(ds.records),
        systems=substrate.systems, baseline=baseline, provenance=prov,
        dimensions=dimensions, detection_comparisons=detection_comparisons,
        agreement=agreement,
        methodology=_methodology(
            ds, substrate, demote_reasons,
            n_resamples=config.n_resamples, alpha=alpha,
            family_size=comparison_family_size, seed=seed,
        ),
        limitations=_limitations(ds), generated_at=prov.timestamp,
        n_resamples=config.n_resamples, alpha=alpha,
        comparison_family_size=comparison_family_size,
        power_thresholds={
            "min_support": thresholds.min_support,
            "max_ci_halfwidth": thresholds.max_ci_halfwidth,
            "min_clusters": thresholds.min_clusters,
            "require_significance": thresholds.require_significance,
        },
    )

    # serialize (numeric gate) + render + PII-egress gate + write
    report_dict = report.to_dict()
    files: dict[str, str] = {}
    if "json" in config.outputs:
        # Do NOT regenerate a mirror when the input is ITSELF a synthetic mirror (a
        # reproducibility round-trip): a mirror-of-a-mirror is redundant (the input already
        # IS the non-real reproduction corpus) and is the sole source of a containment FALSE
        # positive — the regenerated reserved-namespace tokens (e.g. user…@example.com) can
        # coincide with the input mirror's own synthetic tokens. Skipping GENERATION (not the
        # gate) is fail-safe: every emitted artifact is still fully egress-scanned below, so a
        # misclassification can only drop an artifact, never weaken the gate on real data.
        input_is_mirror = is_synthetic_mirror(ds)
        mirror_rows = [] if input_is_mirror else mirror_to_rows(synthesize_mirror(ds, seed=seed))
        files.update(render_json_bundle(report_dict, mirror_rows=mirror_rows))
    if "markdown" in config.outputs:
        files.update(render_markdown(report_dict))
    if "html" in config.outputs:
        files.update(render_html(report_dict))
    if "summary" in config.outputs:
        files.update(render_summary(report_dict))

    raw_corpus = substrate.raw_corpus()
    out_dir = Path(config.out_dir)
    for fname, content in files.items():
        # Every artifact (incl. the synthetic mirror) gets the full gate: the redesigned
        # containment is FP-free (structured-token + detector-surface), so the mirror's
        # generated tokens no longer false-positive and need no exemption.
        assert_safe(
            content, raw_corpus, artifact=fname, min_k=config.egress_min_k, detector=pii.detect,
        )
        path = out_dir / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        # errors="replace": a lone surrogate (corrupted/scraped data) in any string must not
        # crash the write at the UTF-8 boundary — surrogate-safe like the fingerprint encode.
        path.write_text(content, encoding="utf-8", errors="replace")

    return report
