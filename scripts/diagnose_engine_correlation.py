#!/usr/bin/env python3
"""Diagnose engine correlation and redundancy in the pii-anon-swarm.

Loads benchmark records, runs each engine individually, and computes:
  1. Pairwise Jaccard similarity on TP sets (agreement on what they find)
  2. Pairwise error correlation (how often two engines BOTH miss an entity)
  3. Unique-contribution analysis (entities only one engine finds)
  4. Redundancy ranking

Usage:
    python scripts/diagnose_engine_correlation.py [--records N] [--output PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

# Suppress noisy library output during model loading.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*truncate to max_length.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

# Ensure project root is on path.
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from pii_anon.benchmarks import BenchmarkRecord, load_benchmark_dataset
from pii_anon.engines import (
    GLiNERAdapter,
    PresidioAdapter,
    RegexEngineAdapter,
    ScrubadubAdapter,
    SpacyNERAdapter,
    StanzaNERAdapter,
)
from pii_anon.engines.base import EngineAdapter


# ---------------------------------------------------------------------------
# Entity-type normalization (mirrored from competitor_compare.py)
# ---------------------------------------------------------------------------
_NORM_MAP: dict[str, str] = {
    "EMAIL": "EMAIL_ADDRESS",
    "EMAILFILTH": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "PHONENUMBERFILTH": "PHONE_NUMBER",
    "SSN": "US_SSN",
    "US_SSN": "US_SSN",
    "SOCIALSECURITYNUMBERFILTH": "US_SSN",
    "PERSON": "PERSON_NAME",
    "PERSON_NAME": "PERSON_NAME",
    "PERSONFILTH": "PERSON_NAME",
    "PER": "PERSON_NAME",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "CREDITCARD": "CREDIT_CARD",
    "CREDITCARDFILTH": "CREDIT_CARD",
    "CREDIT_CARD": "CREDIT_CARD",
    "IP": "IP_ADDRESS",
    "IPV4": "IP_ADDRESS",
    "IPADDRESS": "IP_ADDRESS",
    "IP_ADDRESS": "IP_ADDRESS",
    "DATE_OF_BIRTH": "DATE_OF_BIRTH",
    "DOB": "DATE_OF_BIRTH",
    "MAC_ADDRESS": "MAC_ADDRESS",
    "DRIVERS_LICENSE": "DRIVERS_LICENSE",
    "PASSPORT": "PASSPORT",
    "ROUTING_NUMBER": "ROUTING_NUMBER",
    "LICENSE_PLATE": "LICENSE_PLATE",
    "BANK_ACCOUNT": "BANK_ACCOUNT",
    "NATIONAL_ID": "NATIONAL_ID",
    "USERNAME": "USERNAME",
    "EMPLOYEE_ID": "EMPLOYEE_ID",
    "MEDICAL_RECORD_NUMBER": "MEDICAL_RECORD_NUMBER",
    "MRN": "MEDICAL_RECORD_NUMBER",
    "ORGANIZATION": "ORGANIZATION",
    "ORG": "ORGANIZATION",
    "ADDRESS": "ADDRESS",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "GPE": "LOCATION",
    "NAME": "PERSON_NAME",
    "IBAN": "IBAN",
    "CRYPTO_WALLET": "CRYPTO_WALLET",
    "CREDIT_CARD_FRAGMENT": "CREDIT_CARD_FRAGMENT",
}

# Entity types that are NER-model phantom labels, not real PII.
_IGNORE_TYPES = {
    "CARDINAL", "FAC", "DATE", "MONEY", "PRODUCT",
    "WORK_OF_ART", "ORDINAL", "QUANTITY", "LAW", "TIME",
    "NORP", "EVENT", "LANGUAGE", "PERCENT",
}


def normalize_entity_type(raw: str) -> str | None:
    """Normalize an entity type string. Returns None if it should be ignored."""
    upper = raw.upper().strip()
    if upper in _IGNORE_TYPES:
        return None
    return _NORM_MAP.get(upper, upper)


# ---------------------------------------------------------------------------
# Gold entity key: (record_id, entity_type, start, end)
# ---------------------------------------------------------------------------
GoldKey = tuple[str, str, int, int]


def gold_entities_for_record(record: BenchmarkRecord) -> list[GoldKey]:
    """Extract gold entity keys from a benchmark record."""
    keys = []
    for label in record.labels:
        etype = normalize_entity_type(label["entity_type"])
        if etype is None:
            continue
        keys.append((record.record_id, etype, label["start"], label["end"]))
    return keys


def engine_detections_match_gold(
    findings: list[Any],
    record: BenchmarkRecord,
    gold_keys: list[GoldKey],
) -> set[GoldKey]:
    """Determine which gold entities an engine's findings match.

    Uses span-overlap matching: a finding matches a gold entity if they share
    the same normalized entity type AND their character spans overlap by at
    least 50% of the gold span length (IoU-lite).
    """
    matched: set[GoldKey] = set()
    for finding in findings:
        start = getattr(finding, "span_start", None)
        end = getattr(finding, "span_end", None)
        if start is None or end is None:
            continue
        raw_type = str(getattr(finding, "entity_type", "UNKNOWN"))
        etype = normalize_entity_type(raw_type)
        if etype is None:
            continue

        for gk in gold_keys:
            _, g_type, g_start, g_end = gk
            if etype != g_type:
                continue
            # Compute overlap.
            overlap_start = max(int(start), g_start)
            overlap_end = min(int(end), g_end)
            overlap = max(0, overlap_end - overlap_start)
            gold_len = g_end - g_start
            if gold_len > 0 and overlap >= gold_len * 0.5:
                matched.add(gk)
    return matched


# ---------------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------------
ENGINE_SPECS: list[tuple[str, type[EngineAdapter]]] = [
    ("regex", RegexEngineAdapter),
    ("presidio", PresidioAdapter),
    ("gliner", GLiNERAdapter),
    ("spacy", SpacyNERAdapter),
    ("stanza", StanzaNERAdapter),
    ("scrubadub", ScrubadubAdapter),
]


def init_engines() -> list[tuple[str, EngineAdapter]]:
    """Instantiate and check availability of each engine."""
    engines: list[tuple[str, EngineAdapter]] = []
    for name, cls in ENGINE_SPECS:
        adapter = cls(enabled=True)
        if not adapter.dependency_available():
            print(f"  [SKIP] {name}: native dependency not available")
            continue
        try:
            adapter.initialize({})
        except Exception as exc:
            print(f"  [SKIP] {name}: initialization failed: {exc}")
            continue
        engines.append((name, adapter))
        print(f"  [OK]   {name} ({adapter.adapter_id})")
    return engines


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity: |A & B| / |A | B|."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def error_correlation(miss_a: set, miss_b: set, total_gold: int) -> float:
    """Fraction of gold entities that BOTH engines miss."""
    if total_gold == 0:
        return 0.0
    return len(miss_a & miss_b) / total_gold


def conditional_co_miss(miss_a: set, miss_b: set) -> float:
    """P(B misses | A misses): how likely B also misses, given A missed."""
    if not miss_a:
        return 0.0
    return len(miss_a & miss_b) / len(miss_a)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run_analysis(
    records: list[BenchmarkRecord],
    engines: list[tuple[str, EngineAdapter]],
) -> dict[str, Any]:
    """Run all engines on all records and compute correlation metrics."""

    engine_names = [name for name, _ in engines]
    n_engines = len(engine_names)

    # Per-engine: set of gold keys detected (TP) and missed (FN).
    engine_tp: dict[str, set[GoldKey]] = defaultdict(set)
    engine_fn: dict[str, set[GoldKey]] = defaultdict(set)
    all_gold: set[GoldKey] = set()

    # Per-entity-type tracking.
    entity_type_engines: dict[str, dict[str, set[GoldKey]]] = defaultdict(
        lambda: defaultdict(set)
    )

    context = {"policy_mode": "balanced", "language": "en"}

    for i, record in enumerate(records):
        gold_keys = gold_entities_for_record(record)
        all_gold.update(gold_keys)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing record {i + 1}/{len(records)}...")

        for name, adapter in engines:
            try:
                findings = adapter.detect({"text": record.text}, context)
            except Exception:
                findings = []

            matched = engine_detections_match_gold(findings, record, gold_keys)
            engine_tp[name].update(matched)
            missed = set(gold_keys) - matched
            engine_fn[name].update(missed)

            # Per-entity-type tracking.
            for gk in matched:
                entity_type_engines[gk[1]][name].add(gk)

    total_gold = len(all_gold)
    print(f"\n  Total gold entities across {len(records)} records: {total_gold}")

    # --- Per-engine recall ---
    engine_recall: dict[str, float] = {}
    for name in engine_names:
        tp_count = len(engine_tp[name])
        engine_recall[name] = tp_count / total_gold if total_gold else 0.0

    # --- Pairwise Jaccard on TP sets ---
    jaccard_matrix: dict[tuple[str, str], float] = {}
    for a, b in combinations(engine_names, 2):
        j = jaccard_similarity(engine_tp[a], engine_tp[b])
        jaccard_matrix[(a, b)] = j

    # --- Pairwise error correlation ---
    error_corr_matrix: dict[tuple[str, str], float] = {}
    cond_comiss_matrix: dict[tuple[str, str], float] = {}
    for a, b in combinations(engine_names, 2):
        ec = error_correlation(engine_fn[a], engine_fn[b], total_gold)
        error_corr_matrix[(a, b)] = ec
        # Conditional: P(B misses | A misses) and P(A misses | B misses).
        cond_comiss_matrix[(a, b)] = conditional_co_miss(engine_fn[a], engine_fn[b])
        cond_comiss_matrix[(b, a)] = conditional_co_miss(engine_fn[b], engine_fn[a])

    # --- Unique contributions ---
    unique_contrib: dict[str, set[GoldKey]] = {}
    for name in engine_names:
        others_tp = set()
        for other_name in engine_names:
            if other_name != name:
                others_tp |= engine_tp[other_name]
        unique_contrib[name] = engine_tp[name] - others_tp

    # --- Per-entity-type recall by engine ---
    entity_types_in_gold = sorted({gk[1] for gk in all_gold})
    per_entity_recall: dict[str, dict[str, float]] = defaultdict(dict)
    for etype in entity_types_in_gold:
        etype_gold = {gk for gk in all_gold if gk[1] == etype}
        etype_count = len(etype_gold)
        for name in engine_names:
            tp_for_etype = engine_tp[name] & etype_gold
            per_entity_recall[etype][name] = (
                len(tp_for_etype) / etype_count if etype_count else 0.0
            )

    # --- Marginal value: what does adding engine X to the union add? ---
    union_all = set()
    for name in engine_names:
        union_all |= engine_tp[name]
    marginal_value: dict[str, float] = {}
    for name in engine_names:
        union_without = set()
        for other in engine_names:
            if other != name:
                union_without |= engine_tp[other]
        lost = union_all - union_without
        marginal_value[name] = len(lost) / total_gold if total_gold else 0.0

    return {
        "engine_names": engine_names,
        "total_gold": total_gold,
        "engine_recall": engine_recall,
        "jaccard_matrix": jaccard_matrix,
        "error_corr_matrix": error_corr_matrix,
        "cond_comiss_matrix": cond_comiss_matrix,
        "unique_contrib": {k: len(v) for k, v in unique_contrib.items()},
        "unique_contrib_pct": {
            k: len(v) / total_gold * 100 if total_gold else 0.0
            for k, v in unique_contrib.items()
        },
        "per_entity_recall": per_entity_recall,
        "marginal_value": marginal_value,
        "union_recall": len(union_all) / total_gold if total_gold else 0.0,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_report(results: dict[str, Any], n_records: int) -> str:
    """Format analysis results as a markdown report."""
    lines: list[str] = []
    w = lines.append

    engine_names = results["engine_names"]
    total_gold = results["total_gold"]

    w("# Engine Correlation and Redundancy Analysis")
    w("")
    w("## Summary")
    w("")
    w(f"- **Records analyzed**: {n_records}")
    w(f"- **Total gold entities**: {total_gold}")
    w(f"- **Engines tested**: {', '.join(engine_names)}")
    w(f"- **Union recall** (all engines combined): {results['union_recall']:.3f}")
    w("")

    # --- Individual engine recall ---
    w("## Individual Engine Recall")
    w("")
    w("| Engine | Recall | TP Count |")
    w("|--------|--------|----------|")
    for name in sorted(engine_names, key=lambda n: -results["engine_recall"][n]):
        recall = results["engine_recall"][name]
        tp = int(recall * total_gold)
        w(f"| {name} | {recall:.3f} | {tp}/{total_gold} |")
    w("")

    # --- Jaccard similarity matrix ---
    w("## Pairwise Jaccard Similarity (TP Sets)")
    w("")
    w("Higher = engines agree more on what they detect. Values near 1.0 indicate")
    w("redundancy (the engines find the same things).")
    w("")
    header = "| | " + " | ".join(engine_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(engine_names)) + "|"
    w(header)
    w(sep)
    for a in engine_names:
        row = f"| **{a}** |"
        for b in engine_names:
            if a == b:
                row += " 1.000 |"
            else:
                key = (a, b) if (a, b) in results["jaccard_matrix"] else (b, a)
                val = results["jaccard_matrix"].get(key, 0.0)
                row += f" {val:.3f} |"
        w(row)
    w("")

    # --- Error correlation matrix ---
    w("## Pairwise Error Correlation (Co-Miss Rate)")
    w("")
    w("Fraction of ALL gold entities that BOTH engines miss. Higher = more correlated errors.")
    w("If engines A and B have high co-miss rate, adding B provides little safety net over A.")
    w("")
    header = "| | " + " | ".join(engine_names) + " |"
    w(header)
    w(sep)
    for a in engine_names:
        row = f"| **{a}** |"
        for b in engine_names:
            if a == b:
                miss_rate = len(
                    [1 for gk in range(total_gold)]
                ) - int(results["engine_recall"][a] * total_gold)
                fn_rate = 1.0 - results["engine_recall"][a]
                row += f" {fn_rate:.3f} |"
            else:
                key = (a, b) if (a, b) in results["error_corr_matrix"] else (b, a)
                val = results["error_corr_matrix"].get(key, 0.0)
                row += f" {val:.3f} |"
        w(row)
    w("")

    # --- Conditional co-miss ---
    w("## Conditional Co-Miss: P(B misses | A misses)")
    w("")
    w("Given that engine A missed an entity, how likely is engine B to also miss it?")
    w("High values mean B provides no safety net when A fails.")
    w("")
    header = "| A \\ B | " + " | ".join(engine_names) + " |"
    w(header)
    w(sep)
    for a in engine_names:
        row = f"| **{a}** |"
        for b in engine_names:
            if a == b:
                row += " - |"
            else:
                val = results["cond_comiss_matrix"].get((a, b), 0.0)
                row += f" {val:.3f} |"
        w(row)
    w("")

    # --- Unique contributions ---
    w("## Unique Contributions (Entities Only This Engine Finds)")
    w("")
    w("| Engine | Unique Entities | % of Gold | Marginal Value |")
    w("|--------|----------------|-----------|----------------|")
    for name in sorted(engine_names, key=lambda n: -results["unique_contrib"][n]):
        uc = results["unique_contrib"][name]
        pct = results["unique_contrib_pct"][name]
        mv = results["marginal_value"][name]
        w(f"| {name} | {uc} | {pct:.1f}% | {mv:.3f} |")
    w("")
    w("**Marginal value**: fraction of gold entities lost if this engine is removed")
    w("from the ensemble. An engine with marginal value 0.000 is fully redundant.")
    w("")

    # --- Per-entity-type recall heatmap ---
    w("## Per-Entity-Type Recall by Engine")
    w("")
    per_entity = results["per_entity_recall"]
    entity_types = sorted(per_entity.keys())
    header = "| Entity Type | " + " | ".join(engine_names) + " |"
    sep2 = "|---|" + "|".join(["---"] * len(engine_names)) + "|"
    w(header)
    w(sep2)
    for etype in entity_types:
        row = f"| {etype} |"
        for name in engine_names:
            val = per_entity[etype].get(name, 0.0)
            row += f" {val:.2f} |"
        w(row)
    w("")

    # --- Most/least redundant pairs ---
    w("## Most Redundant Engine Pairs")
    w("")
    w("Sorted by Jaccard similarity (highest = most redundant):")
    w("")
    pairs = sorted(
        results["jaccard_matrix"].items(), key=lambda x: -x[1]
    )
    w("| Pair | Jaccard | Co-Miss Rate | Conditional Co-Miss (A|B) | Conditional Co-Miss (B|A) |")
    w("|------|---------|-------------|---------------------------|---------------------------|")
    for (a, b), jac in pairs:
        ec_key = (a, b) if (a, b) in results["error_corr_matrix"] else (b, a)
        ec = results["error_corr_matrix"].get(ec_key, 0.0)
        cm_ab = results["cond_comiss_matrix"].get((a, b), 0.0)
        cm_ba = results["cond_comiss_matrix"].get((b, a), 0.0)
        w(f"| {a} + {b} | {jac:.3f} | {ec:.3f} | {cm_ab:.3f} | {cm_ba:.3f} |")
    w("")

    # --- Key findings ---
    w("## Key Findings and Recommendations")
    w("")

    # Find the most redundant pair.
    if pairs:
        (most_a, most_b), most_jac = pairs[0]
        w(f"1. **Most redundant pair**: {most_a} + {most_b} (Jaccard = {most_jac:.3f})")
        ec_key = (most_a, most_b) if (most_a, most_b) in results["error_corr_matrix"] else (most_b, most_a)
        ec_val = results["error_corr_matrix"].get(ec_key, 0.0)
        w(f"   - Co-miss rate: {ec_val:.3f} -- they miss the same entities {ec_val*100:.1f}% of the time")

    # Find least redundant pair.
    if pairs:
        (least_a, least_b), least_jac = pairs[-1]
        w(f"2. **Least redundant pair**: {least_a} + {least_b} (Jaccard = {least_jac:.3f})")
        w(f"   - These engines have the most complementary detection patterns")

    # Engines with zero marginal value.
    zero_mv = [n for n in engine_names if results["marginal_value"][n] < 0.001]
    if zero_mv:
        w(f"3. **Fully redundant engines** (marginal value ~0): {', '.join(zero_mv)}")
        w(f"   - Removing these engines would not reduce ensemble recall")

    # Engines with highest marginal value.
    best_mv = sorted(engine_names, key=lambda n: -results["marginal_value"][n])
    if best_mv:
        top = best_mv[0]
        w(f"4. **Highest marginal value**: {top} ({results['marginal_value'][top]:.3f})")
        w(f"   - This engine contributes the most unique detections to the ensemble")

    w("")
    w("## Implications for Swarm Architecture")
    w("")
    w("If two engines have high Jaccard similarity AND high conditional co-miss rates,")
    w("they are making correlated errors. Adding both to the swarm provides diminishing")
    w("returns. The swarm benefits most from engines with LOW Jaccard similarity")
    w("(complementary strengths) and LOW conditional co-miss (independent error modes).")
    w("")
    w("---")
    w(f"*Generated by `scripts/diagnose_engine_correlation.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    w("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Engine correlation analysis")
    parser.add_argument(
        "--records", type=int, default=50, help="Number of benchmark records to analyze"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(
            _project_root / "pdlc-artifacts" / "swarm" / "discovery" / "engine-correlation.md"
        ),
        help="Output path for the markdown report",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Engine Correlation and Redundancy Diagnosis")
    print("=" * 70)

    # 1. Load benchmark data.
    print(f"\n[1/3] Loading {args.records} benchmark records...")
    try:
        all_records = load_benchmark_dataset()
    except FileNotFoundError:
        print("  WARNING: Full benchmark dataset not found, using default records.")
        from pii_anon.benchmarks.datasets import _DEFAULT_DATASET
        all_records = list(_DEFAULT_DATASET)

    records = all_records[: args.records]
    total_labels = sum(len(r.labels) for r in records)
    print(f"  Loaded {len(records)} records with {total_labels} gold entities")

    # 2. Initialize engines.
    print("\n[2/3] Initializing engines...")
    engines = init_engines()
    if len(engines) < 2:
        print("\n  ERROR: Need at least 2 engines for correlation analysis.")
        print("  Install missing dependencies: pip install presidio-analyzer spacy stanza gliner scrubadub")
        sys.exit(1)

    # 3. Run analysis.
    print(f"\n[3/3] Running {len(engines)} engines on {len(records)} records...")
    t0 = time.time()
    results = run_analysis(records, engines)
    elapsed = time.time() - t0
    print(f"\n  Analysis complete in {elapsed:.1f}s")

    # 4. Generate report.
    report = format_report(results, len(records))

    # Ensure output directory exists.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"\n  Report written to: {output_path}")

    # Print summary to console.
    print("\n" + "=" * 70)
    print("  QUICK SUMMARY")
    print("=" * 70)
    print(f"  Union recall (all engines): {results['union_recall']:.3f}")
    print(f"  Individual recalls:")
    for name in sorted(results["engine_names"], key=lambda n: -results["engine_recall"][n]):
        r = results["engine_recall"][name]
        mv = results["marginal_value"][name]
        print(f"    {name:12s}  recall={r:.3f}  marginal_value={mv:.3f}")

    jac_pairs = sorted(results["jaccard_matrix"].items(), key=lambda x: -x[1])
    if jac_pairs:
        (a, b), j = jac_pairs[0]
        print(f"\n  Most redundant:  {a} + {b} (Jaccard={j:.3f})")
        (a, b), j = jac_pairs[-1]
        print(f"  Most complementary: {a} + {b} (Jaccard={j:.3f})")
    print()


if __name__ == "__main__":
    main()
