#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any
import time
from collections.abc import Callable
from threading import Event, Thread

from pii_anon import PIIOrchestrator
from pii_anon.benchmarks import load_benchmark_dataset
from pii_anon.config import CoreConfig, TrackingConfig
from pii_anon.types import ProcessingProfileSpec, SegmentationPlan


def _progress(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[progress {timestamp}] {message}", flush=True)


def _safe_div(num: float, denom: float) -> float:
    if denom <= 0:
        return 0.0
    return num / denom


def _overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def _match_audit(
    mention_start: int,
    mention_end: int,
    audits: list[dict[str, Any]],
) -> dict[str, Any] | None:
    best: tuple[float, dict[str, Any]] | None = None
    mention_len = max(1, mention_end - mention_start)
    for audit in audits:
        span = audit.get("span", {})
        a_start = int(span.get("start", -1))
        a_end = int(span.get("end", -1))
        if a_start < 0 or a_end <= a_start:
            continue
        shared = _overlap(mention_start, mention_end, a_start, a_end)
        if shared <= 0:
            continue
        coverage = shared / mention_len
        # Prefer high-coverage matches; break ties by larger overlap.
        score = coverage + (shared / 10_000.0)
        if best is None or score > best[0]:
            best = (score, audit)
    if best is None:
        return None
    # Reject weak overlaps to avoid accidental alias inflation.
    score, audit = best
    if score < 0.60:
        return None
    return audit


def _pairwise_f1(true_clusters: list[str], pred_clusters: list[str]) -> tuple[float, float, float]:
    ids = list(range(len(true_clusters)))
    tp = 0
    pred_pos = 0
    true_pos = 0
    for i, j in combinations(ids, 2):
        true_same = true_clusters[i] == true_clusters[j]
        pred_same = pred_clusters[i] == pred_clusters[j]
        if true_same:
            true_pos += 1
        if pred_same:
            pred_pos += 1
        if true_same and pred_same:
            tp += 1

    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / true_pos if true_pos else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _consistency_ratio(groups: dict[str, set[str]]) -> float:
    if not groups:
        return 0.0
    stable = sum(1 for values in groups.values() if len(values) == 1)
    return stable / len(groups)


def _evaluate_tracking_dataset(
    max_samples: int | None,
    *,
    progress_hook: Callable[[str], None] | None = None,
    progress_interval_sec: int = 10,
) -> dict[str, Any]:
    if progress_hook:
        progress_hook("phase: tracking dataset load start")
    all_records = load_benchmark_dataset("pii_anon_benchmark_v1")
    records = [r for r in all_records if r.scenario_id.startswith("continuity_")]
    if progress_hook:
        progress_hook(f"phase: tracking dataset load complete (records={len(records)})")
    if max_samples is not None:
        records = records[:max_samples]
        if progress_hook:
            progress_hook(f"phase: tracking dataset capped (records={len(records)})")

    continuity_config = CoreConfig(
        tracking=TrackingConfig(
            enabled=True,
            min_link_score=0.72,
            allow_email_name_link=True,
            require_unique_short_name=True,
        )
    )
    orchestrator = PIIOrchestrator(token_key="continuity-key", config=continuity_config)
    if progress_hook:
        progress_hook("phase: tracking evaluation start")

    true_clusters: list[str] = []
    pred_clusters: list[str] = []
    pseudo_groups: dict[str, set[str]] = {}
    anon_groups: dict[str, set[str]] = {}
    ambiguous_cases = 0
    ambiguous_cluster_collisions = 0

    last_heartbeat = time.time()
    total_records = max(1, len(records))
    for idx, item in enumerate(records, start=1):
        mentions = [
            label
            for label in item.labels
            if label.get("entity_type") in {"PERSON_NAME", "EMAIL_ADDRESS"} and label.get("entity_cluster_id")
        ]
        if not mentions:
            continue

        scope_key = f"tracking:{item.scenario_id}:{item.record_id}"

        pseudonym_result = orchestrator.run(
            {"text": item.text},
            profile=ProcessingProfileSpec(
                profile_id="continuity-pseudo",
                mode="weighted_consensus",
                language=item.language,
                use_case="long_document",
                objective="accuracy",
                transform_mode="pseudonymize",
                entity_tracking_enabled=True,
            ),
            segmentation=SegmentationPlan(enabled=False),
            scope=scope_key,
            token_version=1,
        )
        anonymize_result = orchestrator.run(
            {"text": item.text},
            profile=ProcessingProfileSpec(
                profile_id="continuity-anon",
                mode="weighted_consensus",
                language=item.language,
                use_case="long_document",
                objective="accuracy",
                transform_mode="anonymize",
                entity_tracking_enabled=True,
            ),
            segmentation=SegmentationPlan(enabled=False),
            scope=scope_key,
            token_version=1,
        )

        pseudo_audits = [audit for audit in pseudonym_result.get("link_audit", []) if isinstance(audit, dict)]
        anon_audits = [audit for audit in anonymize_result.get("link_audit", []) if isinstance(audit, dict)]

        if item.scenario_id == "continuity_ambiguous":
            ambiguous_cases += 1
            token_by_cluster: dict[str, set[str]] = {}
            for mention in mentions:
                m_start = int(mention["start"])
                m_end = int(mention["end"])
                cluster_id = str(mention.get("entity_cluster_id", "unknown"))
                pseudo_audit = _match_audit(m_start, m_end, pseudo_audits)
                if pseudo_audit is None:
                    continue
                token_by_cluster.setdefault(cluster_id, set()).add(str(pseudo_audit.get("replacement", "")))
            representative_tokens = [next(iter(values)) for values in token_by_cluster.values() if values]
            if len(representative_tokens) != len(set(representative_tokens)):
                ambiguous_cluster_collisions += 1
            continue

        for mention in mentions:
            m_start = int(mention["start"])
            m_end = int(mention["end"])
            cluster_id = str(mention.get("entity_cluster_id", "unknown"))
            pseudo_audit = _match_audit(m_start, m_end, pseudo_audits)
            anon_audit = _match_audit(m_start, m_end, anon_audits)
            if pseudo_audit is None or anon_audit is None:
                continue
            true_clusters.append(cluster_id)
            pred_clusters.append(f"{scope_key}:{pseudo_audit.get('cluster_id', 'missing')}")

            pseudo_groups.setdefault(cluster_id, set()).add(str(pseudo_audit.get("replacement", "")))
            anon_groups.setdefault(cluster_id, set()).add(str(anon_audit.get("replacement", "")))

        if progress_hook and (time.time() - last_heartbeat) >= max(1, progress_interval_sec):
            pct = int((idx / total_records) * 100)
            progress_hook(f"phase: tracking evaluation progress {idx}/{total_records} ({pct}%)")
            last_heartbeat = time.time()

    precision, recall, alias_link_f1 = _pairwise_f1(true_clusters, pred_clusters)
    pseudonym_consistency = _consistency_ratio(pseudo_groups)
    anonymize_consistency = _consistency_ratio(anon_groups)

    results = {
        "samples": len(records),
        "mentions_evaluated": len(true_clusters),
        "alias_link_precision": round(precision, 6),
        "alias_link_recall": round(recall, 6),
        "alias_link_f1": round(alias_link_f1, 6),
        "pseudonym_consistency": round(pseudonym_consistency, 6),
        "anonymize_placeholder_consistency": round(anonymize_consistency, 6),
        "ambiguous_cases": ambiguous_cases,
        "ambiguous_overlink_rate": round(
            _safe_div(float(ambiguous_cluster_collisions), float(ambiguous_cases)), 6
        ),
    }
    if progress_hook:
        progress_hook(
            "phase: tracking evaluation complete "
            f"(alias_f1={results['alias_link_f1']:.3f}, pseudonym_consistency={results['pseudonym_consistency']:.3f})"
        )
    return results


def _long_context_smoke(
    long_token_count: int,
    *,
    progress_hook: Callable[[str], None] | None = None,
    progress_token_step: int = 50000,
    progress_interval_sec: int = 10,
) -> dict[str, Any]:
    alias_segment = "Jack Davis alias Jack contacted jackdavis@example.com for updates."
    filler_tokens = [f"context{i % 97}" for i in range(180)]
    segment = f"{alias_segment} {' '.join(filler_tokens)}"
    tokens_per_segment = len(segment.split())
    target_tokens = max(tokens_per_segment, long_token_count)
    if progress_hook:
        progress_hook(
            f"phase: long-context generation start (target_tokens={target_tokens}, step={progress_token_step})"
        )

    parts: list[str] = []
    generated = 0
    next_log = max(tokens_per_segment, progress_token_step)
    while generated < target_tokens:
        parts.append(segment)
        generated += tokens_per_segment
        if progress_hook and generated >= next_log:
            pct = int((generated / target_tokens) * 100)
            progress_hook(f"phase: long-context generation progress tokens={generated} ({pct}%)")
            next_log += max(tokens_per_segment, progress_token_step)

    text = " ".join(parts)
    segment_count = len(parts)
    token_count = len(text.split())
    if progress_hook:
        progress_hook(f"phase: long-context generation complete (actual_tokens={token_count})")

    continuity_config = CoreConfig(
        tracking=TrackingConfig(
            enabled=True,
            min_link_score=0.72,
            allow_email_name_link=True,
            require_unique_short_name=True,
        )
    )
    orchestrator = PIIOrchestrator(token_key="continuity-key", config=continuity_config)
    heartbeat_stop = Event()
    heartbeat_thread: Thread | None = None
    orchestration_start = time.time()
    if progress_hook:
        progress_hook("phase: long-context orchestration start")

        def _heartbeat() -> None:
            while not heartbeat_stop.wait(timeout=max(1, progress_interval_sec)):
                elapsed = int(time.time() - orchestration_start)
                progress_hook(f"phase: long-context orchestration heartbeat elapsed={elapsed}s")

        heartbeat_thread = Thread(target=_heartbeat, daemon=True)
        heartbeat_thread.start()
    try:
        out = orchestrator.run(
            {"text": text},
            profile=ProcessingProfileSpec(
                profile_id="long-context",
                mode="weighted_consensus",
                language="en",
                use_case="long_document",
                objective="accuracy",
                transform_mode="pseudonymize",
                entity_tracking_enabled=True,
            ),
            segmentation=SegmentationPlan(enabled=True, max_tokens=2048, overlap_tokens=128),
            scope="long-context-smoke",
            token_version=1,
        )
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1.0)
    if progress_hook:
        elapsed = int(time.time() - orchestration_start)
        progress_hook(f"phase: long-context orchestration complete elapsed={elapsed}s")

    audits = out.get("link_audit", [])
    alias_set = {"jack davis", "jack", "jackdavis@example.com"}
    target_mentions = [
        item
        for item in audits
        if str(item.get("mention_text", "")).strip().lower() in alias_set
    ]
    replacements = {str(item.get("replacement", "")) for item in target_mentions}
    consistency = 1.0 if target_mentions and len(replacements) == 1 else 0.0
    mentions_expected = segment_count * 3
    mentions_linked = len(target_mentions)
    alias_recall = _safe_div(float(mentions_linked), float(mentions_expected))

    cluster_counts: dict[str, int] = {}
    for item in target_mentions:
        cluster_id = str(item.get("cluster_id", ""))
        if not cluster_id:
            continue
        cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
    primary_cluster = max(cluster_counts, key=cluster_counts.get) if cluster_counts else ""
    primary_cluster_mentions = [
        item for item in audits if str(item.get("cluster_id", "")) == primary_cluster
    ]
    alias_precision = _safe_div(float(mentions_linked), float(len(primary_cluster_mentions)))

    return {
        "token_count": token_count,
        "mentions_expected": mentions_expected,
        "mentions_linked": mentions_linked,
        "long_context_alias_recall": round(alias_recall, 6),
        "long_context_alias_precision": round(alias_precision, 6),
        "mentions_found": len(target_mentions),
        "long_context_pseudonym_consistency": consistency,
        "boundary_trace": out.get("boundary_trace"),
    }


def _to_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Continuity Gate Report",
        "",
        f"- alias_link_precision: `{results['alias_link_precision']:.3f}`",
        f"- alias_link_recall: `{results['alias_link_recall']:.3f}`",
        f"- alias_link_f1: `{results['alias_link_f1']:.3f}`",
        f"- pseudonym_consistency: `{results['pseudonym_consistency']:.3f}`",
        f"- anonymize_placeholder_consistency: `{results['anonymize_placeholder_consistency']:.3f}`",
        f"- ambiguous_cases: `{results['ambiguous_cases']}`",
        f"- ambiguous_overlink_rate: `{results['ambiguous_overlink_rate']:.3f}`",
        f"- long_context_alias_recall: `{results['long_context_alias_recall']:.3f}`",
        f"- long_context_alias_precision: `{results['long_context_alias_precision']:.3f}`",
        f"- mentions_expected: `{results['mentions_expected']}`",
        f"- mentions_linked: `{results['mentions_linked']}`",
        f"- long_context_pseudonym_consistency: `{results['long_context_pseudonym_consistency']:.3f}`",
        f"- continuity_gate_pass: `{results['continuity_gate_pass']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run continuity tracking benchmark and gates")
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for tracking dataset rows")
    parser.add_argument("--long-token-count", type=int, default=100000, help="Synthetic long-context token target")
    parser.add_argument("--progress-interval-sec", type=int, default=10)
    parser.add_argument("--progress-token-step", type=int, default=50000)
    parser.add_argument("--min-alias-f1", type=float, default=0.95)
    parser.add_argument("--min-pseudonym-consistency", type=float, default=0.99)
    parser.add_argument("--min-anonymize-consistency", type=float, default=0.99)
    parser.add_argument("--min-long-alias-recall", type=float, default=0.95)
    parser.add_argument("--min-long-alias-precision", type=float, default=0.90)
    parser.add_argument("--output-json", default="artifacts/benchmarks/continuity-results.json")
    parser.add_argument("--output-markdown", default="artifacts/benchmarks/continuity-gate-report.md")
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument("--quiet-progress", action="store_true")
    args = parser.parse_args()

    progress_hook = None if args.quiet_progress else _progress
    if progress_hook:
        progress_hook("continuity benchmark start")
    tracking = _evaluate_tracking_dataset(
        args.max_samples if args.max_samples > 0 else None,
        progress_hook=progress_hook,
        progress_interval_sec=args.progress_interval_sec,
    )
    long_ctx = _long_context_smoke(
        args.long_token_count,
        progress_hook=progress_hook,
        progress_token_step=args.progress_token_step,
        progress_interval_sec=args.progress_interval_sec,
    )

    results = {
        **tracking,
        **long_ctx,
    }
    results["continuity_gate_pass"] = bool(
        results["alias_link_f1"] >= args.min_alias_f1
        and results["pseudonym_consistency"] >= args.min_pseudonym_consistency
        and results["anonymize_placeholder_consistency"] >= args.min_anonymize_consistency
        and results["long_context_pseudonym_consistency"] >= args.min_pseudonym_consistency
        and results["long_context_alias_recall"] >= args.min_long_alias_recall
        and results["long_context_alias_precision"] >= args.min_long_alias_precision
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")

    out_md = Path(args.output_markdown)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_to_markdown(results), encoding="utf-8")

    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    if progress_hook:
        progress_hook(f"continuity benchmark complete (gate_pass={results['continuity_gate_pass']})")

    if args.enforce and not results["continuity_gate_pass"]:
        raise SystemExit("continuity gate failed")


if __name__ == "__main__":
    main()
