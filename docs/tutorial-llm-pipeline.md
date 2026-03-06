# Tutorial: LLM Pipeline PII Guardrail

## Goal

Sanitize input corpora before LLM training or RAG ingestion, and validate both benchmark and continuity evidence.

## Steps

1. Run baseline detection with explicit transform mode:

```bash
pii-anon detect "User Jack Davis email jackdavis@example.com" --transform-mode pseudonymize --output json
```

2. Validate anonymization output mode:

```bash
pii-anon detect "User Jack Davis email jackdavis@example.com" --transform-mode anonymize --output json
```

3. Evaluate strategies on bundled benchmark dataset:

```bash
pii-anon evaluate \
  --strategies weighted_consensus,union_high_recall,intersection_consensus \
  --dataset pii_anon_benchmark_v1 \
  --output json
```

4. Compare against competitor baselines:

```bash
pii-anon compare-competitors --dataset pii_anon_benchmark_v1 --output json
```

5. Run continuity gate on long-context dataset:

```bash
PYTHONPATH=src python scripts/run_continuity_benchmark.py \
  --max-samples 0 \
  --long-token-count 100000 \
  --enforce
```

6. Use stream mode for large corpus batches:

```bash
pii-anon detect-stream ./sample_payloads.txt --transform-mode pseudonymize --output json
```

## Evidence policy

Every claim in docs must map to reproducible commands and artifacts in `docs/evidence-ledger.md`.
