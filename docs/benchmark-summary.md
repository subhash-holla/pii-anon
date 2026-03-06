## Composite Evaluation Results

Evaluated on 50,000 records from the `pii_anon_benchmark_v1` dataset (research-grade unified benchmark with 7 evaluation dimensions covering core PII detection, long-context entity tracking, and dimension-specific probes) using the **PII-Rate-Elo composite metric** — a weighted score combining F1 (50%), precision (15%), recall (15%), latency (10%), and throughput (10%), followed by an Elo round-robin tournament.

| Rank | System | Composite | F1 | Precision | Recall | Latency (ms) | Docs/hr | Elo |
|:----:|--------|:---------:|:--:|:---------:|:------:|:------------:|:-------:|:---:|
| **1** | **pii-anon** | **0.6233** | **0.6116** | 0.5797 | **0.6472** | 6.942 | 514K | **1525** |
| 2 | scrubadub | 0.5002 | 0.3320 | 0.7305 | 0.2148 | 0.255 | 12.2M | 1494 |
| 3 | spaCy NER | 0.4857 | 0.2576 | 0.9954 | 0.1479 | 0.604 | 5.9M | 1491 |
| 4 | stanza | 0.4676 | 0.2035 | 1.0000 | 0.1133 | 0.025 | 88.1M | 1487 |

**pii-anon leads by 25% on the composite metric** over the next-best competitor (scrubadub at 0.50). It achieves the highest F1, the highest recall, and the highest Elo rating in the tournament. Presidio and LLM Guard were unavailable for this run due to network-dependent model downloads — see [Benchmark Methodology](#benchmark-methodology) for details.

This section is generated from benchmark artifacts.
