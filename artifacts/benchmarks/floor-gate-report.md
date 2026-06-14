# Floor Gate Report (pii_anon_benchmark)

Overall floor pass: `False`
Overall qualification gate pass: `True`
All competitors available: `True`
Failed profiles: `short_chat, structured_form_latency, log_lines`

## Profile `short_chat` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.458, target=0.237, comparator=scrubadub, passed=False
- docs_per_hour: actual=3844235.700, target=8179166.060, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 8179166.06}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.237}}`

## Profile `long_document` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.814, target=0.764, comparator=gliner, passed=True
- recall: actual=0.799, target=0.658, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.763652}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.817971}}`

## Profile `structured_form_accuracy` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.814, target=0.764, comparator=gliner, passed=True
- recall: actual=0.799, target=0.658, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.763652}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.817971}}`

## Profile `structured_form_latency` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.443, target=0.239, comparator=scrubadub, passed=False
- docs_per_hour: actual=3061882.420, target=5301907.160, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5301907.16}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.239}}`

## Profile `log_lines` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.443, target=0.238, comparator=scrubadub, passed=False
- docs_per_hour: actual=3056024.240, target=5329796.600, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5329796.6}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.238}}`

## Profile `multilingual_mix` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.814, target=0.764, comparator=gliner, passed=True
- recall: actual=0.799, target=0.658, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.763652}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.817971}}`
