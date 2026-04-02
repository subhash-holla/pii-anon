# Floor Gate Report (pii_anon_benchmark)

Overall floor pass: `False`
Overall qualification gate pass: `True`
All competitors available: `True`
Failed profiles: `short_chat, structured_form_latency, log_lines`

## Profile `short_chat` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.403, target=0.247, comparator=scrubadub, passed=False
- docs_per_hour: actual=3064894.880, target=4762130.230, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 4762130.23}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.247}}`

## Profile `long_document` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.816, target=0.766, comparator=gliner, passed=True
- recall: actual=0.799, target=0.661, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.766379}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.818066}}`

## Profile `structured_form_accuracy` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.816, target=0.766, comparator=gliner, passed=True
- recall: actual=0.799, target=0.661, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.766379}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.818066}}`

## Profile `structured_form_latency` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.391, target=0.240, comparator=scrubadub, passed=False
- docs_per_hour: actual=3317116.720, target=5254143.280, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5254143.28}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.24}}`

## Profile `log_lines` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.391, target=0.241, comparator=scrubadub, passed=False
- docs_per_hour: actual=3303330.090, target=5249625.200, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5249625.2}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.241}}`

## Profile `multilingual_mix` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.816, target=0.766, comparator=gliner, passed=True
- recall: actual=0.799, target=0.661, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.766379}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.818066}}`
