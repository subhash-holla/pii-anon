# Floor Gate Report (pii_anon_benchmark_v1)

Overall floor pass: `False`
Overall qualification gate pass: `True`
All competitors available: `True`
Failed profiles: `short_chat, structured_form_latency, log_lines`

## Profile `short_chat` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.345, target=0.223, comparator=scrubadub, passed=False
- docs_per_hour: actual=3734200.130, target=5372400.060, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5372400.06}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.223}}`

## Profile `long_document` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.845, target=0.763, comparator=gliner, passed=True
- recall: actual=0.823, target=0.658, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.762852}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.839141}}`

## Profile `structured_form_accuracy` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.845, target=0.763, comparator=gliner, passed=True
- recall: actual=0.823, target=0.658, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.762852}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.839141}}`

## Profile `structured_form_latency` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.347, target=0.225, comparator=scrubadub, passed=False
- docs_per_hour: actual=3807712.970, target=5793045.630, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5793045.63}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.225}}`

## Profile `log_lines` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.346, target=0.224, comparator=scrubadub, passed=False
- docs_per_hour: actual=3831356.770, target=5762120.840, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5762120.84}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.224}}`

## Profile `multilingual_mix` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.845, target=0.763, comparator=gliner, passed=True
- recall: actual=0.823, target=0.658, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.762852}, "recall": {"metric": "recall", "system": "pii-anon-swarm", "value": 0.839141}}`
