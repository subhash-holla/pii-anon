# Floor Gate Report (pii_anon_benchmark_v1)

Overall floor pass: `False`
Overall qualification gate pass: `True`
All competitors available: `True`
Failed profiles: `short_chat, structured_form_latency, log_lines`

## Profile `short_chat` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.368, target=0.262, comparator=scrubadub, passed=False
- docs_per_hour: actual=5077463.350, target=7826710.230, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 7826710.23}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.262}}`

## Profile `long_document` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.874, target=0.774, comparator=gliner, passed=True
- recall: actual=0.892, target=0.667, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.774279}, "recall": {"metric": "recall", "system": "pii-anon-ensemble", "value": 0.922245}}`

## Profile `structured_form_accuracy` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.874, target=0.774, comparator=gliner, passed=True
- recall: actual=0.892, target=0.667, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.774279}, "recall": {"metric": "recall", "system": "pii-anon-ensemble", "value": 0.922245}}`

## Profile `structured_form_latency` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.372, target=0.264, comparator=scrubadub, passed=False
- docs_per_hour: actual=3339855.090, target=5404411.930, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5404411.93}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.264}}`

## Profile `log_lines` (speed)
- floor_pass: `False`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.373, target=0.264, comparator=scrubadub, passed=False
- docs_per_hour: actual=3064305.230, target=5093290.610, comparator=scrubadub, passed=False
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 5093290.61}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.264}}`

## Profile `multilingual_mix` (accuracy)
- floor_pass: `True`
- qualified_competitors: `3`
- f1: actual=0.874, target=0.774, comparator=gliner, passed=True
- recall: actual=0.892, target=0.667, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.774279}, "recall": {"metric": "recall", "system": "pii-anon-ensemble", "value": 0.922245}}`
