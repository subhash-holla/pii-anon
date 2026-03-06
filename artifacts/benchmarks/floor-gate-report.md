# Floor Gate Report (pii_anon_benchmark_v1)

Overall floor pass: `False`
Overall qualification gate pass: `True`
All competitors available: `True`
Failed profiles: `long_document, structured_form_accuracy, multilingual_mix`

## Profile `short_chat` (speed)
- floor_pass: `True`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.012, target=0.287, comparator=scrubadub, passed=True
- docs_per_hour: actual=126367462.180, target=7572884.190, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 7572884.19}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.287}}`

## Profile `long_document` (accuracy)
- floor_pass: `False`
- qualified_competitors: `3`
- f1: actual=0.702, target=0.774, comparator=gliner, passed=False
- recall: actual=0.894, target=0.667, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.774279}, "recall": {"metric": "recall", "system": "pii-anon-full", "value": 0.894644}}`

## Profile `structured_form_accuracy` (accuracy)
- floor_pass: `False`
- qualified_competitors: `3`
- f1: actual=0.702, target=0.774, comparator=gliner, passed=False
- recall: actual=0.894, target=0.667, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.774279}, "recall": {"metric": "recall", "system": "pii-anon-full", "value": 0.894644}}`

## Profile `structured_form_latency` (speed)
- floor_pass: `True`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.011, target=0.265, comparator=scrubadub, passed=True
- docs_per_hour: actual=113478186.910, target=11895404.080, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 11895404.08}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.265}}`

## Profile `log_lines` (speed)
- floor_pass: `True`
- qualified_competitors: `3`
- latency_p50_ms: actual=0.011, target=0.265, comparator=scrubadub, passed=True
- docs_per_hour: actual=73289941.240, target=11869276.330, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "scrubadub", "value": 11869276.33}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "scrubadub", "value": 0.265}}`

## Profile `multilingual_mix` (accuracy)
- floor_pass: `False`
- qualified_competitors: `3`
- f1: actual=0.702, target=0.774, comparator=gliner, passed=False
- recall: actual=0.894, target=0.667, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.774279}, "recall": {"metric": "recall", "system": "pii-anon-full", "value": 0.894644}}`
