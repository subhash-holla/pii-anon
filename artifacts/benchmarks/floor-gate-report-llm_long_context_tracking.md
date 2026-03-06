# Floor Gate Report (llm_long_context_tracking)

Overall floor pass: `False`
Overall qualification gate pass: `False`
All competitors available: `False`
Failed profiles: `long_document, structured_form_accuracy, multilingual_mix`

Unavailable competitors:
- `llm_guard`: short_chat: llm_guard native scanner unavailable; long_document: llm_guard native scanner unavailable; structured_form_accuracy: llm_guard native scanner unavailable; structured_form_latency: llm_guard native scanner unavailable; log_lines: llm_guard native scanner unavailable; multilingual_mix: llm_guard native scanner unavailable

## Profile `short_chat` (speed)
- floor_pass: `True`
- qualified_competitors: `4`
- latency_p50_ms: actual=0.026, target=0.414, comparator=scrubadub, passed=True
- docs_per_hour: actual=75205504.260, target=6407257.010, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "pii-anon-full", "value": 57051466.13}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "pii-anon-full", "value": 0.061}}`

## Profile `long_document` (accuracy)
- floor_pass: `False`
- qualified_competitors: `4`
- f1: actual=0.564, target=0.835, comparator=gliner, passed=False
- recall: actual=0.819, target=0.737, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.834569}, "recall": {"metric": "recall", "system": "pii-anon-minimal", "value": 0.818721}}`

## Profile `structured_form_accuracy` (accuracy)
- floor_pass: `False`
- qualified_competitors: `4`
- f1: actual=0.564, target=0.835, comparator=gliner, passed=False
- recall: actual=0.819, target=0.737, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.834569}, "recall": {"metric": "recall", "system": "pii-anon-minimal", "value": 0.818721}}`

## Profile `structured_form_latency` (speed)
- floor_pass: `True`
- qualified_competitors: `4`
- latency_p50_ms: actual=0.024, target=0.412, comparator=scrubadub, passed=True
- docs_per_hour: actual=140993607.440, target=7068399.890, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "pii-anon-full", "value": 60509604.32}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "pii-anon-full", "value": 0.059}}`

## Profile `log_lines` (speed)
- floor_pass: `True`
- qualified_competitors: `4`
- latency_p50_ms: actual=0.025, target=0.406, comparator=scrubadub, passed=True
- docs_per_hour: actual=132554135.930, target=7882350.150, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "pii-anon-full", "value": 53315137.08}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "pii-anon-full", "value": 0.061}}`

## Profile `multilingual_mix` (accuracy)
- floor_pass: `False`
- qualified_competitors: `4`
- f1: actual=0.564, target=0.835, comparator=gliner, passed=False
- recall: actual=0.819, target=0.737, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 0.834569}, "recall": {"metric": "recall", "system": "pii-anon-minimal", "value": 0.818721}}`
