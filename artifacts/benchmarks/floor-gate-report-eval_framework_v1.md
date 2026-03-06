# Floor Gate Report (eval_framework_v1)

Overall floor pass: `False`
Overall qualification gate pass: `False`
All competitors available: `False`
Failed profiles: `long_document, structured_form_accuracy, multilingual_mix`

Unavailable competitors:
- `llm_guard`: short_chat: llm_guard native scanner unavailable; long_document: llm_guard native scanner unavailable; structured_form_accuracy: llm_guard native scanner unavailable; structured_form_latency: llm_guard native scanner unavailable; log_lines: llm_guard native scanner unavailable; multilingual_mix: llm_guard native scanner unavailable

## Profile `short_chat` (speed)
- floor_pass: `True`
- qualified_competitors: `4`
- latency_p50_ms: actual=0.006, target=0.140, comparator=scrubadub, passed=True
- docs_per_hour: actual=291403565.010, target=24233367.470, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "pii-anon-full", "value": 56378467.68}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "pii-anon-full", "value": 0.058}}`

## Profile `long_document` (accuracy)
- floor_pass: `False`
- qualified_competitors: `4`
- f1: actual=0.800, target=1.000, comparator=gliner, passed=False
- recall: actual=1.000, target=1.000, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 1.0}, "recall": {"metric": "recall", "system": "pii-anon-minimal", "value": 1.0}}`

## Profile `structured_form_accuracy` (accuracy)
- floor_pass: `False`
- qualified_competitors: `4`
- f1: actual=0.800, target=1.000, comparator=gliner, passed=False
- recall: actual=1.000, target=1.000, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 1.0}, "recall": {"metric": "recall", "system": "pii-anon-minimal", "value": 1.0}}`

## Profile `structured_form_latency` (speed)
- floor_pass: `True`
- qualified_competitors: `4`
- latency_p50_ms: actual=0.006, target=0.135, comparator=scrubadub, passed=True
- docs_per_hour: actual=459574525.940, target=24600246.150, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "pii-anon-full", "value": 59593276.8}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "pii-anon-full", "value": 0.057}}`

## Profile `log_lines` (speed)
- floor_pass: `True`
- qualified_competitors: `4`
- latency_p50_ms: actual=0.007, target=0.141, comparator=scrubadub, passed=True
- docs_per_hour: actual=372144297.040, target=23477189.020, comparator=scrubadub, passed=True
- winners: `{"docs_per_hour": {"metric": "docs_per_hour", "system": "pii-anon-full", "value": 60738136.87}, "latency_p50_ms": {"metric": "latency_p50_ms", "system": "pii-anon-full", "value": 0.056}}`

## Profile `multilingual_mix` (accuracy)
- floor_pass: `False`
- qualified_competitors: `4`
- f1: actual=0.800, target=1.000, comparator=gliner, passed=False
- recall: actual=1.000, target=1.000, comparator=gliner, passed=True
- winners: `{"f1": {"metric": "f1", "system": "gliner", "value": 1.0}, "recall": {"metric": "recall", "system": "pii-anon-minimal", "value": 1.0}}`
