# Anonymization vs Pseudonymization — two distinct scorer families

pii-anon treats **anonymization** and **pseudonymization** as two structurally
distinct de-identification families that are **never merged** into a single
de-id score (the project's AX-004 axiom; the **no-merge** CI guard in
`tests/test_deid_families.py` is the standing regression gate). They answer
different questions and carry different risk semantics:

| | Anonymization | Pseudonymization |
|---|---|---|
| Question | Is the original **unrecoverable**? | Is reversal **authorized-only**, referentially consistent, and key-separated? |
| Reversibility | IRREVERSIBLE by design | REVERSIBLE under a controlled key |
| Scorer | `AnonymizationScorer` → `AnonymizationScore` | `PseudonymizationIntegrityScorer` → `PseudonymizationIntegrityScore` |
| Headline number | `irreversibility_score` = `1 − max(risk axes)` (re-identification, leakage, canary exposure) | `integrity_score` over the reversible-integrity axes |
| Key axes | `ReidentificationRiskMetric`, `LeakageDetectionMetric`, `CanaryExposureMetric` | `unauthorized_reversal_rate` (must be 0), `referential_integrity` (same plaintext → same surrogate), `collision_rate`, `key_state_separation_ok` (the artifact alone must not re-join — Art-4(5) separation) |

Both scorers live in `pii_anon.eval_framework.metrics.deid_families` and are
pure, deterministic functions of their inputs. The carrier record
(`DeidFamilyScores`) holds the two sub-records **side by side — never
combined**: there is no `combined`, `deid_score`, or `privacy_score` field
anywhere on it.

```python
from pii_anon.eval_framework.metrics.deid_families import (
    AnonymizationScorer,
    PseudonymizationIntegrityScorer,
)
```

At runtime the same split is visible in the transform surface
(`ProcessingProfileSpec.transform_mode: "pseudonymize" | "anonymize"`) and in
the reversible token machinery (`EncryptedSQLiteTokenStore` — AEAD-encrypted
at rest, keyed surrogates, authorized reversal only). See
[configuration.md](configuration.md) for the profile fields and
[api-reference.md](api-reference.md) for the full surface.

## Vanilla vs swarm — where each family earns its keep

- **Vanilla `pii-anon`** is the single-pipeline detector: the always-on,
  checksum/keyword-gated regex shared layer plus optional native engines.
  Fast, deterministic, dependency-light — the right default for log
  scrubbing and latency-sensitive interception, in either transform mode.
- **`pii-anon-swarm`** is the multi-engine MoE ensemble on top: learned
  routing, fusion, and the **recall-floor by construction**
  ([recall-floor.md](recall-floor.md) — `entities(output) ⊇
  entities(shared)`, so no downstream gate can drop a shared-layer span).
  The swarm is where recall-critical de-identification (the anonymization
  family's leak axes) gains headroom, at the cost of heavier engines.
- The **evaluation framework scores both on the identical path**
  ([evaluate-your-pipeline.md](evaluate-your-pipeline.md)): detection
  quality (F1/F2, latency, throughput) via the composite, the two de-id
  families via their distinct scorers, and — for certified claims — the
  `canonical-run` / `supremacy` CLI, whose G2 guarantee reads the
  pseudonymization-integrity family **separately** from the anonymization
  axes, exactly because the two are never merged.
