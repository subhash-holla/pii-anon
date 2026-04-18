# Extending pii-anon-swarm

Two extension workflows, both first-class:

1. **Bring your own detector** — plug a proprietary / domain-specific PII
   engine into the swarm alongside the built-in regex / Presidio / GLiNER /
   Stanza pool. The fusion pipeline picks it up automatically.
2. **Bring your own data** — retrain the swarm meta-learner, Dawid-Skene
   confusion matrices, and F2-tuned emission threshold on your own
   labeled PII corpus.

The two compose — you can do either independently, or chain them (plug in
your detector, then retrain against your domain data so Dawid-Skene learns
its confusion and the meta-learner learns its weight).

For the pipeline architecture itself see [swarm-architecture.md](swarm-architecture.md).
For the evaluation framework see [pii-rate-elo.md](pii-rate-elo.md).

---

## Workflow 1 — Plug your own detector into the swarm

### The `EngineAdapter` contract

Every engine — built-in or custom — implements the same ABC at
[engines/base.py](../src/pii_anon/engines/base.py). Only one method is
mandatory:

```python
from pii_anon.engines import EngineAdapter, EngineFinding
from pii_anon.types import Payload

class MyDetector(EngineAdapter):
    adapter_id = "my-detector"          # unique engine ID
    supported_entity_types = {"PERSON_NAME", "EMAIL_ADDRESS"}

    def detect(self, payload: Payload, context: dict) -> list[EngineFinding]:
        text = payload["text"]
        return [
            EngineFinding(
                entity_type="EMAIL_ADDRESS",
                confidence=0.92,
                span_start=8,
                span_end=25,
                engine_id=self.adapter_id,
                language=context.get("language", "en"),
                explanation="my-detector:rule-A",
            ),
        ]
```

Lifecycle hooks (`initialize`, `dependency_available`, `capabilities`,
`health_check`, `shutdown`) have working base-class implementations —
override only when your adapter needs them. See
[engine-plugin-guide.md](engine-plugin-guide.md) for the full walkthrough
and the MoE expert-profile integration.

### Registering your engine

Three registration paths are available (pick one):

```python
# Runtime Python API — easiest for notebooks and scripts
from pii_anon import PIIOrchestrator
orch = PIIOrchestrator(token_key="...")
orch.register_engine(MyDetector(enabled=True))
```

```toml
# Entry point auto-discovery — best for pip-installed plugins
# in your own package's pyproject.toml:
[project.entry-points."pii_anon.engines"]
my-detector = "my_package.detector:MyDetector"
```

```yaml
# Config-driven — best for production deployments
engines:
  my-detector:
    enabled: true
    confidence_floor: 0.6
```

See [engine-plugin-guide.md](engine-plugin-guide.md) for the full registry
surface.

### Ensuring your engine survives Layer 2 pruning

The swarm's Layer 2 greedy set-cover pruner keeps the most diverse
engines (ranked by distinct entity types they detect) and drops any whose
type set has Jaccard similarity ≥ 0.85 with an already-selected engine.
`regex-oss` is always pinned because its checksum validators are
stronger than the heuristic. **Pin your engine the same way** via
`SwarmConfig.force_include_engines`:

```python
from pii_anon import PIIOrchestrator
from pii_anon.swarm import SwarmConfig

swarm_cfg = SwarmConfig(force_include_engines=("my-detector",))
orch = PIIOrchestrator(
    token_key="...",
    swarm_config=swarm_cfg,
)
orch.register_engine(MyDetector(enabled=True))
```

Pinned engines:

- Always survive Layer 2 regardless of type-set overlap.
- Bypass the `max_engines` cap (so pinning two custom engines plus
  `regex-oss` is fine even with the default `max_engines=4`).
- Participate in the Dawid-Skene vote, temperature calibration, and
  meta-learner feature vector exactly like built-in engines.

### Graceful degradation without retraining

When you run the swarm with a newly-added engine *before* retraining,
the library degrades gracefully rather than silently ignoring your
findings:

| Component | Behavior for unknown engine |
|---|---|
| Dawid-Skene | Engine's vote doesn't participate in the Bayesian update (no confusion matrix), but it still increments `corroboration_count` |
| Temperature scaling | Default `T=1.0` (identity — no calibration) |
| Informativeness | Default weight `0.5` (half-weight in boundary voting) |
| Meta-learner | Finding flows into `max/min/mean` engine-confidence features (features 4–7) and `corroboration_count` (feature 2). No per-engine feature slot exists, so no retrain is required for basic participation |
| Layer 4 corroboration gate | Your engine contributes to the corroboration count for SEMANTIC_TYPES |

This means **you get useful fusion output on day 1**. Retraining unlocks
the precision-tuning path (see Workflow 2) — higher-quality Dawid-Skene
correction and informativeness weights specific to your engine.

### When to retrain

Retrain after adding a custom engine when:

- Your engine is systematically noisy on a class of entities and you
  want Dawid-Skene to learn its confusion matrix to dampen the noise.
- Your engine overlaps heavily with an existing engine and you want the
  meta-learner to deconflict their votes rather than just counting them.
- Your engine changes the entity-type coverage of the swarm — e.g. it
  detects new types (`MEDICAL_DEVICE_UDI`, `BIOMETRIC_ID`) that
  weren't in the training distribution.

See [swarm-architecture.md#retraining](swarm-architecture.md#retraining) for the
retrain procedure. The `manifest.json` records the set of engines the
artifacts were trained against — the loader rejects an artifact whose
engine set has drifted from the current registry.

### Industry-leading engines you can plug in

The `pii-anon` swarm already bundles `regex-oss`, Presidio, GLiNER,
Stanza, spaCy, and LLM Guard out of the box. The pii-rate-elo paper
evaluation pipeline exercises three additional modern PII detectors
that are trivially pluggable via the same `EngineAdapter` interface.
All three are MIT/Apache licensed, but each comes with a different
dependency / license footprint — register them only when you want
the swarm to include them in Dawid-Skene + meta-learner fusion:

| Engine | Paper / model | Strengths | Dependency notes |
|---|---|---|---|
| **Piiranha** | `iiiorg/piiranha-v1-detect-personal-information` — DeBERTa-v3-base 2024 | High single-model F1 (~0.97 in-distribution) across 17 entity types, 6 languages | Requires `transformers` + ~440 MB model weights; CUDA optional |
| **Flair NER** | `flair/ner-english-large` — transformer-based SequenceTagger | Strong English NER (PER / LOC / ORG) with span offsets; robust on long documents | Requires `flair` + PyTorch; CUDA optional |
| **Gretel GLiNER-bi** | `gretelai/gretel-gliner-bi-base-v1.0` — multilingual GLiNER | Zero-shot across 9+ languages; good for domain-specific label sets | Requires `gliner` + `transformers`; already installed if you have the base `[engines]` extra |

To wire one up, subclass `EngineAdapter` and register it via any of
the three paths in the previous section. A minimal Piiranha adapter:

```python
from pii_anon.engines.base import EngineAdapter
from pii_anon.types import EngineFinding, Payload

_PIIRANHA_TO_CANONICAL = {
    "I-GIVENNAME": "PERSON_NAME", "I-SURNAME": "PERSON_NAME",
    "I-EMAIL": "EMAIL_ADDRESS", "I-TELEPHONENUM": "PHONE_NUMBER",
    "I-SOCIALNUM": "US_SSN", "I-CREDITCARDNUMBER": "CREDIT_CARD",
    "I-DATEOFBIRTH": "DATE_OF_BIRTH", "I-STREET": "ADDRESS",
    # ... map the remaining BIO labels to your taxonomy
}

class PiiranhaAdapter(EngineAdapter):
    adapter_id = "piiranha"
    native_dependency = "transformers"
    supported_entity_types = set(_PIIRANHA_TO_CANONICAL.values())

    def initialize(self, config):
        from transformers import pipeline
        self._ner = pipeline(
            "token-classification",
            model="iiiorg/piiranha-v1-detect-personal-information",
            aggregation_strategy="simple",
        )

    def detect(self, payload: Payload, context: dict) -> list[EngineFinding]:
        text = payload.get("text", "")
        findings = []
        for ent in self._ner(text):
            canonical = _PIIRANHA_TO_CANONICAL.get(ent["entity_group"])
            if not canonical:
                continue
            findings.append(EngineFinding(
                entity_type=canonical,
                confidence=float(ent["score"]),
                span_start=int(ent["start"]),
                span_end=int(ent["end"]),
                engine_id=self.adapter_id,
                language=context.get("language", "en"),
            ))
        return findings
```

Pin the engine past the Layer 2 pruner and retrain the swarm with it
in the pool:

```python
from pii_anon import PIIOrchestrator
from pii_anon.swarm import SwarmConfig

orch = PIIOrchestrator(
    token_key="...",
    swarm_config=SwarmConfig(force_include_engines=("piiranha",)),
)
orch.register_engine(PiiranhaAdapter(enabled=True))
```

```bash
make train-swarm SWARM_DATASETS=pii_anon_eval,ai4privacy_400k,tab,meddocan \
                 SWARM_MAX_RECORDS=0 SWARM_KFOLD=5
```

After retraining, Dawid-Skene knows Piiranha's confusion matrix,
temperature scaling calibrates its logits, and the XGBoost meta-learner
gives it weight based on its demonstrated F1 on your training mix.
See [pii-rate-elo.md](pii-rate-elo.md) for the full algorithm and
[swarm-architecture.md](swarm-architecture.md) for the retrain cadence.

---

## Workflow 2 — Train the swarm on your own data

You have a JSONL file of labeled PII for your domain (medical notes,
legal filings, internal chat logs, KYC documents, etc.). You want the
swarm meta-learner, Dawid-Skene confusion, and F2-tuned emission
threshold to specialise on it.

### The expected JSONL schema

Each line is one record:

```json
{
  "record_id": "doc-0001",
  "text": "Patient John Smith, DOB 1985-04-12, ...",
  "annotations": [
    {"entity_type": "PERSON_NAME", "start": 8, "end": 18},
    {"entity_type": "DATE_OF_BIRTH", "start": 24, "end": 34}
  ],
  "language": "en"
}
```

Required: `text`, `annotations[].entity_type`, `annotations[].start`,
`annotations[].end`. `record_id` and `language` are optional
(auto-generated / default to `"en"`). Gzip-compressed files (`.jsonl.gz`)
are auto-decoded.

### Canonical labels: no mapping needed

If your annotations already use the canonical `pii-anon` taxonomy
(`PERSON_NAME`, `EMAIL_ADDRESS`, `US_SSN`, `DATE_OF_BIRTH`, etc.), just
point the loader at the file:

```python
from pii_anon.swarm_datasets import load_jsonl

records = load_jsonl("my_clinical_data.jsonl")
```

### Custom vocabulary: register a taxonomy

If your data uses private entity names, register the mapping once
before loading. Raw labels mapped to `"_IGNORE"` are dropped:

```python
from pii_anon.swarm_datasets import register_taxonomy, load_jsonl

register_taxonomy("clinical_v1", {
    "patient_name":     "PERSON_NAME",
    "patient_dob":      "DATE_OF_BIRTH",
    "mrn":              "MEDICAL_RECORD_NUMBER",
    "prescribing_doc":  "PERSON_NAME",
    "note_timestamp":   "_IGNORE",   # drop these from training
})
records = load_jsonl("my_clinical_data.jsonl", taxonomy_name="clinical_v1")
```

### Running the retrain

Two equivalent ways to push your data through the training pipeline:

**Option A — file path on the CLI** (shortest path):

```bash
make train-swarm \
    SWARM_DATASETS=/abs/path/to/my_clinical_data.jsonl \
    SWARM_MAX_RECORDS=0 \
    SWARM_KFOLD=5
```

The training script sees the `.jsonl` suffix, dispatches to `load_jsonl`,
and proceeds normally. Combine with built-in datasets to preserve
general-purpose coverage:

```bash
make train-swarm SWARM_DATASETS=pii_anon_eval,/path/to/my_clinical_data.jsonl
```

**Option B — register a named loader** (best when your data needs
bespoke pre-processing):

```python
# my_loaders.py
from pii_anon.swarm_datasets import (
    TrainingRecord, load_jsonl, register_dataset_loader, register_taxonomy,
)

def load_my_clinical(max_records=None):
    register_taxonomy("clinical_v1", {
        "patient_name": "PERSON_NAME",
        "mrn": "MEDICAL_RECORD_NUMBER",
    })
    return load_jsonl(
        "my_clinical_data.jsonl",
        taxonomy_name="clinical_v1",
        max_records=max_records,
        source_label="clinical_v1",
    )

register_dataset_loader("my_clinical", load_my_clinical)
```

Then invoke it by name (assuming `my_loaders` runs at import time via an
entry point, your own init module, or `PYTHONSTARTUP`):

```bash
python -c "import my_loaders"      # pre-register
make train-swarm SWARM_DATASETS=my_clinical,pii_anon_eval
```

### What the retrain learns from your data

All of the following are fit per-retrain from the combined training
pool:

| Artifact | What it captures |
|---|---|
| `ds_params.json` | Per-engine confusion matrices (how often each engine is right on each entity type in *your* data) |
| `temperature.json` | Per-engine logit calibration that minimizes ECE on your mix |
| `informativeness.json` | Per-engine Jaccard weight used in boundary voting |
| `xgboost_model.ubj` | 21-feature meta-learner TP/FP classifier over your distribution |
| `emission_threshold` | F2-optimal cut-off on a held-out split of your data ([swarm_learner.select_f2_threshold](../src/pii_anon/swarm_learner.py)) |
| `manifest.json` | Dataset mix, fold metrics, feature version — audit trail for the run |

Because the meta-learner features are per-span aggregates (not
per-engine slots), your retrain doesn't require the training pool to be
identical to the inference-time engine set. But a retrain that includes
the engines you deploy with is strictly better — Dawid-Skene and
temperature scaling gain explicit per-engine entries.

### Tier 3 annotations (optional)

If your domain data carries the `pii-anon-datasets` v1.3.0+ Tier 3
signals (`behavioral_signals`, `privacy_risk.re_identification_resistance_score`,
`tier3_evaluation.persona_id`), the loader captures them and
[`compute_sample_weights_from_records`](../src/pii_anon/swarm_learner.py)
turns low-RRS records into higher-weight training examples so the
meta-learner sharpens on the hard cases. Without Tier 3 annotations,
the weights fall back to a uniform `default_weight=1.0` and training
proceeds normally — nothing to do.

### End-to-end example

```python
from pii_anon.swarm_datasets import register_taxonomy, load_jsonl
from pii_anon.swarm_learner import (
    XGBoostMetaLearner,
    compute_sample_weights_from_records,
    select_f2_threshold,
)
from pii_anon.swarm import DawidSkeneAggregator

# 1. Load + taxonomize
register_taxonomy("my_domain", {"CLIENT_NAME": "PERSON_NAME"})
records = load_jsonl("my_domain.jsonl", taxonomy_name="my_domain")

# 2. Run engines on each record, collect per-span labels + features
features, labels, per_record_idx = build_training_set(records)   # your pipeline

# 3. Fit Dawid-Skene on engine votes
ds = DawidSkeneAggregator()
ds.train_em(engine_votes_per_span)   # from step 2

# 4. Train meta-learner with RRS-derived sample weights
weights = compute_sample_weights_from_records(
    [records[i] for i in per_record_idx],
    rrs_boost=2.0,
)
meta = XGBoostMetaLearner()
meta.train(features, labels, sample_weights=weights)

# 5. Pick F2-optimal emission threshold on held-out split
probs = meta.predict(held_out_features)
threshold, f2 = select_f2_threshold(probs, held_out_labels)
print(f"Chose emission_threshold={threshold:.3f} (F2={f2:.3f})")
```

The `train_swarm.py` script handles steps 2, 3, and 5 for you — see
[release-guide.md](release-guide.md) for the Make targets.

---

## Combining both workflows

The power-user story: plug your engine in *and* retrain on your domain
so the swarm learns your engine's confusion and your domain's
distribution in one pass.

```python
from pii_anon import PIIOrchestrator
from pii_anon.swarm import SwarmConfig
from pii_anon.swarm_datasets import register_taxonomy

# 1. Register your engine via any of the three paths
orch = PIIOrchestrator(
    token_key="...",
    swarm_config=SwarmConfig(force_include_engines=("my-detector",)),
)
orch.register_engine(MyDetector(enabled=True))

# 2. Register your taxonomy so the loader understands your labels
register_taxonomy("my_domain", {"CLIENT_NAME": "PERSON_NAME"})

# 3. Train (your custom engine will be run during step 2 of the
#    training pipeline and its findings will land in the confusion
#    matrix and feature vectors alongside the built-ins)
```

```bash
make train-swarm \
    SWARM_DATASETS=pii_anon_eval,/path/to/my_domain.jsonl \
    SWARM_MAX_RECORDS=0 SWARM_KFOLD=5
```

The training script picks up your engine from the registry — `regex-oss`
plus any engines added via `register_engine` or entry points become the
engine pool for that run.

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Unknown dataset 'my_data'; skipping` | Name isn't in `DATASET_LOADERS` and doesn't look like a path. Either end it in `.jsonl` / include a `/`, or call `register_dataset_loader` first. |
| Custom engine's findings never appear in swarm output | Layer 2 Jaccard pruner dropped it. Add its ID to `SwarmConfig.force_include_engines`. |
| All my annotations are being dropped | Labels aren't canonical and you didn't pass `taxonomy_name`. Either pre-canonicalize or register a taxonomy. |
| `ValueError: dataset 'X' already registered` | Pass `replace=True` to `register_dataset_loader` if you intend to override a previously registered loader. |
| Retrained model works in training but fails at inference | `FEATURE_VERSION` drifted. Check `~/.pii_anon/swarm/manifest.json` — loader rejects artifacts with a mismatching version. |
| Training warns "behavioral_signals present, RRS missing" for your data | Your JSONL doesn't carry Tier 3 annotations; weights default to uniform. That's fine for domain retraining — Tier 3 is optional. |

---

## Reference

- [engine-plugin-guide.md](engine-plugin-guide.md) — EngineAdapter contract + MoE integration
- [swarm-architecture.md](swarm-architecture.md) — 4-layer pipeline, feature vector, retrain procedure
- [pii-rate-elo.md](pii-rate-elo.md) — evaluation framework
- [evaluate-your-pipeline.md](evaluate-your-pipeline.md) — scoring your extended swarm against the baselines
- `src/pii_anon/swarm_datasets.py` — `load_jsonl`, `register_taxonomy`,
  `register_dataset_loader`, `load_training_data` path dispatch
- `src/pii_anon/swarm.py` — `SwarmConfig.force_include_engines`
