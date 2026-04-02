# Design Critique and Recommendation

## 1. Proposals Evaluated

- **Proposal A (Simplicity-Focused):** Implements the four-layer pipeline (regex fast-pass, Dawid-Skene, XGBoost meta-learner, post-processing) in 5 new files totaling approximately 1,380 lines of code. Pure-Python Dawid-Skene implementation with zero required runtime dependencies. XGBoost as the optional meta-learner operating at span level.

- **Proposal B (Research-Aligned Scalable):** Implements the same four-layer pipeline across a 15-file `swarm/` package with significantly more infrastructure: BIO tokenization, crowd-kit Dawid-Skene at token level, HMM BIO correction, CRF sequence meta-learner, Platt scaling with informativeness scoring, selective engine activation with greedy set-cover, and a formal dataset registry. Six optional ML dependencies (crowd-kit, sklearn-crfsuite, scikit-learn, hmmlearn, numpy, HuggingFace datasets).

## 2. Scoring Matrix

| Criterion | Weight | Proposal A | Proposal B |
|-----------|--------|-----------|-----------|
| Requirement Coverage | Critical | 8/10 | 9/10 |
| Simplicity | High | 9/10 | 4/10 |
| Correctness Likelihood | High | 7/10 | 7/10 |
| Latency Feasibility | High | 9/10 | 6/10 |
| Integration Risk | High | 9/10 | 5/10 |
| Dependency Risk | Medium | 9/10 | 4/10 |
| Training Pipeline Quality | Medium | 7/10 | 8/10 |
| **Weighted Total** | | **8.3** | **6.0** |

Weighted total formula: Critical=3x, High=2x, Medium=1x.
- A: (8x3 + 9x2 + 7x2 + 9x2 + 9x2 + 9x1 + 7x1) / 13 = 108/13 = 8.3
- B: (9x3 + 4x2 + 7x2 + 6x2 + 5x2 + 4x1 + 8x1) / 13 = 78/13 = 6.0

## 3. Detailed Analysis

### 3.1 Requirement Coverage

**Proposal A: 8/10.** Covers all 16 high-priority functional requirements explicitly. The requirement coverage matrix is thorough and maps each FR to a specific design element. However, FR-002 (selective engine activation) is addressed only implicitly -- the fast-pass partitions findings, but there is no true coverage-gap analysis or greedy engine selection. The proposal relies on the existing MoERouter, which routes per entity type but does not prune redundant engines (e.g., it would still run both spaCy and Stanza despite their 0.891 Jaccard overlap). Medium-priority requirements and NFRs are less explicitly addressed.

**Proposal B: 9/10.** Covers all 16 high-priority FRs plus explicitly addresses medium-priority FRs and all 15 NFRs. FR-002 is addressed with a proper SelectiveActivator that performs greedy set-cover with Jaccard redundancy pruning -- directly exploiting the known spaCy/Stanza redundancy. The requirement coverage matrix includes NFR-001 (latency), NFR-004/005/006 (accuracy targets), NFR-011 (optional deps), and NFR-013/014 (backward compatibility). One point deducted because the design is so complex that some requirements may be "technically covered" but practically difficult to verify in integration.

**Winner: Proposal B**, for explicitly addressing selective activation and covering NFRs.

### 3.2 Simplicity

**Proposal A: 9/10.** Five new files totaling roughly 1,380 lines (plus one JSON config). The core logic fits in a single 400-line file. The SpanCandidate dataclass is the only new data model. No new package structure -- files go in the existing `src/pii_anon/` directory. The dependency graph is trivially flat: SwarmFusionStrategy calls three helpers. Cognitive complexity is low because span-level processing matches the existing codebase's mental model (the existing fusion.py already operates on spans). A new developer can read swarm.py top to bottom in 20 minutes and understand the entire pipeline. The pure-Python DS implementation avoids the need to understand crowd-kit internals.

**Proposal B: 4/10.** Fifteen new files organized in a `swarm/` package. Introduces five new data models (BIOTag, TokenizedText, EngineAnnotation, AggregatedToken, SwarmConfig, TrainingManifest, PlattParams). The BIO tokenization layer is a fundamental conceptual addition that requires developers to understand span-to-BIO and BIO-to-span conversions, along with the alignment pitfalls involved. The three-stage Layer 3 (DS then HMM then CRF) is a deep processing pipeline within a pipeline. The error hierarchy introduces six new exception classes. The DatasetSpec/DatasetRegistry adds structured infrastructure for something that could be a 50-line loader script. The InformativenessScorer adds another weighting dimension on top of calibration. Onboarding a new developer would require understanding BIO conventions, crowd-kit's DataFrame API, HMM Viterbi decoding, and CRF feature engineering.

**Winner: Proposal A**, by a wide margin. The simplicity advantage is the single most consequential difference between these proposals.

### 3.3 Correctness Likelihood (Achieving F1 >= 0.85)

**Proposal A: 7/10.** The span-level approach is simpler but has a correctness concern: Dawid-Skene at span level requires clustering overlapping spans into candidates before aggregation. The quality of this clustering directly determines the quality of the DS input. If two engines disagree on span boundaries (e.g., "John" vs "John Smith"), the IoU clustering and weighted boundary voting must correctly merge them. The existing `_cluster_overlapping_spans()` function handles this, but it was designed for simple voting, not for feeding into a Bayesian model. XGBoost as a span-level classifier is a reasonable choice -- it can achieve high accuracy on tabular feature classification. The 20-feature design is solid. The logistic fallback (no XGBoost) is simple but may not achieve F1 >= 0.85 alone. The main concern is that without BIO-level modeling, boundary precision may suffer for multi-token entities like addresses and names, and the corroboration filter at Layer 4 with a meta_score override threshold of 0.85 may be too aggressive for singleton detections.

**Proposal B: 7/10.** The token-level BIO approach is theoretically more principled for handling boundary disagreements. However, this comes with a significant practical risk: whitespace tokenization may misalign with engine span boundaries. The proposal acknowledges this risk but the mitigation (midpoint-based token assignment) is fragile. A span like "Dr. Smith" could tokenize as ["Dr.", "Smith"] but an engine might return span [0, 9] which covers both tokens. The BIO conversion must handle partial overlaps correctly, and errors here propagate through the entire DS/HMM/CRF pipeline. Additionally, crowd-kit's DawidSkene operates on pandas DataFrames, which introduces vectorization overhead and may behave differently on small token sets (a single short text may have only 10-20 tokens). The CRF meta-learner is theoretically better at sequence-level consistency than XGBoost, but the training data must be sufficient to learn meaningful CRF transition features. Both proposals are roughly equally likely to hit F1 >= 0.85; neither has a clear empirical advantage without benchmarking.

**Winner: Tie.** Different risk profiles but similar overall likelihood. A's risk is boundary precision; B's risk is tokenization alignment and pipeline complexity introducing subtle bugs.

### 3.4 Latency Feasibility (<= 200ms)

**Proposal A: 9/10.** The latency analysis is convincing. Layer 1 is simple iteration (microseconds). Temperature scaling is a per-finding sigmoid (microseconds). DS inference is a single forward pass with pre-trained parameters -- O(spans * engines * types), which for a typical record (20 spans, 6 engines, 20 types) is 2,400 multiplications. XGBoost predict_proba on 20 features for 20 spans takes roughly 1ms. Post-processing is simple iteration. Total pipeline overhead: 5-10ms. This leaves the full 190ms budget for engine execution, which is the actual bottleneck. The proposal does not introduce any new engine invocations at inference time (no selective activation Layer 2 that triggers engine runs), so the orchestrator's existing engine execution dominates.

**Proposal B: 6/10.** Three latency concerns. First, the BIO tokenization and DataFrame construction for crowd-kit adds overhead -- building a pandas DataFrame with rows = (tokens * engines) for a 50-token text with 4 engines produces 200 rows. crowd-kit then runs EM even with warm-start, which involves matrix operations per iteration. Second, the CRF inference involves feature extraction per token (dictionary construction with 20+ keys per token) plus CRF Viterbi decoding. sklearn-crfsuite's inference is typically 2-5ms for a 50-token sequence. Third, and most critically, Layer 2 (SelectiveActivator) may trigger additional engine invocations at inference time. If regex found a low-confidence PERSON_NAME and the activator determines GLiNER and Presidio should run, those engines must execute within the 200ms budget. The existing orchestrator runs all engines upfront, but Proposal B's architecture implies selective post-routing execution. If the engines are already run by the orchestrator, the selective activation is redundant; if they are not, the additional engine calls risk blowing the latency budget. The proposal sets engine_timeout_ms to 150ms, which suggests engines run within the pipeline, leaving only 50ms for all other processing -- tight.

**Winner: Proposal A.** The latency story is straightforward and the overhead is minimal. Proposal B introduces multiple latency-uncertain components.

### 3.5 Integration Risk

**Proposal A: 9/10.** The integration surface is minimal. One new case in `build_fusion()`, one new FusionStrategy subclass. The `merge()` method signature is unchanged. The SwarmFusionStrategy loads its artifacts at construction and operates purely on the `list[EngineFinding]` input, producing `list[EnsembleFinding]` output. No modifications to the orchestrator are required beyond the factory addition. The existing CalibrationStore is extended (not replaced) with temperature parameters in the metadata dict. No new import-time side effects. The lazy import of `swarm` in the `build_fusion()` factory means the module is only loaded when mode="swarm" is selected.

**Proposal B: 5/10.** The integration is more invasive. The SwarmFusionStrategy needs access to the original text (for BIO tokenization) and the engine/expert registries (for selective activation). This breaks the FusionStrategy abstraction: existing strategies only receive `list[EngineFinding]` and return `list[EnsembleFinding]`. Proposal B acknowledges this with a NOTE in the code: "Unlike other FusionStrategies, this one needs access to the original text and the engine registry." This requires modifying the orchestrator to pass additional context to the fusion strategy constructor, which is a leaky abstraction. The `swarm/` package adds 15 files, a package init, and taxonomy map JSON files -- a significant structural addition. Three existing files need modification (fusion.py, types.py, orchestrator.py) vs. only fusion.py for Proposal A. The joblib serialization of the CRF model introduces a cross-Python-version portability concern that JSON-based serialization avoids.

**Winner: Proposal A.** Clean integration vs. abstraction-breaking changes to the orchestrator.

### 3.6 Dependency Risk

**Proposal A: 9/10.** Three optional dependencies, all well-established: XGBoost (3.6K GitHub stars, active development, cross-platform), scikit-learn (ubiquitous), numpy (ubiquitous). All are in the `[swarm-ml]` optional extra. The runtime path without ML deps is functional (Layers 1, 2, 4 work; Layer 3 falls back to logistic). No pandas dependency. No crowd-sourcing-niche libraries.

**Proposal B: 4/10.** Six optional dependencies with varying maturity levels:
- **crowd-kit** (Toloka): Niche library from a specific crowdsourcing platform. 290 GitHub stars. Active but small community. API changes between versions are possible. Depends on pandas, which is a heavy transitive dependency for a PII detection library.
- **sklearn-crfsuite**: Last PyPI release was in 2019 (version 0.3.6). The project appears unmaintained. The underlying crfsuite C library is stable, but the Python wrapper has known compatibility issues with newer Python versions. This is a significant risk for a project targeting Python 3.10+.
- **hmmlearn**: Moderately maintained (last release 2023). Reasonable choice for HMM but adds another C extension dependency with platform-specific build requirements.
- **HuggingFace datasets**: Well-maintained but large dependency tree (adds pyarrow, multiprocess, dill, fsspec). Only needed for training, but the swarm-train extra pulls in a significant dependency chain.
- **scipy**: Already a transitive dep of scikit-learn, so no incremental risk.

The sklearn-crfsuite maintenance status is the most concerning risk. If this library stops working on Python 3.12+, the entire CRF meta-learner becomes unusable, and the fallback (returning DS consensus labels) loses a critical pipeline stage.

**Winner: Proposal A.** Far fewer dependencies, all well-maintained and widely used.

### 3.7 Training Pipeline Quality

**Proposal A: 7/10.** The training pipeline is a single script (`swarm_train.py`, ~300 lines) with a clear 9-step flow: load, run engines, align, train DS, train temperature, extract features, train XGBoost, evaluate, serialize. The dataset integration table covers all 6 datasets with format descriptions. The taxonomy mapping is a single JSON file with all datasets. Temperature scaling uses sigmoid(logit/T) which is the standard approach. The training-time EM for Dawid-Skene is described algorithmically. However, the pipeline lacks some robustness features: no versioned training manifest, no explicit cross-validation strategy, no handling of dataset quality weighting (mentioned only as a risk mitigation), and no formal dataset registry for future extensibility.

**Proposal B: 8/10.** The training pipeline is more structured: TrainingOrchestrator with explicit stage decomposition, DatasetRegistry with typed DatasetSpec entries, separate taxonomy maps per dataset, a TrainingManifest that records package versions and training metadata. The pipeline includes Platt scaling as a separate calibration step (vs. temperature scaling integrated into training), cross-domain validation split, and explicit dominance verification as a gating step. The DatasetSpec approach with HuggingFace paths and split maps is well-designed for reproducibility. The 12-step training flow is more comprehensive but also more fragile (more steps that can fail). The BenchmarkEvaluator and DominanceVerifier as separate classes promote reuse.

**Winner: Proposal B**, for better training infrastructure, manifest tracking, and formal dominance verification. But the margin is moderate.

## 4. Strengths and Weaknesses

### Proposal A

**Strengths:**
1. **Minimal footprint.** Five files, approximately 1,380 lines, zero new runtime dependencies. This is achievable in a 1-2 week sprint by a single developer. The risk of schedule overrun is low.
2. **Clean integration.** The FusionStrategy abstraction is preserved. No orchestrator modifications beyond the factory. Drop-in replacement for existing fusion modes.
3. **Latency headroom.** The pipeline adds only 5-10ms of overhead, leaving the full engine execution budget intact. No inference-time EM, no DataFrame construction, no Viterbi decoding.

**Weaknesses:**
1. **Span-level DS is a simplification.** Operating Dawid-Skene at span level (after IoU clustering) loses information about boundary disagreements. When Engine A says [10,18] and Engine B says [10,20], the clustering merges them into one candidate, but the DS model does not get to reason about which boundary is correct per-token.
2. **No explicit selective activation.** FR-002 is addressed only through fast-pass partitioning and the existing MoERouter. The known spaCy/Stanza redundancy (Jaccard 0.891) is not explicitly exploited -- both may still run unnecessarily.
3. **Training pipeline lacks infrastructure.** No DatasetSpec registry, no training manifest with version tracking, no explicit dominance verification class. The training script is functional but not production-hardened for CI gating.

**Unique ideas worth preserving:**
- Pure-Python DS implementation (~120 lines) with no crowd-kit dependency. This avoids pandas overhead and version-coupling risk.
- Logistic fallback function for the meta-learner when XGBoost is unavailable. The formula `x = 2.0 * ds_confidence + 0.5 * min(corroboration, 4) - 2.0` is a practical heuristic.
- Single-file core module (`swarm.py`) with DawidSkeneAggregator, TemperatureScaler, and SwarmFusionStrategy co-located for minimal import complexity.

### Proposal B

**Strengths:**
1. **Principled token-level aggregation.** BIO tokenization followed by DS at token level followed by HMM correction is the academically correct approach. It handles boundary disagreements at the granularity where they actually occur.
2. **Selective engine activation.** The SelectiveActivator with greedy set-cover and Jaccard pruning directly addresses the spaCy/Stanza redundancy problem, potentially saving 50-80ms of engine execution time per record.
3. **Comprehensive training infrastructure.** DatasetRegistry, TrainingManifest, BenchmarkEvaluator, and DominanceVerifier are reusable components that support CI integration and reproducibility.

**Weaknesses:**
1. **Excessive complexity.** 15 files, 5+ new data models, 6 new exception classes, BIO tokenization layer, HMM corrector, CRF meta-learner. The cognitive overhead for a developer maintaining this system is high. The DS-then-HMM-then-CRF three-stage aggregation within Layer 3 is a pipeline-within-a-pipeline.
2. **Dependency on sklearn-crfsuite (unmaintained).** The CRF meta-learner depends on a library last released in 2019. This is a ticking time bomb for Python version upgrades. The fallback (DS consensus labels) loses the meta-learner entirely, which is a larger degradation than Proposal A's logistic fallback.
3. **Integration breaks the FusionStrategy abstraction.** Requiring text and registry injection into the fusion strategy constructor modifies the orchestrator and creates a leaky abstraction. This makes the fusion strategy non-interchangeable with existing strategies at the factory level.

**Unique ideas worth preserving:**
- SelectiveActivator with greedy set-cover and Jaccard redundancy pruning. This is the correct approach to FR-002 and would save real latency.
- InformativenessScorer that detects and downweights engines with fixed/uninformative confidence distributions. This directly addresses the known problem of 3 of 6 engines using fixed confidence scores.
- TrainingManifest with package version recording for reproducibility.
- Platt scaling (A*conf + B) vs. temperature scaling (logit/T). Both are calibration approaches, but Platt's 2-parameter model is slightly more flexible.

## 5. Recommendation

### Decision: Hybrid -- Proposal A as the base, with targeted elements from Proposal B

### Rationale

Proposal A wins on five of seven criteria, including the three high-weight criteria of Simplicity, Latency Feasibility, and Integration Risk. In a library codebase where maintainability and developer onboarding are critical, Proposal A's minimal footprint is a decisive advantage. The pure-Python DS implementation avoids the most problematic dependency (crowd-kit/pandas) while achieving equivalent inference-time behavior (pre-trained parameters, no EM at inference).

However, Proposal A has real gaps that Proposal B addresses well. The three most impactful elements to adopt from Proposal B are:

1. **Selective engine activation** -- the greedy set-cover with Jaccard pruning is the correct solution to FR-002 and directly exploits the known spaCy/Stanza redundancy, saving 50-80ms of engine execution.
2. **InformativenessScorer** -- downweighting engines with fixed confidence distributions addresses a known data quality problem (3 of 6 engines use fixed scores).
3. **Training manifest and dominance verification** -- production-grade training infrastructure for CI gating.

The hybrid preserves Proposal A's architecture (span-level processing, pure-Python DS, XGBoost meta-learner, minimal files) while incorporating specific, well-scoped additions from Proposal B.

### Element Sources

| Design Element | Source | Rationale |
|---------------|--------|-----------|
| Overall architecture (4-layer single merge() call) | Proposal A | Preserves FusionStrategy abstraction, minimal integration risk |
| Pure-Python Dawid-Skene at span level | Proposal A | Zero-dependency inference, avoids crowd-kit/pandas overhead |
| XGBoost span-level meta-learner | Proposal A | Simpler than CRF, avoids unmaintained sklearn-crfsuite; sequence validity addressed by corroboration filter |
| Temperature scaling (logit/T) | Proposal A | Single-parameter calibration is sufficient; equivalent to Platt when B=0 |
| Logistic fallback for meta-learner | Proposal A | Practical heuristic when XGBoost unavailable |
| Selective engine activation | **Proposal B** | Greedy set-cover with Jaccard pruning for FR-002; add as a pre-processing step in the orchestrator or within the strategy, but keep within the FusionStrategy contract by accepting pre-computed engine findings |
| Informativeness scoring | **Proposal B** | Detect and downweight fixed-confidence engines; add as 2-3 functions in swarm.py, not a separate class |
| Training manifest with version tracking | **Proposal B** | Add TrainingManifest dataclass to swarm_train.py; serialize as part of artifacts |
| Dominance verification | **Proposal B** | Add dominance check as a final step in swarm_train.py; fail training if swarm < best engine per type |
| Dataset registry (DatasetSpec) | **Proposal B** (simplified) | Adopt the DatasetSpec pattern but implement as a dict of specs in swarm_train.py, not a separate class/file |
| Error handling (graceful degradation) | Proposal A | Layer-by-layer fallback without new exception hierarchy |
| File structure (flat, minimal) | Proposal A | 6-7 files in src/pii_anon/ (add one file for selective activation if needed), not a 15-file package |
| Configuration schema (SwarmConfig JSON) | Proposal A | Simpler config with fewer knobs; add coverage_similarity_threshold and max_engines from B |
| Post-processing (checksum + corroboration) | Proposal A | Identical approach in both proposals |
| Taxonomy mapping | Proposal A | Single JSON file with all datasets; simpler than per-dataset files |

### Implementation Note on Selective Activation

The selective activation from Proposal B requires careful integration to avoid breaking the FusionStrategy abstraction. The recommended approach is:

1. The orchestrator already runs all engines before calling `fusion.merge()`. Keep this behavior.
2. Add a `_prune_redundant_findings()` step at the beginning of `SwarmFusionStrategy.merge()` that filters out findings from engines deemed redundant for the detected entity types. This operates on the already-collected findings rather than controlling engine execution.
3. If future latency optimization requires skipping engine execution entirely, modify the orchestrator's engine dispatch (separate from the fusion strategy) to consult the expert registry's Jaccard similarity data. This is a future optimization, not a launch requirement.

This preserves the `merge(findings) -> ensemble_findings` contract while still reducing noise from redundant engines.

### Requirement Gap Analysis

| Req ID | Status in Recommended Design | Notes |
|--------|------------------------------|-------|
| FR-001 | Fully addressed | Regex fast-pass from Proposal A |
| FR-002 | Fully addressed | Selective activation logic from Proposal B, implemented as finding pruning within merge() |
| FR-004 | Fully addressed | Pure-Python DS from Proposal A |
| FR-005 | Fully addressed | Existing _cluster_overlapping_spans() + weighted boundary voting |
| FR-006 | Fully addressed | XGBoost meta-learner from Proposal A with logistic fallback |
| FR-007 | Fully addressed | Corroboration filter in Layer 4 |
| FR-008 | Fully addressed | Checksum/format validation using existing validators.py |
| FR-009 | Fully addressed | Temperature scaling from Proposal A |
| FR-010 | Fully addressed | Temperature-scaled confidences + informativeness scoring from B |
| FR-011 | Fully addressed | Multi-dataset training with 6 datasets |
| FR-012 | Fully addressed | Taxonomy mapping via JSON config |
| FR-013 | Fully addressed | CLI training script producing serialized artifacts |
| FR-014 | Fully addressed | Per-entity F1 with bootstrap CIs in training pipeline |
| FR-015 | Fully addressed | Dominance verification (adopted from B) gates training |
| FR-016 | Fully addressed | mode="swarm" via build_fusion() extension |
| NFR-001 | Fully addressed | Minimal pipeline overhead (5-10ms) + redundant engine pruning saves 50-80ms |
| NFR-004 | Addressed (needs validation) | DS + XGBoost + corroboration targets F1 >= 0.85; empirical validation required |
| NFR-005 | Fully addressed | Corroboration filter + checksum validation target P >= 0.80 |
| NFR-006 | Fully addressed | Fast-pass preserves regex recall; DS considers all engine votes |
| NFR-011 | Fully addressed | XGBoost, sklearn, numpy all optional; core pipeline works without |
| NFR-013 | Fully addressed | mode="swarm" is additive; existing modes untouched |
| NFR-014 | Fully addressed | ProcessingProfileSpec.mode string; run() signature unchanged |

## 6. Risk Mitigation Recommendations

1. **Boundary precision risk (span-level DS).** The hybrid retains span-level DS, which may produce less precise boundaries than token-level. Mitigate by: (a) adding unit tests with known boundary disagreement cases (e.g., "John" vs "John Smith"), (b) tracking boundary-level precision as a separate metric during training evaluation, (c) if boundary precision is empirically poor (< 0.80), consider adding a lightweight boundary refinement step after DS (not full BIO tokenization, but a targeted boundary expansion/contraction heuristic).

2. **XGBoost dependency availability.** The logistic fallback must be validated to achieve at least F1 >= 0.80 (close enough to the target that the degradation is acceptable). If the fallback cannot reach F1 >= 0.80, consider bundling a small decision tree model serialized as JSON (no XGBoost needed to load).

3. **Stale training artifacts.** Add a version check in SwarmFusionStrategy.__init__() that compares the artifact schema_version and training timestamp against the installed pii-anon version. Log a warning if artifacts are older than 90 days or were trained with a different library version.

4. **Selective activation accuracy.** The greedy set-cover algorithm should be validated against the known engine strength matrix. Add a test that verifies: given the current 6 engines and their known Jaccard similarities, the activator correctly prunes spaCy when Stanza is already selected (or vice versa).

5. **Training pipeline robustness.** The training script should handle dataset unavailability gracefully (skip unavailable datasets with a warning, do not fail). The i2b2 dataset requires a Data Use Agreement and may not be available in all environments. The training pipeline should produce valid artifacts even if only 3-4 of 6 datasets are available.

6. **Empirical validation before committing.** Before finalizing the architecture, implement a minimal prototype of the four-layer pipeline and run it on the pii-anon-eval-data test split. If span-level DS + XGBoost cannot reach F1 >= 0.83 on this prototype, escalate to the token-level BIO approach from Proposal B. This validation should happen in the first sprint, not after full implementation.

## 7. Recommended Next Steps

1. **Prototype validation (Week 1).** Build a minimal prototype of the four-layer pipeline using Proposal A's architecture. Run on pii-anon-eval-data. Measure F1, precision, recall, and latency. This gates the final architecture choice.

2. **Selective activation design (Week 1).** Detail the finding-pruning approach for FR-002 within the merge() contract. Decide whether to prune at the finding level (simpler) or to modify the orchestrator's engine dispatch (more impactful for latency but more invasive).

3. **Training pipeline (Week 2).** Implement the training script with dataset loading, taxonomy mapping, DS EM training, temperature scaling, and XGBoost training. Include the TrainingManifest and dominance verification from Proposal B.

4. **Informativeness scoring integration (Week 2).** Implement the informativeness scorer from Proposal B as utility functions within the training module. Use it to assign fusion weights that downweight the three fixed-confidence engines.

5. **Medium/low priority requirements.** Incorporate FR-003 (engine diversity score) as a diagnostic metric in the training evaluation. Add FR-017 (configurable thresholds) through the SwarmConfig JSON schema. Add FR-018 (audit trail/explanation) by enriching the EnsembleFinding.explanation field with layer-by-layer provenance.
