# Formal Analysis: MoE Architecture & Guarantee Proof
## pii-anon Mixture-of-Experts for PII Detection

**Document Version**: 1.0
**Date**: March 2026
**Status**: Research & Analysis Only (No Source Code Modifications)
**Scope**:
- `src/pii_anon/moe.py` - MoE implementation (ExpertRegistry, MoERouter, MoEFusionStrategy)
- `src/pii_anon/fusion.py` - Fusion module and clustering logic
- `src/pii_anon/evaluation/competitor_compare.py` - Integration point (_ensemble_detector at line 743)

---

## Executive Summary

The pii-anon MoE implementation adapts Mixtral 8x7B's sparse expert-routing architecture for PII span detection. While the architecture claims a performance-floor guarantee ("ensemble >= best individual expert"), **a critical gap exists in the implementation**: the MoEFusionStrategy.merge() method filters out findings from non-routed experts, breaking the union-based guarantee stated in the docstrings and architectural design.

**Key Finding**: Lines 388-405 in moe.py implement expert filtering that violates the promised guarantee. This analysis identifies the root cause, proves the theoretical guarantee should hold, and proposes a fix.

---

## A. Mixtral 8x7B Architecture Comparison

### A.1 Mixtral Sparse MoE Architecture

Mixtral 8x7B (Mixture-of-Experts by Mistral AI) implements a sparse gating mechanism in its transformer:

- **Expert Pool**: 8 expert FFN blocks (7B parameters each), only K=2 active per token
- **Gating Mechanism**: Learned per-token router that routes to top-K experts (K=2)
- **Output Aggregation**: Linear combination of top-K expert outputs weighted by softmax(router scores)
- **Sparse Activation**: Only 2 of 8 experts compute for each token, reducing compute vs dense models
- **Performance Guarantee**: Top-K selection ensures best expert always participates, preventing degradation

**Gating Formula**:
```
for each token t:
  scores = router(t) ∈ ℝ^8
  top_k_experts = argsort(scores)[0:2]
  weights = softmax(scores[top_k_experts])
  output = Σ(weights[i] * expert[i](token))
```

### A.2 pii-anon MoE Adaptation

The pii-anon implementation adapts Mixtral's approach for **PII entity span detection** (not vector/embedding space):

**Key Structural Parallelism**:

| Aspect | Mixtral | pii-anon |
|--------|---------|----------|
| **Experts** | 8 FFN blocks | N detection engines (regex, GLiNER, Presidio, etc.) |
| **Routing Granularity** | Per-token | Per-entity-type (EMAIL_ADDRESS, PERSON_NAME, etc.) |
| **Expert Selection** | Top-K by learned router score | Top-K by entity-type strength score |
| **Output Space** | Dense vectors (logits) | Discrete spans (entity_type, span_start, span_end, confidence) |
| **Aggregation** | Weighted sum of FFN outputs | Weighted-average confidence + majority-vote span boundaries |
| **Performance Floor** | Best expert always activated | Best expert always routed (if performance_floor=True) |

**Architectural Implementation in pii-anon**:

1. **ExpertRegistry** (moe.py:59-156): Maintains pool of expert specs with entity-type strengths
   - Each expert declares capabilities: `entity_strengths: dict[str, float]`
   - Higher strength = better performance on that entity type (analogous to Mixtral router learning)

2. **MoERouter** (moe.py:158-290): Sparse routing logic per entity type
   ```python
   def route(entity_type: str) -> list[tuple[str, float]]:
       # Select top-K experts by entity_strengths[entity_type]
       scores = [(expert_id, expert.entity_strengths[entity_type])
                 for expert in registry.available_experts()
                 if entity_type in expert.entity_strengths]
       selected = top_k_select(scores)
       # Performance floor: ensure best expert included
       if best_expert_id not in selected:
           swap_worst_for_best()
       return softmax_normalize(selected)
   ```

3. **MoEFusionStrategy.merge()** (moe.py:348-425): Fusion of findings with MoE weighting
   - Clusters overlapping spans by entity type
   - Routes each entity type to top-K experts
   - Computes weighted-average confidence using only routed experts
   - Resolves span boundaries via weighted voting

### A.3 Key Differences & Appropriateness

**Why Span Detection ≠ Vector Aggregation**:

| Difference | Impact | Handling in pii-anon |
|-----------|--------|-------------------|
| Experts produce **discrete spans** not vectors | Cannot use linear combination | Clustering + weighted voting on confidence & boundaries |
| Multiple experts may find **different spans** for same entity | Span boundary consensus needed | IoU-based clustering + per-boundary weighted vote |
| Entity type is **routing dimension**, not output dimension | Per-entity routing needed, not per-record | MoERouter.route(entity_type) called per cluster |
| Single expert finding should **not be penalized** | Weighted average would dilute single high-confidence detection | No penalty for single-expert findings (full confidence propagates) |
| **Exact matching matters** (evaluation metrics) | Boundary correctness critical | Best-boundary selection via weighted voting prevents false negatives |

**Appropriateness Assessment**: ✅ The adaptation is well-motivated:
- Per-entity-type routing respects PII detector specialization (regex ⊕ structured, NER models ⊕ semantic)
- Span boundary voting preserves true positives by using most-voted boundaries
- Clustering with IoU threshold handles legitimate boundary disagreements
- Span-level aggregation is fundamentally different from Mixtral's token-level vectors, justifying custom fusion logic

---

## B. Guarantee Proof: Ensemble >= Best Individual Expert

### B.1 Formal Theorem Statement

**Theorem (Ensemble Superset Guarantee)**:

For any record R, any single expert E_i in the registry, and a set of findings F produced by E_i on R:

```
entities(E_union) ⊇ entities(E_i)
```

Where:
- `E_union` = ensemble output from MoEFusionStrategy.merge()
- `entities(X)` = {(entity_type, span_start, span_end) | finding ∈ X}
- "⊇" denotes superset or equal (ensemble never **loses** detections)

**Intuition**: Since all experts run independently and their findings are collected into a union before routing decisions, no expert's original finding should be dropped by the merger unless it conflicts with the fusion logic.

### B.2 Proof Sketch (High-level)

**Claim**: Under union-based fusion, ensemble output ⊇ best individual expert output.

**Proof by case analysis on _ensemble_detector flow** (competitor_compare.py:743-871):

1. **Step 1: All engines run independently**
   ```python
   all_findings: list[EngineFinding] = []

   # Regex runs
   all_findings.extend(regex_engine.detect(...))

   # Each competitor runs independently
   for comp_name, comp_detector in competitor_detectors.items():
       all_findings.extend(comp_detector(record))
   ```
   - **Property**: all_findings is the **union** of all expert detections
   - **By construction**: If expert E_i finds entity e at span [s, e), then e ∈ all_findings

2. **Step 2: Clustering groups overlapping spans**
   ```python
   clusters = _cluster_overlapping_spans(findings, iou_threshold=0.5)
   ```
   - **Property**: Clustering groups findings with IoU(span1, span2) >= 0.5
   - **Preservation**: Each expert's finding either:
     - Forms its own cluster (if no overlap with others) → output 1:1
     - Joins an existing cluster → merged with similar findings

3. **Step 3: MoE routing assigns experts per entity type**
   ```python
   routed = self.router.route(entity_type)  # [(expert_id, weight), ...]
   routed_dict = dict(routed)
   ```
   - **Property**: router.route() returns a list of (expert_id, weight) tuples for experts routed to handle entity_type
   - **Performance floor**: If performance_floor=True, best expert always included in routed list

4. **Step 4: Filtering occurs in merge loop**
   ```python
   for item in cluster:
       weight = routed_dict.get(item.engine_id)
       if weight is None:
           continue  # ← FINDING FROM NON-ROUTED EXPERT IS SKIPPED
   ```
   - **CRITICAL ISSUE**: Findings from experts not in routed_dict are silently dropped
   - **Breaks guarantee**: If expert E_i finds an entity but is not routed for that entity_type, the finding is lost

### B.3 Conditions Where the Guarantee FAILS

**Failure Condition 1: Expert not routed for entity type**

```
Scenario: EMAIL_ADDRESS detection
- Experts: regex-oss (strength=0.99), gliner-compatible (strength=0.70)
- top_k = 2
- Both are routed (top-2 by strength)

BUT if gliner-compatible did NOT declare EMAIL_ADDRESS in entity_strengths:
- router.route("EMAIL_ADDRESS") filters: "if entity_type in expert.entity_strengths"
- gliner-compatible is excluded from routing
- If gliner-compatible detects an EMAIL_ADDRESS anyway:
  → Finding is in all_findings
  → But weight = routed_dict.get("gliner-compatible") returns None
  → Line 391-392 skips it: "if weight is None: continue"
  → Finding is DROPPED
  → Guarantee violated!
```

**Failure Condition 2: Top-K truncation without best-expert guarantee**

```
Scenario: US_SSN detection with 6 experts
- top_k = 2
- router selects only top-2 by strength
- If performance_floor=False:
  → Lowest-strength expert (E_6) might not be selected
  → If E_6 finds a US_SSN that E_1 and E_2 miss:
    → Finding is in all_findings
    → But E_6 not in routed_dict
    → Finding is DROPPED
    → Guarantee violated!
```

**Current Code Status**:
- performance_floor=True is the default (moe.py:188, 332)
- ALL expert declarations in build_default_registry() include their handled entity types
- **Therefore**: In practice, the guarantee holds IF experts truthfully declare their capabilities

**Failure Risk**: If an expert is registered with incomplete entity_strengths (e.g., declaring only EMAIL_ADDRESS but also detecting PERSON_NAME), detections outside declared strengths will be silently dropped.

### B.4 Root Cause Analysis: Lines 388-405 (moe.py)

**Critical Code Section**:
```python
# Line 388-405: The problematic merge loop
for item in cluster:
    # Use routed weight if expert is routed, else skip
    weight = routed_dict.get(item.engine_id)
    if weight is None:
        continue  # ← THIS LINE BREAKS THE GUARANTEE

    weighted_sum += item.confidence * weight
    total_weight += weight
    engines.append(item.engine_id)
    # ... boundary voting ...
```

**Why This Breaks the Guarantee**:

1. Line 373: `routed = self.router.route(entity_type)` returns only routed experts
2. Line 379: `routed_dict = dict(routed)` maps expert_id → weight for **routed experts only**
3. Line 390: `weight = routed_dict.get(item.engine_id)` returns None for non-routed experts
4. Line 391-392: Non-routed experts' findings are skipped entirely

**The Assumption** (unstated in docstring):
```
ASSUMPTION: All finding sources (engines) in cluster will be
            in routed_dict (i.e., all engines have declared
            knowledge of this entity_type in registry)
```

This assumption is not enforced or documented.

### B.5 Theoretical Guarantee (Under Corrected Implementation)

**Corrected Theorem**:

If all experts in the registry declare their true capabilities in entity_strengths (or at minimum, any entity_type they can detect), then:

```
entities(E_union) ⊇ entities(E_best)
```

Proof:
1. All experts run independently → all_findings = union of all detections
2. Each finding e ∈ all_findings has a source expert E_i
3. E_i declared e.entity_type in its entity_strengths → E_i is routed for e.entity_type
4. E_i in routed_dict → e will not be skipped in line 391-392
5. e is included in ensemble output
6. Therefore, ensemble ⊇ best individual expert ∎

**Conditions for guarantee to hold**:
- ✅ performance_floor=True (ensures best expert always routed)
- ✅ All experts truthfully declare entity_strengths
- ✅ Experts do not detect outside their declared capabilities
- ✅ iou_threshold appropriately calibrated (0.5 is reasonable)

---

## C. Gap Analysis & Proposed Fix

### C.1 The Gap: Non-Routed Expert Filtering

**Location**: src/pii_anon/moe.py, lines 388-405

**Problem Summary**: Findings from experts not explicitly routed for an entity type are dropped during fusion, violating the union-based guarantee.

**Evidence from Code**:

```python
# moe.py line 372-379: Get routed experts for this entity_type
routed = self.router.route(entity_type)
if not routed:
    # No routing info; skip this cluster
    continue

# Build a dict of routed expert ID -> weight
routed_dict = dict(routed)

# moe.py line 388-405: Process cluster findings
for item in cluster:
    # Use routed weight if expert is routed, else skip
    weight = routed_dict.get(item.engine_id)
    if weight is None:
        continue  # ← LINE 391-392: SILENTLY DROP NON-ROUTED EXPERTS
```

**Impact**:

| Scenario | Impact |
|----------|--------|
| Expert declares entity type in registry | ✅ Routed, finding included |
| Expert detects type NOT in entity_strengths | ❌ Not routed, finding dropped |
| Incomplete registry declarations | ❌ Silent loss of true positives |

**Docstring Mismatch**:

The module docstring (moe.py:1-11) claims:
```
guaranteeing ensemble performance >= best individual expert per entity type
```

But the implementation (line 391-392) can exclude expert findings, breaking this guarantee.

### C.2 Proposed Fix: Minimum Floor Weight for Non-Routed Experts

**Approach**: Instead of skipping non-routed experts, assign them a minimum floor weight that preserves their findings while downweighting them.

**Rationale**:
- Union-based guarantees require all expert findings to participate
- Downweighting (not dropping) unknown experts reflects uncertainty about their reliability
- Minimum floor weight (e.g., 0.05–0.15) prevents single-expert findings from being dropped

**Proposed Implementation**:

```python
class MoEFusionStrategy(FusionStrategy):
    def __init__(
        self,
        registry: ExpertRegistry | None = None,
        top_k: int = 3,
        *,
        iou_threshold: float = 0.5,
        performance_floor: bool = True,
        min_expert_weight: float = 0.15,
        non_routed_floor: float = 0.05,  # ← NEW PARAMETER
    ) -> None:
        # ... existing code ...
        self.non_routed_floor = non_routed_floor  # ← ADD THIS

    def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
        clusters = _cluster_overlapping_spans(findings, iou_threshold=self.iou_threshold)
        merged: list[EnsembleFinding] = []

        for cluster in clusters:
            entity_type = cluster[0].entity_type
            routed = self.router.route(entity_type)

            if not routed:
                # No routing info; fall back to equal weights for all experts
                routed_dict = {}
                num_experts = len({item.engine_id for item in cluster})
                fallback_weight = 1.0 / num_experts if num_experts > 0 else 0.0
                routed_dict = {item.engine_id: fallback_weight for item in cluster}
            else:
                routed_dict = dict(routed)

            weighted_sum = 0.0
            total_weight = 0.0
            engines: list[str] = []
            start_votes: dict[int, float] = {}
            end_votes: dict[int, float] = {}

            for item in cluster:
                # Get weight: routed weight, or floor weight for unrouted
                weight = routed_dict.get(item.engine_id)
                if weight is None:
                    # Non-routed expert: use floor weight instead of skipping
                    weight = self.non_routed_floor

                weighted_sum += item.confidence * weight
                total_weight += weight
                engines.append(item.engine_id)

                s = item.span_start or 0
                e = item.span_end or 0
                start_votes[s] = start_votes.get(s, 0.0) + weight
                end_votes[e] = end_votes.get(e, 0.0) + weight

            if total_weight <= 0:
                continue

            best_span_start = max(start_votes, key=lambda k: start_votes[k])
            best_span_end = max(end_votes, key=lambda k: end_votes[k])

            representative = cluster[0]
            merged.append(
                EnsembleFinding(
                    entity_type=representative.entity_type,
                    confidence=(weighted_sum / total_weight),
                    engines=sorted(set(engines)),
                    field_path=representative.field_path,
                    span_start=best_span_start,
                    span_end=best_span_end,
                    language=representative.language,
                    explanation=f"MoE routing: {', '.join(sorted(set(engines)))}",
                )
            )

        return merged
```

**Key Changes**:

1. **Line 391-392 (OLD)**: `if weight is None: continue` (skip)
2. **Line 391-392 (NEW)**: `if weight is None: weight = self.non_routed_floor` (include with floor)

3. **Add parameter** to MoEFusionStrategy.__init__: `non_routed_floor: float = 0.05`

4. **Fallback for empty routing**: If router returns empty list (no experts routed), assign equal weights to all findings in cluster

**Benefits**:

- ✅ Preserves union-based guarantee (all findings preserved)
- ✅ Downweights unrouted experts (reflects unknown reliability)
- ✅ Configurable floor (can be set per deployment)
- ✅ Graceful degradation if registry is incomplete
- ✅ Still respects expert specialization (routed experts get higher weights)

**Tuning Recommendations**:

```python
# Default (conservative): 0.05 weight for unrouted experts
MoEFusionStrategy(non_routed_floor=0.05)

# Aggressive (trust all experts equally if not routed): 0.20
MoEFusionStrategy(non_routed_floor=0.20)

# Strict (only routed experts matter, with minimum floor): 0.01
MoEFusionStrategy(non_routed_floor=0.01)
```

### C.3 Alternative Approach: Registry Validation

A complementary fix would enforce that all registered experts declare all entity types they can detect:

```python
class ExpertRegistry:
    def validate_expert(self, spec: ExpertSpec) -> list[str]:
        """Validate that expert declarations are complete.

        Returns list of warnings if expert strengths are incomplete.
        """
        warnings = []
        if not spec.entity_strengths:
            warnings.append(f"Expert {spec.expert_id} declares no entity strengths")
        if spec.entity_weaknesses and not spec.entity_strengths:
            warnings.append(f"Expert {spec.expert_id} has weaknesses but no strengths")
        return warnings
```

This would catch incomplete registry definitions at initialization time rather than silently dropping findings at runtime.

### C.4 Impact Assessment

**Current Risk Level**: 🔴 **HIGH**
- Undocumented behavior (docstring claims guarantee, code violates it)
- Silent data loss (no warnings when findings dropped)
- Only mitigated by assumption that experts declare their capabilities truthfully

**After Non-Routed Floor Fix**: 🟢 **LOW**
- Explicit parameter controls behavior
- All findings preserved (though weighted appropriately)
- Registry validation catches incomplete declarations

**Testing Recommendations**:

```python
def test_ensemble_includes_all_expert_findings():
    """Verify union-based guarantee: ensemble ⊇ best individual expert."""
    registry = ExpertRegistry()
    registry.register_expert(ExpertSpec(
        expert_id="detector-a",
        display_name="Detector A",
        entity_strengths={"PERSON_NAME": 0.95, "EMAIL_ADDRESS": 0.90}
    ))
    registry.register_expert(ExpertSpec(
        expert_id="detector-b",
        display_name="Detector B",
        entity_strengths={"PERSON_NAME": 0.80}  # Only PERSON_NAME
    ))

    strategy = MoEFusionStrategy(
        registry=registry,
        top_k=2,
        non_routed_floor=0.05  # Important!
    )

    # Simulate detector-b finding an EMAIL that it didn't declare
    findings = [
        EngineFinding(
            entity_type="EMAIL_ADDRESS",
            confidence=0.99,
            engine_id="detector-b",
            span_start=10, span_end=25,
            field_path=None, language="en",
            explanation="Unexpected EMAIL from detector-b"
        )
    ]

    merged = strategy.merge(findings)

    # With old code: merged == [] (finding dropped)
    # With fixed code: merged == [EnsembleFinding(...)] (finding included with floor weight)
    assert len(merged) == 1, "Union guarantee violated: expert finding was dropped"
```

---

## Summary & Recommendations

### Summary of Findings

| Finding | Severity | Evidence |
|---------|----------|----------|
| Architecture well-adapted for PII spans | ✅ N/A | Per-entity routing, span-aware fusion |
| Guarantee stated in docstrings | ⚠️ MEDIUM | moe.py:1-11, docstring claims >= best expert |
| Implementation breaks guarantee | 🔴 HIGH | Lines 391-392 skip non-routed findings |
| Silent data loss | 🔴 HIGH | No warnings/logs when findings dropped |
| Depends on registry completeness | 🟡 MEDIUM | Only safe if all experts declare all entity types |

### Recommended Actions

**Short-term (Immediate)**:
1. Add test case (test_ensemble_superset_guarantee) verifying union property
2. Add warning log when findings from non-routed experts are skipped
3. Document the assumption that experts must declare their capabilities

**Medium-term (Next Sprint)**:
1. Implement non_routed_floor parameter fix (C.2)
2. Add registry validation (C.3)
3. Update docstrings to clearly state when guarantee holds

**Long-term (Architecture)**:
1. Consider dynamic expert registration (e.g., auto-detect capabilities from sample runs)
2. Add telemetry to track dropped findings by entity type and expert
3. Benchmark MoE strategy against WeightedConsensus to validate performance gains

### Key Takeaway

The pii-anon MoE architecture is theoretically sound and well-adapted for PII detection. However, a gap exists between the **stated guarantee** (ensemble ⊇ best expert) and the **implementation** (silently drops non-routed findings). The proposed non_routed_floor fix restores the guarantee while maintaining expert specialization through weighted routing.

---

## Appendix: Code References

### A.1 Critical Sections

**moe.py:1-11** — Docstring claiming guarantee:
```python
"""Mixtral-inspired Mixture-of-Experts ensemble for PII detection.
...
guaranteeing ensemble performance >= best individual expert per entity type.
"""
```

**moe.py:195-263** — MoERouter.route() implementing top-K with performance floor:
```python
def route(self, entity_type: str) -> list[tuple[str, float]]:
    # Lines 227-229: Only route experts that explicitly declare this entity type
    if entity_type in expert.entity_strengths:
        strength = expert.entity_strengths[entity_type]
        scores.append((expert.expert_id, strength))

    # Lines 241-246: Performance floor ensures best expert included
    if self.performance_floor and selected:
        best_id = scores[0][0]
        if best_id not in {eid for eid, _ in selected}:
            selected[-1] = (best_id, scores[0][1])
```

**moe.py:348-425** — MoEFusionStrategy.merge() with filtering:
```python
def merge(self, findings: list[EngineFinding]) -> list[EnsembleFinding]:
    # Line 373: Route entity type
    routed = self.router.route(entity_type)

    # Line 379: Build dict of routed experts only
    routed_dict = dict(routed)

    # Lines 388-392: THE GAP
    for item in cluster:
        weight = routed_dict.get(item.engine_id)
        if weight is None:
            continue  # ← Non-routed experts' findings are dropped
```

**competitor_compare.py:743-871** — _ensemble_detector function:
```python
def _ensemble_detector(...) -> Callable[[BenchmarkRecord], list[LabelSpan]]:
    # Lines 811-850: All engines run independently
    # Step 1: Regex findings added to all_findings
    # Step 2: Each competitor runs independently
    # all_findings = union of all detections

    # Line 856: Fuse through MoE
    ensemble_findings = moe_fusion.merge(all_findings)
```

### A.2 File Locations

- `/sessions/dreamy-blissful-gauss/mnt/pii-anon-core/pii-anon-code/src/pii_anon/moe.py` — Main MoE implementation
- `/sessions/dreamy-blissful-gauss/mnt/pii-anon-core/pii-anon-code/src/pii_anon/fusion.py` — Fusion strategies & clustering
- `/sessions/dreamy-blissful-gauss/mnt/pii-anon-core/pii-anon-code/src/pii_anon/evaluation/competitor_compare.py` — Integration point

---

**Analysis Completed**: March 2026
**No source code modifications made** — research and analysis only
