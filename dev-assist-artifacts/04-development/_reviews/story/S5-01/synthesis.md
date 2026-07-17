# Story Gate Synthesis — S5-01 (ReidAttack protocol + baseline body + attacks import-boundary, DC-09)

**Aggregate verdict: APPROVE** (iteration 1). 5/5 reviewers APPROVE; 0 SHOWSTOPPER / 0 CATASTROPHIC / 0 MAJOR. 3 code-quality MINORs polished in-loop post-APPROVE; the rest are non-blocking OBSERVATIONs deferred to the S5 adversarial close / sprint gate.

## Reviewer set + verdicts

| Reviewer | Verdict | Findings |
|---|---|---|
| security-sast (PRIMARY) | **APPROVE** | 2 OBSERVATION — registry-merge sound (idempotent, same-object, allow-list is the real gate; `json.loads` has no object-hook injection); the `as_outcome()` mutable-dict + `object.__setattr__` frozen-escape are out-of-threat-model |
| axiom-compliance | **APPROVE** | 1 OBSERVATION — `as_outcome()` returns a mutable dict (a caller could strip the caveat from *their own copy*; the source object + exported `AttackResult.outcome` are non-strippable) |
| code-quality | **APPROVE** | 3 MINOR (polished `e569761`) + 2 OBSERVATION |
| requirements-coverage | **APPROVE** | 1 OBSERVATION — NFR-016 is per-surface (re-id side satisfied; DATA pseudonymization side is its own surface) |
| traceability | **APPROVE** | 2 OBSERVATION — matrix backfill (batched); AX-001 asserted in docstring |

## Key confirmations

- **NFR-016 (the satisfied MUST) — genuinely verified, stronger than claimed.** requirements-coverage confirmed the non-strippable anti-anonymity caveat survives the FULL export path: the sandbox copies the outcome via `dict(outcome)` at `sandbox.py:306` WITHOUT stripping keys, so the caveat is on the *exported* `AttackResult.outcome` (A7 asserts `result.outcome["caveat"]`), exactly what NFR-016 demands.
- **FR-011/FR-013 foundation honesty — no over-claim.** All 5 reviewers confirmed the story frames FR-011 (real Tier-3 LLM adversary → S5-02) + FR-013 (real LiRA@128 MIA → S5-03) as a PROTOCOL FOUNDATION (the `@runtime_checkable ReidAttack` + `MiaAttack` seams + a representative deterministic baseline), never as discharged. The `MiaAttack` Protocol is a declared seam (A1 asserts the baseline is NOT a `MiaAttack`). NFR-012/013 power thresholds honestly absent.
- **AX-002 (determinism) — independently verified.** The baseline ranks by a genuine total order `(similarity desc, persona_id asc)`; `reid.py` imports zero `random`/`uuid`/`time`/`secrets`; the sandbox `AttackResult` has no wall-clock field, so A9's equality genuinely excludes timing.
- **The starred registry-merge concern — cleared.** The import-time additive merge of `REID_ATTACK_REGISTRY` into the sandbox's `DEFAULT_ATTACK_REGISTRY` is sound: same mutable dict object, idempotent, disjoint keys (no recon-runner override), allow-list is the real gate (`runner="<dangerous>"` refused). Mutating the `Final[Mapping]`-annotated dict is a typing nuance, not a new smell (the substrate's pre-existing mutability).

## In-loop polish (`e569761`, post-APPROVE, type/comment only)

- **CQ-S5-01-02:** `REID_ATTACK_REGISTRY` retyped `Final[Mapping[str, AttackCallable]]` (imported the sandbox's own runner type → the two registries can't drift).
- **CQ-S5-01-03:** dropped the redundant double `int(candidate_set_size)` casts (the param is a real `int` on both call paths).
- **CQ-S5-01-01:** expanded the `# type: ignore[index]` registry-merge suppressor into a full `# SAFETY:` note documenting the same-object coupling + the `MappingProxyType → extend_default_registry()` escape hatch (sandbox.py stays byte-identical).

Suite 3401 pass / 0 fail, cov 87.48%, ruff + both-mypy clean throughout.

## Deferred (non-blocking)

- The `as_outcome()`/`MappingProxyType` immutability hardening (security + axiom OBSERVATIONs) → probed at the **S5 work-stream adversarial close** (the standing catch-net for this attack-surface work).
- `traceability-matrix.md` S5-01 forward Story/Test row + UC-09 link → batched to the S5 sprint gate (standing deferral).

## Next

APPROVE → DONE. S5-01 unblocks S5-02 (Tier-3, reuses `ReidAttack` + scorer) + S5-03 (MIA, reuses the `MiaAttack` seam + the boundary test). Feeds SDO G5 audit half + Paper 1 Tier-2/3.
