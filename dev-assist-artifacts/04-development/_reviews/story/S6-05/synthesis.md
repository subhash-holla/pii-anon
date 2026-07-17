# Story Gate Synthesis — S6-05 (agentic leakage-Sankey + prompt-injection resistance, DC-13)

**Aggregate verdict: APPROVE** (iteration 1). 5/5 reviewers APPROVE; 0 SHOWSTOPPER / 0 CATASTROPHIC / 0 blocking MAJOR. The traceability "MAJOR" is the standing matrix-backfill gap (batched). 2 code-quality MINORs + 2 OBS hardenings polished in-loop post-APPROVE.

## Reviewer set + verdicts

| Reviewer | Verdict | Findings |
|---|---|---|
| security-sast (PRIMARY) | **APPROVE** | 2 OBSERVATION — non-circular leak detection HOLDS under attack (a leaked edge needs a verbatim survival of caller-declared ground-truth; an attacker controlling only the outbound payload can't forge one); whitespace-survivor filter + pre-existing package-root eager-import noted |
| traceability | **APPROVE** | 1 MAJOR (standing matrix backfill, batched) + 1 MINOR + 2 OBS — **ruling on 4-vs-6: the 4-source/6-node reading is acceptable for this gate (DC-13 binds to 4)** |
| requirements-coverage | **APPROVE** | 1 MINOR — **UC-22 explicitly names 6 source channels** (prompt/retrieval/tool/memory/output/trace); `retrieval`+`output` not covered (unscheduled deferral) |
| axiom-compliance | **APPROVE** | 1 OBSERVATION — DATA-branch determinism not contract-pinned (now fixed) |
| code-quality | **APPROVE** | 2 MINOR — `_ZERO_WIDTH` raw U+200B literal; redundant test import |

## The cross-reviewer signal — the 4-vs-6 channel inconsistency (a genuine spec defect, NOT an S6-05 defect)

A real **requirements↔design inconsistency** surfaced: **FR-025 + DC-13 specify 4 agent channels** (prompt/memory/tool-I/O/trace; the frozen `AgentChannel`), while **FR-028 + UC-22 name 6 source channels** (prompt/retrieval/tool/memory/output/trace). S6-05 builds a 4-source/6-node leakage-Sankey, faithful to the **binding** DC-13 + the frozen 4-channel interception (S6-02/FR-025). The traceability reviewer ruled the 4-source reading acceptable for this gate (DC-13 is the binding constraint; the "6" is a requirement-text defect to reconcile at source). The requirements-coverage reviewer flagged that `retrieval`+`output` aren't covered and the deferral is unscheduled.

**Orchestrator decision (documented PO-level call):** accept the 4-source/6-node Sankey for v1 — a 6-source Sankey over a 4-source interception would be partly vacuous (the 2 extra sources have no ledger records). The FR-028/UC-22-vs-FR-025/DC-13 inconsistency is recorded as a **tracked SPEC-RECONCILIATION follow-up** on the Pass-2/user-decision list (widening to 6 source channels requires widening interception — a scope/design decision warranting the PO's input). `_SOURCE_CHANNELS` derives from `AgentChannel`, so a wider taxonomy plugs in with zero module change. This is honest (no claim that 6 sources are covered) + faithful to the binding design.

## In-loop polish (`216a44c`, post-APPROVE)

- `_ZERO_WIDTH = "​"` (explicit escape + U+200B comment; was a raw invisible literal reading as `""`).
- Dropped a redundant in-function `import builtins` in a test.
- `.strip()` on the survivor filter (whitespace-only known-values dropped; defense-in-depth on trusted input).
- **Added `test_ax002_data_path_injection_scoring_is_deterministic`** — pins determinism of the ACTIVE DATA `build_payloads` path (matters for the S7 keystone's reproducibility; `pii_anon_datasets` is installed here).

Suite 3422 pass / 0 fail, cov 87.55% (new module 98%), ruff + both-mypy clean throughout.

## Next

APPROVE → DONE. S6-05's leakage-Sankey + injection-resistance are SDO G5 audit inputs + S7 canonical-run audit evidence. With S6-02 + S5-01, the core agentic+reid G5 audit surface is complete. Deferred: the 4-vs-6 spec reconciliation (Pass-2/PO), the matrix backfill (S6 sprint gate), and a whole-S6 adversarial close (the agentic security surface).
