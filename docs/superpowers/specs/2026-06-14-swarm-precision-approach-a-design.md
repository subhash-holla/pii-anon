> ⚠️ **SUPERSEDED (2026-06-14) by [2026-06-14-swarm-precision-v2-design.md](2026-06-14-swarm-precision-v2-design.md).** An SME panel (unanimous MAJOR_REVISION) + a measured A/B re-grounding found this version rests on fabricated mechanisms (EMAIL≠MAC, single-engine override unreachable), targets the wrong fusion module (the benchmark runs MoEFusionStrategy, not SwarmFusionStrategy), and mis-frames a precision problem as a recall regression. Kept for history only.

# Swarm Precision — Approach A: Authoritative Regex + Tightened Gates

- **Status:** Approved design (pending spec review) — 2026-06-14
- **Scope:** Slice 1 of the best-in-class program **A → B → C** (A = inference-time; B = train the meta-learner; C = latency/routing).
- **Priority target:** privacy-first F2 > balanced F1 > latency. Approach A lifts **both** F2 and F1.
- **Blast radius:** `src/pii_anon/swarm.py`, `SwarmConfig`, and `src/pii_anon/regex/patterns.py` (Unit 5, shared with the core detector). Isolated from the user-WIP `orchestrator.py`.

## 1. Context & verified root cause

On the English benchmark (~148,994 records) the swarm detector sits **3rd of 5** on balanced F1 (0.610) behind core pii-anon (0.756) and gliner (0.764), with precision **0.486** / recall **0.818** and **~3.06M spurious false positives**. On the recall-weighted F2 tournament it is effectively co-leader (F2 0.885). The gap between those two views is precision.

A read-only diagnosis (4-agent fan-out, 2026-06-14) found **one root cause expressed two ways**: the swarm trusts multi-engine *consensus* over checksum/regex *certainty*, and the arbiter meant to referee that — the 21-feature XGBoost meta-learner — **is dormant**.

**Verified:** `~/.pii_anon/swarm/` contains `ds_params.json`, `temperature.json`, `informativeness.json` but **no `xgboost_model.ubj`** (none anywhere in the repo). `swarm.py:621-626` therefore loads no meta-learner and logs "Meta-learner not available; using logistic fallback"; scoring is the hand-tuned linear `_logistic_fallback_score` (`swarm.py:535-549`): `meta = 2.0·ds_conf + 0.5·corroboration + 0.8·regex + 0.3·structured − 2.0`. `train_swarm.py:594-598` confirms the training script never produces the model ("logistic fallback until trained on pipeline output"). Reviving the brain is **Approach B**; Approach A makes the *deterministic* path correct regardless.

The two symptoms:

- **Precision collapse (~3.06M FPs).** PERSON_NAME + PHONE_NUMBER = 1.71M FPs (56%), passing because a single high-confidence NER engine bypasses `corroboration_min=2` via the `corroboration_override_threshold=0.85` (`swarm.py:686-693`). EMAIL = 475K FPs (permissive regex matching MAC addresses, `regex/patterns.py`). SSN/BANK/IP = 535K FPs (STRUCTURED types are *exempt* from corroboration and ride the lax `emission_threshold=0.50`).
- **Structured-type regression** (EMAIL 0.635, PHONE 0.567, US_SSN 0.580, IP 0.631 vs single-detector ~0.93–1.0). A checksum-perfect regex span is (a) re-typed by weighted max-vote in `_build_candidate` (`swarm.py:720-755`), (b) re-typed again by Dawid-Skene when `ds_conf>0.8` (`swarm.py:774-779`), (c) marked `is_structured=0` from the *mutated* type in feature extraction (`swarm_learner.py:142`), and (d) dropped by the corroboration gate when regex is the only voter. Pure-checksum scrubadub beats the swarm on SSN (0.93 vs 0.58).

## 2. Goals / non-goals

**Goals**
- Make a high-confidence regex/validator detection of a STRUCTURED type authoritative: locked type + boundaries, no NER/DS override, no corroboration requirement.
- Close the FP corridor for NER-driven candidates (raise emission floor, remove the single-engine override bypass, per-type thresholds for the noisy types).
- Tighten the permissive EMAIL regex (MAC ≠ email) — **Unit 5**, shared with core; verify the core path does not regress.
- Keep every change **deterministic** and **config-tunable** with conservative defaults.

**Non-goals (this slice)**
- Training the XGBoost meta-learner (Approach B).
- Engine routing / latency / fast-path short-circuit (Approach C).
- Any change to `orchestrator.py` (user-WIP) or the SDO `competitive_supremacy.py` gate.

## 3. Design — five units

A single helper centralizes "what counts as authoritative" so the three layers stay consistent and unit-testable without running the full pipeline:

```
_regex_authority(engine_findings: dict[str, Finding], cfg: SwarmConfig)
    -> RegexAuthority | None   # (entity_type, start, end) when regex-oss fires a
                               # STRUCTURED type at confidence >= regex_authority_threshold
```

| # | Unit | Location | Behaviour |
|---|---|---|---|
| 1 | **Type + boundary lock** | `_build_candidate` (swarm.py:720-755) | If `_regex_authority(...)` is not None, set `entity_type`/`start`/`end` from it and skip the weighted max-vote for type and boundaries. |
| 2 | **Suppress DS re-typing** | Layer 3c (swarm.py:774-779) | If authoritative, keep `ds_confidence` as a *signal* but do **not** apply the `ds_type` override. |
| 3 | **Corroboration bypass** | Layer 4 (swarm.py:686-693) | An authoritative candidate emits regardless of `corroboration_count`. |
| 4 | **Close the FP corridor** | Layer 4 + `SwarmConfig` | (a) raise default `emission_threshold` 0.50→0.60; (b) gate the single-engine corroboration bypass behind `allow_single_engine_override` (default **False** ⇒ SEMANTIC types always need ≥2 engines; the raised `corroboration_override_threshold=0.95` only applies if it is re-enabled); (c) consult `per_type_emission_thresholds` (e.g. PERSON_NAME/PHONE_NUMBER = 0.65) before emitting NER-driven candidates. |
| 5 | **Tighten EMAIL regex** | `regex/patterns.py` | EMAIL pattern must not match MAC-address-like strings; real emails still match. Shared with core detector. |

**Interaction with Unit 4 thresholds:** authoritative regex (Units 1-3) is evaluated *before* the Unit 4 NER gates, so tightening the corridor never suppresses a trusted regex/validator span — it only constrains NER-driven candidates.

## 4. Config surface (`SwarmConfig`, all tunable, conservative defaults)

| Field | Default | Meaning |
|---|---|---|
| `regex_authority_threshold` | `0.90` | Min regex-oss confidence for a STRUCTURED span to be authoritative. |
| `emission_threshold` | `0.50` → **`0.60`** | Min meta_score for an NER-driven candidate to emit. |
| `corroboration_override_threshold` | `0.85` → **`0.95`** | Meta_score at which a single engine may bypass corroboration (raised). |
| `allow_single_engine_override` | **`False`** | When False, SEMANTIC types always need ≥2 engines (overrides the threshold bypass). |
| `per_type_emission_thresholds` | `{}` (seed PERSON_NAME/PHONE_NUMBER = 0.65) | Per-type meta_score floor for NER-driven candidates. |

Changing `emission_threshold`/override defaults changes swarm behaviour by design; defaults are chosen conservatively and every knob is overridable for tuning.

## 5. Testing strategy (TDD — RED → GREEN → REFACTOR)

Per-unit unit tests with synthetic engine findings (no full pipeline):
- **U1:** regex EMAIL (conf 0.95) + NER PERSON on the same span → candidate emits **EMAIL** with the regex boundaries.
- **U2:** authoritative candidate's type is unchanged even when `ds.infer()` returns a different type at `ds_conf>0.8`.
- **U3:** regex-only (count=1) high-confidence US_SSN emits (was dropped).
- **U4:** single-NER PERSON_NAME at meta_score 0.86 is suppressed (override removed); a 2-engine PERSON_NAME still emits; an NER candidate below `per_type_emission_thresholds` is suppressed.
- **U5:** a MAC address string is **not** matched as EMAIL; representative real emails still match.
- **Integration:** one `SwarmFusionStrategy.merge` test combining an authoritative SSN (emits) and a single-NER PERSON_NAME (suppressed).

**Gates:** `ruff check src tests`, `mypy src/pii_anon`, full suite via `PYTHONPATH=src .venv/bin/python -m pytest -n auto`.

## 6. Validation & success metrics

The diagnosis effect ranges are **projections, not measurements**. After GREEN, run a measured validation:
- `scripts/diagnose_swarm_precision.py` on a benchmark sample for a fast precision read, then a swarm benchmark pass for the full numbers.
- **Success:** material precision gain (target ≥ +0.15 absolute) with structured-type F1 recovery (EMAIL/PHONE/SSN/IP up toward single-detector levels) and **no F2 regression** vs the current tournament (F2 ≈ 0.885). Record before/after per-entity F1 + FP counts in the spec's results appendix.
- **Core-path check (Unit 5):** confirm the non-swarm pii-anon EMAIL F1 does not regress from the tightened pattern.

## 7. Risks & mitigations

- **Recall trade-off on noisy types** — raising thresholds/removing the override can drop true single-engine NER hits. *Mitigation:* per-type thresholds tuned on the validation pass; F2-no-regression is a hard success criterion.
- **Unit 5 blast radius (shared regex)** — tightening EMAIL affects core. *Mitigation:* core-path regression check is part of slice 1's definition of done.
- **Threshold defaults change committed behaviour** — *Mitigation:* conservative defaults + every knob overridable; validation measures the actual movement before we lock defaults.
- **Authoritative regex over-trust** — a permissive non-EMAIL structured pattern could now bypass corroboration. *Mitigation:* `regex_authority_threshold=0.90` is high; Unit 5 tightens the worst offender; future patterns get checksum/keyword gating (already used for CVV/PIN).

## 8. Traceability

- Diagnosis run: `wf_cf5e9b3c-008` (FP-source, structured-regression, retrain, latency).
- Benchmark evidence: `benchmark-diagnostics.json` per-entity errors (swarm spurious_fp=3,058,191; PERSON_NAME 1,123,530; PHONE 590,106; EMAIL 475,023; US_SSN 328,569); `docs/benchmark-summary.md`; `artifacts/ratings/sp2-12player/tournament.json` (F2 0.885).
- Code anchors: `swarm.py` 50-83 / 535-549 / 620-626 / 686-693 / 720-755 / 774-779; `swarm_learner.py` 142; `regex/patterns.py` (EMAIL); `train_swarm.py` 594-598.
