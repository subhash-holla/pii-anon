# Security Checklist — pii-anon (pii-anon-code library)

This library detects and transforms PII. The load-bearing security/ethics control is **axiom [AX-pii-anon-001](../00-axioms/project-axioms.yaml): synthetic-only, no real PII** in any shipped artifact (src, tests, fixtures, docs, bundled data). No domain pack is active; this checklist is the manual baseline until an automated real-PII leakage pattern set is authored in Stage 4 Development.

## Release-blocking invariants

- [ ] **No real PII in the repo.** Every PII-shaped value in tests/fixtures/docs traces to a synthetic generator or surrogate pool. No real natural person is identifiable.
- [ ] **No accidental valid identifiers.** Structurally-valid SSNs, Luhn-valid card numbers, real IBANs/routing numbers, real API keys/tokens appear ONLY as synthetic-generator output — never copied from real sources.
- [ ] **No secrets in the repo.** No real API keys, tokens, or credentials in code, configs, fixtures, CI, or docs (use `sk-ant-test-…` / `XXXX` placeholders). Cloud/LLM baseline + attacker code (Theme 2/3) must read credentials from the environment, never hard-code them.
- [ ] **No raw PII persisted post-masking.** Agentic interception (Theme 3) must not write raw sensitive content to traces/logs/memory after the masking step (axiom AX-006).
- [ ] **License preserved.** `LICENSE` intact; third-party model/data licenses for new baselines/benchmarks documented and gate-checked (cloud/LLM baselines stay out of the MIT-qualified headline claim).

## Distinguishing synthetic PII (allowed) from real-PII leakage (blocked)

A PII library's tests are *supposed* to contain PII-shaped strings; a naive scanner would flag everything. The Stage 4 scanner must detect **leakage** signals instead:
- Identifiers that pass real-world checksums AND were not emitted by a declared synthetic generator.
- Real public-figure names co-occurring with real contact details.
- Real corporate email domains tied to named individuals outside the synthetic org pool.

## Provenance & determinism

- [ ] Pseudonymization/tokenization is deterministic given (value, key, scope) — reproducible surrogates (axiom [AX-pii-anon-002](../00-axioms/project-axioms.yaml)).
- [ ] Any LLM-based detector/attacker/judge records model id + prompt + decoding params + seed.

_Scan exceptions (if any) are tracked in [exceptions.yaml](exceptions.yaml) with sign-off citations._
