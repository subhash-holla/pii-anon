# Engine Correlation and Redundancy Analysis

## Summary

- **Records analyzed**: 50
- **Total gold entities**: 337
- **Engines tested**: regex, presidio, gliner, spacy, stanza, scrubadub
- **Union recall** (all engines combined): 1.000

## Individual Engine Recall

| Engine | Recall | TP Count |
|--------|--------|----------|
| regex | 0.988 | 333/337 |
| gliner | 0.727 | 245/337 |
| presidio | 0.691 | 233/337 |
| stanza | 0.374 | 125/337 |
| spacy | 0.344 | 116/337 |
| scrubadub | 0.252 | 85/337 |

## Pairwise Jaccard Similarity (TP Sets)

Higher = engines agree more on what they detect. Values near 1.0 indicate
redundancy (the engines find the same things).

| | regex | presidio | gliner | spacy | stanza | scrubadub |
|---|---|---|---|---|---|---|
| **regex** | 1.000 | 0.680 | 0.725 | 0.332 | 0.362 | 0.255 |
| **presidio** | 0.680 | 1.000 | 0.643 | 0.424 | 0.397 | 0.365 |
| **gliner** | 0.725 | 0.643 | 1.000 | 0.236 | 0.233 | 0.289 |
| **spacy** | 0.332 | 0.424 | 0.236 | 1.000 | 0.891 | 0.000 |
| **stanza** | 0.362 | 0.397 | 0.233 | 0.891 | 1.000 | 0.000 |
| **scrubadub** | 0.255 | 0.365 | 0.289 | 0.000 | 0.000 | 1.000 |

## Pairwise Error Correlation (Co-Miss Rate)

Fraction of ALL gold entities that BOTH engines miss. Higher = more correlated errors.
If engines A and B have high co-miss rate, adding B provides little safety net over A.

| | regex | presidio | gliner | spacy | stanza | scrubadub |
|---|---|---|---|---|---|---|
| **regex** | 0.012 | 0.000 | 0.006 | 0.000 | 0.000 | 0.012 |
| **presidio** | 0.000 | 0.309 | 0.136 | 0.273 | 0.237 | 0.309 |
| **gliner** | 0.006 | 0.136 | 0.273 | 0.134 | 0.107 | 0.240 |
| **spacy** | 0.000 | 0.273 | 0.134 | 0.656 | 0.620 | 0.404 |
| **stanza** | 0.000 | 0.237 | 0.107 | 0.620 | 0.626 | 0.374 |
| **scrubadub** | 0.012 | 0.309 | 0.240 | 0.404 | 0.374 | 0.748 |

## Conditional Co-Miss: P(B misses | A misses)

Given that engine A missed an entity, how likely is engine B to also miss it?
High values mean B provides no safety net when A fails.

| A \ B | regex | presidio | gliner | spacy | stanza | scrubadub |
|---|---|---|---|---|---|---|
| **regex** | - | 0.000 | 0.500 | 0.000 | 0.000 | 1.000 |
| **presidio** | 0.000 | - | 0.442 | 0.885 | 0.769 | 1.000 |
| **gliner** | 0.022 | 0.500 | - | 0.489 | 0.391 | 0.880 |
| **spacy** | 0.000 | 0.416 | 0.204 | - | 0.946 | 0.615 |
| **stanza** | 0.000 | 0.379 | 0.171 | 0.991 | - | 0.597 |
| **scrubadub** | 0.016 | 0.413 | 0.321 | 0.540 | 0.500 | - |

## Unique Contributions (Entities Only This Engine Finds)

| Engine | Unique Entities | % of Gold | Marginal Value |
|--------|----------------|-----------|----------------|
| regex | 25 | 7.4% | 0.074 |
| presidio | 0 | 0.0% | 0.000 |
| gliner | 0 | 0.0% | 0.000 |
| spacy | 0 | 0.0% | 0.000 |
| stanza | 0 | 0.0% | 0.000 |
| scrubadub | 0 | 0.0% | 0.000 |

**Marginal value**: fraction of gold entities lost if this engine is removed
from the ensemble. An engine with marginal value 0.000 is fully redundant.

## Per-Entity-Type Recall by Engine

| Entity Type | regex | presidio | gliner | spacy | stanza | scrubadub |
|---|---|---|---|---|---|---|
| ADDRESS | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| BANK_ACCOUNT | 1.00 | 0.00 | 0.67 | 0.00 | 0.00 | 0.00 |
| CREDIT_CARD | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| CREDIT_CARD_FRAGMENT | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| DATE_OF_BIRTH | 1.00 | 0.00 | 0.57 | 0.00 | 0.00 | 0.00 |
| EMAIL_ADDRESS | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.96 |
| EMPLOYEE_ID | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| IBAN | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| IP_ADDRESS | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| MAC_ADDRESS | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| NATIONAL_ID | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| ORGANIZATION | 1.00 | 0.00 | 0.00 | 0.48 | 0.84 | 0.00 |
| PASSPORT | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| PERSON_NAME | 0.96 | 0.96 | 0.67 | 0.96 | 0.97 | 0.00 |
| PHONE_NUMBER | 1.00 | 0.95 | 1.00 | 0.00 | 0.00 | 0.00 |
| ROUTING_NUMBER | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| USERNAME | 1.00 | 0.00 | 0.80 | 0.00 | 0.00 | 0.00 |
| US_SSN | 1.00 | 1.00 | 0.69 | 0.00 | 0.00 | 0.97 |

## Most Redundant Engine Pairs

Sorted by Jaccard similarity (highest = most redundant):

| Pair | Jaccard | Co-Miss Rate | Conditional Co-Miss (A|B) | Conditional Co-Miss (B|A) |
|------|---------|-------------|---------------------------|---------------------------|
| spacy + stanza | 0.891 | 0.620 | 0.946 | 0.991 |
| regex + gliner | 0.725 | 0.006 | 0.500 | 0.022 |
| regex + presidio | 0.680 | 0.000 | 0.000 | 0.000 |
| presidio + gliner | 0.643 | 0.136 | 0.442 | 0.500 |
| presidio + spacy | 0.424 | 0.273 | 0.885 | 0.416 |
| presidio + stanza | 0.397 | 0.237 | 0.769 | 0.379 |
| presidio + scrubadub | 0.365 | 0.309 | 1.000 | 0.413 |
| regex + stanza | 0.362 | 0.000 | 0.000 | 0.000 |
| regex + spacy | 0.332 | 0.000 | 0.000 | 0.000 |
| gliner + scrubadub | 0.289 | 0.240 | 0.880 | 0.321 |
| regex + scrubadub | 0.255 | 0.012 | 1.000 | 0.016 |
| gliner + spacy | 0.236 | 0.134 | 0.489 | 0.204 |
| gliner + stanza | 0.233 | 0.107 | 0.391 | 0.171 |
| spacy + scrubadub | 0.000 | 0.404 | 0.615 | 0.540 |
| stanza + scrubadub | 0.000 | 0.374 | 0.597 | 0.500 |

## Key Findings and Recommendations

1. **Most redundant pair**: spacy + stanza (Jaccard = 0.891)
   - Co-miss rate: 0.620 -- they miss the same entities 62.0% of the time
2. **Least redundant pair**: stanza + scrubadub (Jaccard = 0.000)
   - These engines have the most complementary detection patterns
3. **Fully redundant engines** (marginal value ~0): presidio, gliner, spacy, stanza, scrubadub
   - Removing these engines would not reduce ensemble recall
4. **Highest marginal value**: regex (0.074)
   - This engine contributes the most unique detections to the ensemble

## Implications for Swarm Architecture

If two engines have high Jaccard similarity AND high conditional co-miss rates,
they are making correlated errors. Adding both to the swarm provides diminishing
returns. The swarm benefits most from engines with LOW Jaccard similarity
(complementary strengths) and LOW conditional co-miss (independent error modes).

---
*Generated by `scripts/diagnose_engine_correlation.py` on 2026-03-27 15:40:40*
