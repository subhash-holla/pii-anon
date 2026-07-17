# pii-rate-elo — assessment tournament

Source: `artifacts/ratings/v1.7.0rc1-13player/` merged assessment (11 competitors @ sp5 vintage + first-party fresh v1.7.0rc1) · matching policy `strict-v1` · split `test` · languages `en` · records 31048 · gold spans 201880

Matches: 66 gold-supported entity types x 13 players (all pairs) = 5148 matches.

## Leaderboard

| # | System | Elo | ±RD | 95% CI | F2 (micro) | F2 (macro) | Precision | Recall | Coverage |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | pii_anon_swarm | 1868.71 | 30.00 | [1810, 1928] | 0.908 | 0.772 | 0.853 | 0.923 | 66/66 |
| 2 | pii_anon | 1867.81 | 30.00 | [1809, 1927] | 0.905 | 0.772 | 0.867 | 0.915 | 66/66 |
| 3 | aws | 1541.40 | 30.00 | [1483, 1600] | 0.736 | 0.273 | 0.769 | 0.728 | 24/66 |
| 4 | gliner | 1489.45 | 30.00 | [1431, 1548] | 0.734 | 0.242 | 0.813 | 0.716 | 23/66 |
| 5 | gcp | 1448.46 | 30.00 | [1390, 1507] | 0.704 | 0.146 | 0.722 | 0.700 | 18/66 |
| 6 | azure | 1444.68 | 30.00 | [1386, 1503] | 0.696 | 0.133 | 0.730 | 0.688 | 17/66 |
| 7 | piiranha | 1437.43 | 30.00 | [1379, 1496] | 0.345 | 0.108 | 0.441 | 0.327 | 16/66 |
| 8 | scrubadub | 1431.10 | 30.00 | [1372, 1490] | 0.201 | 0.041 | 0.818 | 0.169 | 12/66 |
| 9 | regex | 1414.63 | 30.00 | [1356, 1473] | 0.396 | 0.121 | 0.857 | 0.349 | 9/66 |
| 10 | presidio | 1393.84 | 30.00 | [1335, 1453] | 0.526 | 0.091 | 0.419 | 0.562 | 20/66 |
| 11 | flair | 1385.71 | 30.00 | [1327, 1445] | 0.326 | 0.021 | 0.565 | 0.295 | 3/66 |
| 12 | spacy | 1385.59 | 30.00 | [1327, 1444] | 0.317 | 0.018 | 0.464 | 0.294 | 3/66 |
| 13 | stanza | 1385.22 | 30.00 | [1326, 1444] | 0.340 | 0.021 | 0.583 | 0.308 | 3/66 |

## Pairwise significance

`Y` = rating gap exceeds 2·sqrt(RD_i² + RD_j²) (statistically distinguishable); `~` = within noise.

| vs | pii_anon_swarm | pii_anon | aws | gliner | gcp | azure | piiranha | scrubadub | regex | presidio | flair | spacy | stanza |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pii_anon_swarm | — | ~ | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| pii_anon | ~ | — | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| aws | Y | Y | — | ~ | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| gliner | Y | Y | ~ | — | ~ | ~ | ~ | ~ | ~ | Y | Y | Y | Y |
| gcp | Y | Y | Y | ~ | — | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| azure | Y | Y | Y | ~ | ~ | — | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| piiranha | Y | Y | Y | ~ | ~ | ~ | — | ~ | ~ | ~ | ~ | ~ | ~ |
| scrubadub | Y | Y | Y | ~ | ~ | ~ | ~ | — | ~ | ~ | ~ | ~ | ~ |
| regex | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | — | ~ | ~ | ~ | ~ |
| presidio | Y | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | — | ~ | ~ | ~ |
| flair | Y | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | ~ | — | ~ | ~ |
| spacy | Y | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | ~ | ~ | — | ~ |
| stanza | Y | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | ~ | ~ | ~ | — |

## Per-system strengths and blind spots

| System | Strongest entity types (F2) | Weakest entity types (F2) |
|---|---|---|
| pii_anon_swarm | AUTHENTICATION_TOKEN (1.00), BAR_NUMBER (1.00), CVV (1.00) | POLITICAL_OPINION (0.02), AGE (0.02), ETHNICITY (0.12) |
| pii_anon | AUTHENTICATION_TOKEN (1.00), BAR_NUMBER (1.00), CVV (1.00) | POLITICAL_OPINION (0.02), AGE (0.02), ETHNICITY (0.12) |
| aws | EMAIL_ADDRESS (0.98), PHONE_NUMBER (0.96), IP_ADDRESS (0.93) | AUTHENTICATION_TOKEN (0.00), BAR_NUMBER (0.00), BIOMETRIC_ID (0.00) |
| gliner | PHONE_NUMBER (0.98), EMAIL_ADDRESS (0.97), DATE_OF_BIRTH (0.96) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| gcp | IP_ADDRESS (0.99), EMAIL_ADDRESS (0.99), PHONE_NUMBER (0.97) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| azure | EMAIL_ADDRESS (0.97), STREET_ADDRESS (0.93), PERSON_NAME (0.89) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| piiranha | DATE_OF_BIRTH (0.93), EMAIL_ADDRESS (0.91), PHONE_NUMBER (0.82) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| scrubadub | EMAIL_ADDRESS (0.86), SOCIAL_MEDIA_HANDLE (0.66), URL (0.63) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| regex | TIMESTAMP (1.00), IP_ADDRESS (0.99), EMAIL_ADDRESS (0.99) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| presidio | IP_ADDRESS (0.99), EMAIL_ADDRESS (0.98), SOCIAL_SECURITY_NUMBER (0.92) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| flair | PERSON_NAME (0.83), ORGANIZATION_NAME (0.53), LOCATION_NAME (0.02) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| spacy | PERSON_NAME (0.79), ORGANIZATION_NAME (0.38), LOCATION_NAME (0.02) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| stanza | PERSON_NAME (0.84), ORGANIZATION_NAME (0.51), LOCATION_NAME (0.04) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |

## Axes evaluated

- Detection quality (precision / recall / F1 / F2, micro + macro, per-entity): from this artifact, all systems.
- Entity-type coverage (label-map projection ceiling): from this artifact, all systems.
- Latency / throughput: NOT carried by this artifact — no rating credit or penalty assigned. (First-party systems carry measured latency in the internal benchmark artifact.)
- Tier-3 re-identification resistance: NOT carried by this artifact — no rating credit or penalty assigned.
