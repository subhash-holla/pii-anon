# pii-rate-elo — assessment tournament

Source: `/Users/subhashholla/Development/pii_anonymize_pseudonymize/pii-anon-core/pii-anon-eval-data/results/baselines/sp2-tier1-en-12/baseline_results.json` · matching policy `strict-v1` · split `test` · languages `en` · records 30995 · gold spans 201701

Matches: 63 gold-supported entity types x 12 players (all pairs) = 4158 matches.

## Leaderboard

| # | System | Elo | ±RD | 95% CI | F2 (micro) | F2 (macro) | Precision | Recall | Coverage |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | pii_anon | 1824.89 | 30.00 | [1766, 1884] | 0.884 | 0.685 | 0.869 | 0.888 | 63/63 |
| 2 | pii_anon_swarm | 1823.81 | 30.00 | [1765, 1883] | 0.885 | 0.680 | 0.859 | 0.892 | 63/63 |
| 3 | aws | 1541.88 | 30.00 | [1483, 1601] | 0.737 | 0.285 | 0.769 | 0.729 | 24/63 |
| 4 | gliner | 1494.93 | 30.00 | [1436, 1554] | 0.735 | 0.255 | 0.812 | 0.718 | 23/63 |
| 5 | gcp | 1449.16 | 30.00 | [1390, 1508] | 0.705 | 0.151 | 0.722 | 0.701 | 18/63 |
| 6 | azure | 1447.00 | 30.00 | [1388, 1506] | 0.696 | 0.140 | 0.730 | 0.688 | 17/63 |
| 7 | piiranha | 1434.68 | 30.00 | [1376, 1493] | 0.347 | 0.115 | 0.444 | 0.329 | 16/63 |
| 8 | scrubadub | 1426.34 | 30.00 | [1368, 1485] | 0.199 | 0.043 | 0.817 | 0.168 | 12/63 |
| 9 | regex | 1416.09 | 30.00 | [1357, 1475] | 0.395 | 0.127 | 0.856 | 0.348 | 9/63 |
| 10 | presidio | 1388.73 | 30.00 | [1330, 1448] | 0.527 | 0.096 | 0.419 | 0.563 | 20/63 |
| 11 | spacy | 1376.33 | 30.00 | [1318, 1435] | 0.317 | 0.019 | 0.463 | 0.294 | 3/63 |
| 12 | stanza | 1376.12 | 30.00 | [1317, 1435] | 0.340 | 0.022 | 0.581 | 0.308 | 3/63 |

## Pairwise significance

`Y` = rating gap exceeds 2·sqrt(RD_i² + RD_j²) (statistically distinguishable); `~` = within noise.

| vs | pii_anon | pii_anon_swarm | aws | gliner | gcp | azure | piiranha | scrubadub | regex | presidio | spacy | stanza |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pii_anon | — | ~ | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| pii_anon_swarm | ~ | — | Y | Y | Y | Y | Y | Y | Y | Y | Y | Y |
| aws | Y | Y | — | ~ | Y | Y | Y | Y | Y | Y | Y | Y |
| gliner | Y | Y | ~ | — | ~ | ~ | ~ | ~ | ~ | Y | Y | Y |
| gcp | Y | Y | Y | ~ | — | ~ | ~ | ~ | ~ | ~ | ~ | ~ |
| azure | Y | Y | Y | ~ | ~ | — | ~ | ~ | ~ | ~ | ~ | ~ |
| piiranha | Y | Y | Y | ~ | ~ | ~ | — | ~ | ~ | ~ | ~ | ~ |
| scrubadub | Y | Y | Y | ~ | ~ | ~ | ~ | — | ~ | ~ | ~ | ~ |
| regex | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | — | ~ | ~ | ~ |
| presidio | Y | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | — | ~ | ~ |
| spacy | Y | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | ~ | — | ~ |
| stanza | Y | Y | Y | Y | ~ | ~ | ~ | ~ | ~ | ~ | ~ | — |

## Per-system strengths and blind spots

| System | Strongest entity types (F2) | Weakest entity types (F2) |
|---|---|---|
| pii_anon | BAR_NUMBER (1.00), DEA_NUMBER (1.00), DEVICE_IDENTIFIER (1.00) | AGE (0.00), AUTHENTICATION_TOKEN (0.00), ETHNICITY (0.00) |
| pii_anon_swarm | BAR_NUMBER (1.00), DEA_NUMBER (1.00), DEVICE_IDENTIFIER (1.00) | AGE (0.00), AUTHENTICATION_TOKEN (0.00), ETHNICITY (0.00) |
| aws | EMAIL_ADDRESS (0.98), PHONE_NUMBER (0.96), IP_ADDRESS (0.93) | AGE (0.00), AUTHENTICATION_TOKEN (0.00), BAR_NUMBER (0.00) |
| gliner | PHONE_NUMBER (0.98), EMAIL_ADDRESS (0.97), DATE_OF_BIRTH (0.96) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| gcp | IP_ADDRESS (0.99), EMAIL_ADDRESS (0.99), PHONE_NUMBER (0.98) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| azure | EMAIL_ADDRESS (0.97), STREET_ADDRESS (0.93), PERSON_NAME (0.89) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| piiranha | DATE_OF_BIRTH (0.92), EMAIL_ADDRESS (0.91), BANK_ACCOUNT_NUMBER (0.82) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| scrubadub | EMAIL_ADDRESS (0.86), SOCIAL_MEDIA_HANDLE (0.70), URL (0.64) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| regex | TIMESTAMP (1.00), IP_ADDRESS (0.99), EMAIL_ADDRESS (0.99) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| presidio | IP_ADDRESS (0.99), EMAIL_ADDRESS (0.98), SOCIAL_SECURITY_NUMBER (0.92) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| spacy | PERSON_NAME (0.80), ORGANIZATION_NAME (0.37), LOCATION_NAME (0.02) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |
| stanza | PERSON_NAME (0.84), ORGANIZATION_NAME (0.50), LOCATION_NAME (0.04) | AGE (0.00), API_KEY (0.00), AUTHENTICATION_TOKEN (0.00) |

## Axes evaluated

- Detection quality (precision / recall / F1 / F2, micro + macro, per-entity): from this artifact, all systems.
- Entity-type coverage (label-map projection ceiling): from this artifact, all systems.
- Latency / throughput: NOT carried by this artifact — no rating credit or penalty assigned. (First-party systems carry measured latency in the internal benchmark artifact.)
- Tier-3 re-identification resistance: NOT carried by this artifact — no rating credit or penalty assigned.
