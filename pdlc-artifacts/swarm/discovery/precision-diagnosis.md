# Swarm Precision Diagnosis Report

**Records analyzed**: 100
**Gold labels**: 690

## Aggregate Results

| Metric | Regex-Only | Swarm (MoE) | Delta |
|--------|-----------|-------------|-------|
| TP | 651 | 659 | +8 |
| FP | 114 | 495 | +381 |
| FN | 39 | 31 | -8 |
| Precision | 0.8510 | 0.5711 | -0.2799 |
| Recall | 0.9435 | 0.9551 | +0.0116 |
| F1 | 0.8948 | 0.7148 | -0.1800 |

## Per-Entity-Type Breakdown (sorted by swarm FP desc)

| Entity Type | Regex TP/FP/FN | Swarm TP/FP/FN | FP Delta |
|-------------|---------------|----------------|----------|
| PERSON_NAME | 184/12/37 | 192/115/29 | +103 |
| EMAIL_ADDRESS | 102/0/0 | 102/112/0 | +112 |
| PHONE_NUMBER | 80/42/0 | 80/93/0 | +51 |
| US_SSN | 65/2/0 | 65/76/0 | +74 |
| DATE_OF_BIRTH | 17/14/0 | 17/20/0 | +6 |
| ADDRESS | 71/0/0 | 71/17/0 | +17 |
| BANK_ACCOUNT | 14/0/0 | 14/16/0 | +16 |
| IP_ADDRESS | 5/0/0 | 5/11/0 | +11 |
| ORGANIZATION | 48/39/2 | 48/11/2 | -28 |
| CREDIT_CARD | 11/0/0 | 11/9/0 | +9 |
| DRIVERS_LICENSE | 4/0/0 | 4/7/0 | +7 |
| USERNAME | 7/0/0 | 7/6/0 | +6 |
| MAC_ADDRESS | 8/0/0 | 8/1/0 | +1 |
| MEDICAL_RECORD_NUMBER | 7/0/0 | 7/1/0 | +1 |
| EMPLOYEE_ID | 15/3/0 | 15/0/0 | -3 |
| LICENSE_PLATE | 1/1/0 | 1/0/0 | -1 |
| LOCATION | 1/0/0 | 1/0/0 | +0 |
| NATIONAL_ID | 2/1/0 | 2/0/0 | -1 |
| PASSPORT | 8/0/0 | 8/0/0 | +0 |
| ROUTING_NUMBER | 1/0/0 | 1/0/0 | +0 |

## Engine FP Attribution

Which engines contributed to false positive findings in the swarm:

### gliner-compatible (412 FP contributions)

| Entity Type | FP Count |
|-------------|----------|
| EMAIL_ADDRESS | 102 |
| PERSON_NAME | 102 |
| PHONE_NUMBER | 80 |
| US_SSN | 48 |
| DATE_OF_BIRTH | 20 |
| ADDRESS | 17 |
| IP_ADDRESS | 11 |
| BANK_ACCOUNT | 10 |
| CREDIT_CARD | 9 |
| DRIVERS_LICENSE | 7 |
| USERNAME | 6 |

### presidio-compatible (397 FP contributions)

| Entity Type | FP Count |
|-------------|----------|
| EMAIL_ADDRESS | 102 |
| PERSON_NAME | 102 |
| PHONE_NUMBER | 78 |
| US_SSN | 68 |
| ADDRESS | 17 |
| BANK_ACCOUNT | 16 |
| DRIVERS_LICENSE | 7 |
| IP_ADDRESS | 5 |
| MEDICAL_RECORD_NUMBER | 1 |
| MAC_ADDRESS | 1 |

### regex-oss (32 FP contributions)

| Entity Type | FP Count |
|-------------|----------|
| PERSON_NAME | 13 |
| ORGANIZATION | 11 |
| PHONE_NUMBER | 6 |
| US_SSN | 2 |

### scrubadub-compatible (210 FP contributions)

| Entity Type | FP Count |
|-------------|----------|
| EMAIL_ADDRESS | 102 |
| US_SSN | 64 |
| PHONE_NUMBER | 44 |

## FP Examples (first 30)

- **[CUR-026659]** `EMAIL_ADDRESS` [40:65]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...ike Becker, E-Mail: heike.becker59@outlook.de, Telefon: +1 (459) ...`
- **[CUR-026659]** `PERSON_NAME` [18:30]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...Mitarbeiterdaten: Heike Becker, E-Mail: heike.beck...`
- **[CUR-026659]** `US_SSN` [122:133]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...ersicherungsnummer: 659-29-9659. Adresse: 6959 Dogw...`
- **[CUR-026659]** `PHONE_NUMBER` [76:93]
  - Engines: gliner-compatible
  - Context: `...utlook.de, Telefon: +1 (459) 413-5567. Sozialversicherung...`
- **[SYN-010165]** `EMAIL_ADDRESS` [133:156]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...d, PA 20165. Email: rthompson@starklabs.com. Phone: +1 (765) 95...`
- **[SYN-010165]** `PERSON_NAME` [9:24]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...Employee Rachel Thompson (ID: EMP-20165) at ...`
- **[SYN-010165]** `US_SSN` [68:79]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...rdyne Systems. SSN: 365-95-2165. Address: 365 Maple...`
- **[SYN-010165]** `PHONE_NUMBER` [165:182]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...arklabs.com. Phone: +1 (765) 955-7145. Device id MAC: 6a:...`
- **[SYN-010165]** `IP_ADDRESS` [199:216]
  - Engines: gliner-compatible
  - Context: `...145. Device id MAC: 6a:1f:89:f3:c7:31....`
- **[SYN-008310]** `EMAIL_ADDRESS` [56:83]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...holas Walker
Email: nicholas.walker10@gmail.com
Phone: +1 (510) 770...`
- **[SYN-008310]** `EMAIL_ADDRESS` [661:688]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...ion will be sent to nicholas.walker10@gmail.com. Is there anything ...`
- **[SYN-008310]** `PERSON_NAME` [33:48]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...t #58310

Customer: Nicholas Walker
Email: nicholas.wal...`
- **[SYN-008310]** `PHONE_NUMBER` [91:108]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...10@gmail.com
Phone: +1 (510) 770-1030
Account: 10108030

...`
- **[SYN-008310]** `BANK_ACCOUNT` [118:126]
  - Engines: presidio-compatible
  - Context: `...) 770-1030
Account: 10108030

[Nicholas]: Hi, I ...`
- **[SYN-008310]** `US_SSN` [394:398]
  - Engines: gliner-compatible
  - Context: `...sing the SSN ending 9310. Let me look into t...`
- **[SYN-008310]** `CREDIT_CARD` [441:460]
  - Engines: gliner-compatible
  - Context: `...transaction on card 4000-0000-0030-7470.

[Nicholas]: Thank...`
- **[SYN-002873]** `EMAIL_ADDRESS` [514:541]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...349 and my email is janet.hall73@protonmail.com.

The witness, Jame...`
- **[SYN-002873]** `EMAIL_ADDRESS` [614:639]
  - Engines: scrubadub-compatible
  - Context: `...unt. Witness can be reached at james.phillips@protonmail.com....`
- **[SYN-002873]** `EMAIL_ADDRESS` [625:654]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...s can be reached at james.phillips@protonmail.com....`
- **[SYN-002873]** `PERSON_NAME` [50:60]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...o. 32873

Deponent: Janet Hall, residing at 2973 H...`
- **[SYN-002873]** `PERSON_NAME` [234:244]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...cord.
A: My name is Janet Hall. Some colleagues ca...`
- **[SYN-002873]** `US_SSN` [401:412]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...security number?
A: 273-93-3873.

Q: Can you confir...`
- **[SYN-002873]** `PHONE_NUMBER` [483:497]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...Yes, my phone is +1 (673) 311-2349 and my email is jan...`
- **[SYN-002873]** `USERNAME` [300:312]
  - Engines: gliner-compatible
  - Context: `... my email handle is janet.hall73.

Q: What is your d...`
- **[SYN-026328]** `EMAIL_ADDRESS` [44:72]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...ng Mueller, E-Mail: wolfgang.mueller28@gmail.com, Telefon: +1 (928) ...`
- **[SYN-026328]** `PERSON_NAME` [18:34]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...Mitarbeiterdaten: Wolfgang Mueller, E-Mail: wolfgang.m...`
- **[SYN-026328]** `US_SSN` [129:140]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...ersicherungsnummer: 328-58-9328. Adresse: 6628 Spru...`
- **[SYN-026328]** `PHONE_NUMBER` [83:100]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...gmail.com, Telefon: +1 (928) 496-1264. Sozialversicherung...`
- **[CUR-014517]** `EMAIL_ADDRESS` [346:370]
  - Engines: gliner-compatible, presidio-compatible, scrubadub-compatible
  - Context: `...t meeting. Contact: lisa.ramirez17@yahoo.com, +1 (317) 219-9721....`
- **[CUR-014517]** `PERSON_NAME` [0:12]
  - Engines: gliner-compatible, presidio-compatible
  - Context: `...Lisa Ramirez submitted the quart...`

## Key Findings

**Entity types with most excess FPs in swarm:**

- EMAIL_ADDRESS: +112 FPs (total 112)
- PERSON_NAME: +103 FPs (total 115)
- US_SSN: +74 FPs (total 76)
- PHONE_NUMBER: +51 FPs (total 93)
- ADDRESS: +17 FPs (total 17)
- BANK_ACCOUNT: +16 FPs (total 16)
- IP_ADDRESS: +11 FPs (total 11)
- CREDIT_CARD: +9 FPs (total 9)
- DRIVERS_LICENSE: +7 FPs (total 7)
- DATE_OF_BIRTH: +6 FPs (total 20)

**Engines most responsible for FPs:**

- gliner-compatible: 412 FP contributions
- presidio-compatible: 397 FP contributions
- scrubadub-compatible: 210 FP contributions
- regex-oss: 32 FP contributions
