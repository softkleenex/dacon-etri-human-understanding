# Post-ID342 public-scored target-swap gate

Created: `2026-06-24 05:16:25 UTC+09:00`

This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Stable public-donor pass count: `0`
- Failed-frontier trap count: `44`
- Selected stable targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Do not use ID287/ID329 target swaps despite large local gains. Stable public-scored donors do not pass the Q3/S3 gate, so this lane is anatomy evidence only.

## Top Stable Donor Rows

| config | donor type | wins | mean delta | max delta | projected public |
|---|---|---:|---:|---:|---:|
| `ID342_Q3_donor178_w0.3` | `stable_public_scored` | 3 | -0.000170705 | 0.000844442 | 0.596058865 |
| `ID342_Q3_donor337_w0.3` | `stable_public_scored` | 3 | -0.000170705 | 0.000844442 | 0.596058865 |
| `ID342_Q3_donor340_w0.3` | `stable_public_scored` | 3 | -0.000170705 | 0.000844442 | 0.596058865 |
| `ID342_Q3_donor178_w0.5` | `stable_public_scored` | 3 | -0.000153121 | 0.001553838 | 0.596061377 |
| `ID342_Q3_donor337_w0.5` | `stable_public_scored` | 3 | -0.000153121 | 0.001553838 | 0.596061377 |
| `ID342_Q3_donor340_w0.5` | `stable_public_scored` | 3 | -0.000153121 | 0.001553838 | 0.596061377 |
| `ID342_Q3_donor178_w0.2` | `stable_public_scored` | 3 | -0.000139899 | 0.000534015 | 0.596063266 |
| `ID342_Q3_donor337_w0.2` | `stable_public_scored` | 3 | -0.000139899 | 0.000534015 | 0.596063266 |
| `ID342_Q3_donor340_w0.2` | `stable_public_scored` | 3 | -0.000139899 | 0.000534015 | 0.596063266 |
| `ID342_Q3_donor178_w0.15` | `stable_public_scored` | 3 | -0.000114679 | 0.000389715 | 0.596066869 |

## Top Failed-Frontier Rows

These rows are deliberately not promotable because ID287 and ID329 are confirmed public failures.

| config | donor type | wins | mean delta | max delta | projected public |
|---|---|---:|---:|---:|---:|
| `ID342_Q3_donor287_w1` | `failed_frontier` | 5 | -0.219765365 | -0.205917267 | 0.564688200 |
| `ID342_Q3_donor287_w0.5` | `failed_frontier` | 5 | -0.120979284 | -0.112159054 | 0.578800497 |
| `ID342_Q3_donor287_w0.3` | `failed_frontier` | 5 | -0.075905770 | -0.069999171 | 0.585239571 |
| `ID342_Q3_donor287_w0.2` | `failed_frontier` | 5 | -0.051842998 | -0.047665725 | 0.588677109 |
| `ID342_Q3_donor287_w0.15` | `failed_frontier` | 5 | -0.039377006 | -0.036146101 | 0.590457965 |
| `ID342_Q3_donor287_w0.1` | `failed_frontier` | 5 | -0.026596237 | -0.024372847 | 0.592283789 |
| `ID342_S3_donor287_w1` | `failed_frontier` | 5 | -0.023867650 | -0.021548740 | 0.592673588 |
| `ID342_Q3_donor287_w0.08` | `failed_frontier` | 5 | -0.021390999 | -0.019589045 | 0.593027395 |
| `ID342_Q3_donor329_w1` | `failed_frontier` | 5 | -0.020574133 | -0.015393301 | 0.593144090 |
| `ID342_Q3_donor287_w0.05` | `failed_frontier` | 5 | -0.013478865 | -0.012330118 | 0.594157700 |
