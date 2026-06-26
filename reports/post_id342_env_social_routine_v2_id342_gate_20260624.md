# Post-ID342 env/social/routine v2 ID342 gate

Created: `2026-06-24 04:54:48 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Research basis

- [ETRI Lifelog Dataset 2024](https://arxiv.org/html/2508.03698v1): Dataset paper frames the task around continuous sensing for sleep quality and stress.
- [Human Understanding AI Paper Challenge 2024 - Dataset Design](https://cfiles.dacon.co.kr/competitions/236690/Human%20Understanding%20AI%20Paper%20Challenge%202024%20-%20Dataset%20Design.pdf): DACON/ETRI design note emphasizes daily experiences, sleep patterns, emotional states, and stress.
- [Monitoring Daily Sleep, Mood, and Affect Using Digital Technologies and Wearables](https://www.mdpi.com/1424-8220/24/14/4701): Systematic review motivates day-to-day sleep/affect dynamics under ambulatory monitoring.
- [Detection and monitoring of stress using wearables: a systematic review](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1478851/full): Wearable stress review supports low-level feature computation and cautious validation.

## Feature input

- Aligned train rows: `450`
- Feature columns used: `494`
- Mean missing rate: `0.0221`

## Selected target configs

| target | candidate | wins | mean Δ342 | max Δ342 |
|---|---|---:|---:|---:|
| `Q2` | `blend_id342_envsocial_lr_c0_20_w0.025` | 5 | -0.001177 | -0.000095 |

## Top rows by target

### `Q2`

| rank | candidate | wins | mean Δ342 | max Δ342 |
|---:|---|---:|---:|---:|
| 1 | `blend_id342_envsocial_lr_c0_20_w0.025` | 5 | -0.001177 | -0.000095 |
| 2 | `blend_id342_envsocial_lr_c0_08_w0.025` | 5 | -0.001109 | -0.000022 |
| 3 | `blend_id342_envsocial_lr_c0_20_w0.01` | 5 | -0.000538 | -0.000111 |
| 4 | `blend_id342_envsocial_lr_c0_08_w0.01` | 5 | -0.000502 | -0.000052 |
| 5 | `blend_id342_envsocial_lr_c0_20_w0.005` | 5 | -0.000281 | -0.000068 |

### `Q3`

| rank | candidate | wins | mean Δ342 | max Δ342 |
|---:|---|---:|---:|---:|
| 1 | `blend_id342_envsocial_lr_c0_02_w0.005` | 0 | +0.000254 | +0.000362 |
| 2 | `blend_id342_envsocial_lr_c0_08_w0.005` | 0 | +0.000311 | +0.000542 |
| 3 | `blend_id342_envsocial_lr_c0_20_w0.005` | 0 | +0.000334 | +0.000565 |
| 4 | `blend_id342_envsocial_lr_c0_02_w0.01` | 0 | +0.000518 | +0.000735 |
| 5 | `blend_id342_envsocial_lr_c0_08_w0.01` | 0 | +0.000636 | +0.001100 |

### `S3`

| rank | candidate | wins | mean Δ342 | max Δ342 |
|---:|---|---:|---:|---:|
| 1 | `blend_id342_envsocial_lr_c0_02_w0.005` | 0 | +0.000503 | +0.000829 |
| 2 | `blend_id342_envsocial_lr_c0_20_w0.005` | 0 | +0.000598 | +0.001043 |
| 3 | `blend_id342_envsocial_lr_c0_08_w0.005` | 0 | +0.000600 | +0.001012 |
| 4 | `blend_id342_envsocial_lr_c0_02_w0.01` | 0 | +0.001018 | +0.001662 |
| 5 | `blend_id342_envsocial_lr_c0_20_w0.01` | 0 | +0.001216 | +0.002096 |

## Decision

- Model smoke completed: `True`
- Selected targets: `['Q2']`
- Any env/social/routine signal found: `True`
- Estimated all-target Δ vs ID342: `-0.000168099`
- Projected public proxy: `0.595915153`
- Gap to Public 0.55 proxy: `+0.045915153`
- Public-proxy reviewable: `False`
- Public 0.55 evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Selected rows are too small or too narrow for promotion; require a separate geometry/artifact gate before any remote action.

