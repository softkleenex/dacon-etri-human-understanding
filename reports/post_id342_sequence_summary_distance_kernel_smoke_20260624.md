# Post-ID342 sequence-summary distance/kernel smoke

Created: `2026-06-24 03:14:11 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Feature Contract

- Source: 44 half-hour raw-sensor sequence table plus label-free calendar/progress features
- Domain read: Q3 stress is screened through activity, phone usage, screen state, ambience, and heart-rate shape; S3 sleep-onset-latency is screened through evening/pre-sleep light, movement, screen, charging, and heart-rate shape.
- Sequence rows: `809`
- Total channels: `68`
- Sequence summary features: `2108`
- Calendar/progress features: `8`
- Total features: `2116`

## Target Results

| target | selected | best config | wins | mean delta | max delta |
|---|---:|---|---:|---:|---:|
| Q3 | `False` | `Q3_id342_seqdist_q3_stress_activity_usage_hgb_l7_w0.005` | 2 | +0.000055 | +0.000234 |
| S3 | `False` | `S3_id342_seqdist_all_ssknn_k1_w0.005` | 1 | +0.000092 | +0.000237 |

## Top Rows

### `Q3`

| rank | config | wins | mean delta | max delta |
|---:|---|---:|---:|---:|
| 1 | `Q3_id342_seqdist_q3_stress_activity_usage_hgb_l7_w0.005` | 2 | +0.000055 | +0.000234 |
| 2 | `Q3_id342_seqdist_s3_sol_light_motion_hgb_l7_w0.005` | 2 | +0.000094 | +0.000363 |
| 3 | `Q3_id342_seqdist_q3_stress_activity_usage_hgb_l7_w0.01` | 2 | +0.000116 | +0.000474 |
| 4 | `Q3_id342_seqdist_all_hgb_l7_w0.005` | 1 | +0.000118 | +0.000389 |
| 5 | `Q3_id342_seqdist_paper_core_rbf_k25_s1_0_w0.005` | 2 | +0.000127 | +0.000406 |

### `S3`

| rank | config | wins | mean delta | max delta |
|---:|---|---:|---:|---:|
| 1 | `S3_id342_seqdist_all_ssknn_k1_w0.005` | 1 | +0.000092 | +0.000237 |
| 2 | `S3_id342_seqdist_paper_core_hgb_l7_w0.005` | 1 | +0.000112 | +0.000267 |
| 3 | `S3_id342_seqdist_paper_core_ssknn_k3_w0.005` | 1 | +0.000125 | +0.000263 |
| 4 | `S3_id342_seqdist_q3_stress_activity_usage_hgb_l7_w0.005` | 1 | +0.000133 | +0.000238 |
| 5 | `S3_id342_seqdist_paper_core_rbf_k25_s1_0_w0.005` | 1 | +0.000134 | +0.000238 |

## Decision

- Selected targets: `[]`
- Sequence distance has Q3/S3 signal: `False`
- Partial signal targets: `[]`
- Final gate pass: `False`
- Projected public proxy from ID342: `0.596083252`
- Gap to Public 0.55 after proxy: `+0.046083252`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Use Kaggle no-submit only if the final gate passes; otherwise keep this as negative/partial sequence-distance evidence and do not create a DACON artifact.
