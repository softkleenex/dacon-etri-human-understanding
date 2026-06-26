# Post-ID342 sleep-boundary Q3/S3 rescue smoke

Created: `2026-06-24 03:19:56 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Feature Contract

- Domain read: S3 is sleep-onset-latency guideline adherence; Q3 is pre-sleep stress. This smoke emphasizes screen-off, charging, low movement, light drop, and heart-rate boundary features.
- Rows: `700` total / `450` train / `250` test
- Feature columns: `294`
- Train/test missing mean: `0.0687` / `0.0673`

## Target Results

| target | selected | best config | wins | mean delta | max delta |
|---|---:|---|---:|---:|---:|
| Q3 | `False` | `Q3_id342_sleep_boundary_stress_hr_activity_screen_hgb_l5_l2_2_w0.005` | 0 | +0.000175 | +0.000317 |
| S3 | `False` | `S3_id342_sleep_boundary_all_hgb_l5_l2_2_w0.005` | 1 | +0.000101 | +0.000260 |

## Top Rows

### `Q3`

| rank | config | wins | mean delta | max delta |
|---:|---|---:|---:|---:|
| 1 | `Q3_id342_sleep_boundary_stress_hr_activity_screen_hgb_l5_l2_2_w0.005` | 0 | +0.000175 | +0.000317 |
| 2 | `Q3_id342_sleep_boundary_stress_hr_activity_screen_lr_c0_05_w0.005` | 0 | +0.000184 | +0.000421 |
| 3 | `Q3_id342_sleep_boundary_sol_light_screen_motion_hgb_l5_l2_2_w0.005` | 0 | +0.000200 | +0.000352 |
| 4 | `Q3_id342_sleep_boundary_all_hgb_l5_l2_2_w0.005` | 0 | +0.000205 | +0.000383 |
| 5 | `Q3_id342_sleep_boundary_stress_hr_activity_screen_lr_c0_2_w0.005` | 0 | +0.000227 | +0.000439 |

### `S3`

| rank | config | wins | mean delta | max delta |
|---:|---|---:|---:|---:|
| 1 | `S3_id342_sleep_boundary_all_hgb_l5_l2_2_w0.005` | 1 | +0.000101 | +0.000260 |
| 2 | `S3_id342_sleep_boundary_sol_light_screen_motion_hgb_l5_l2_2_w0.005` | 1 | +0.000109 | +0.000243 |
| 3 | `S3_id342_sleep_boundary_stress_hr_activity_screen_hgb_l5_l2_2_w0.005` | 1 | +0.000121 | +0.000235 |
| 4 | `S3_id342_sleep_boundary_stress_hr_activity_screen_lr_c0_2_w0.005` | 0 | +0.000196 | +0.000288 |
| 5 | `S3_id342_sleep_boundary_all_hgb_l5_l2_2_w0.01` | 1 | +0.000209 | +0.000533 |

## Decision

- Selected targets: `[]`
- Sleep boundary has Q3/S3 signal: `False`
- Partial signal targets: `[]`
- Final gate pass: `False`
- Projected public proxy from ID342: `0.596083252`
- Gap to Public 0.55 after proxy: `+0.046083252`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If this gate fails, keep sleep-boundary/SOL features as closed ID342-aware Q3/S3 evidence. If both targets pass, use Kaggle no-submit for parity only.
