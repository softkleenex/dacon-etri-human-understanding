# Sleep-boundary transition smoke — post ID341

Created: `2026-06-18 04:35:28 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Feature coverage

- Rows: `700` total / `450` train / `250` test
- Feature columns: `294` (`147` raw numeric + missing indicators)
- Mean missing rate: train `0.0687`, test `0.0673`

Top train missing features:

- `sbt_hr_std_night_24_30`: 0.8044
- `sbt_hr_evening_minus_night`: 0.8022
- `sbt_hr_morning_minus_night`: 0.7867
- `sbt_hr_min_night_24_30`: 0.7600
- `sbt_hr_mean_night_24_30`: 0.7600
- `sbt_hr_std_late_22_24`: 0.6644
- `sbt_hr_mean_late_22_24`: 0.6356
- `sbt_hr_min_late_22_24`: 0.6356
- `sbt_pedo_low_step_gap_hours`: 0.5044
- `sbt_pedo_step_sum_night_24_30`: 0.3889
- `sbt_pedo_has_step_ratio_night_24_30`: 0.3889
- `sbt_wlight_mean_night_24_30`: 0.3511

## Best by frozen split

| split | best candidate | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | `blend_id341_sleep_boundary_lr_c0_2_w0.01` | 0.538557 | 0.538584 | -0.000027 | 0.538638 | -0.000081 | `True` |
| `subject_timeseries_5` | `blend_id341_sleep_boundary_lr_c0_2_w0.03` | 0.539370 | 0.539688 | -0.000318 | 0.539747 | -0.000377 | `True` |
| `actual_test_band_holdout` | `blend_id341_sleep_boundary_lr_c0_05_w0.01` | 0.541417 | 0.541465 | -0.000048 | 0.541604 | -0.000187 | `True` |
| `test_boundary_neighbor2_holdout` | `blend_id341_sleep_boundary_hgb_l5_l2_2_w0.1` | 0.583912 | 0.586137 | -0.002225 | 0.586290 | -0.002378 | `True` |
| `test_boundary_neighbor5_holdout` | `blend_id341_sleep_boundary_lr_c0_05_w0.05` | 0.557063 | 0.557920 | -0.000857 | 0.557986 | -0.000923 | `True` |
| `loso_10` | `blend_id341_sleep_boundary_lr_c0_2_w0.01` | 0.538678 | 0.538584 | +0.000094 | 0.538638 | +0.000040 | `False` |

## Target-level smoke signal vs ID341 on primary splits

| target | primary wins | mean Δ | min Δ | max Δ |
|---|---:|---:|---:|---:|
| `Q1` | 5/5 | -0.003786 | -0.012668 | -0.000120 |
| `Q2` | 1/5 | -0.002294 | -0.011471 | +0.000000 |
| `Q3` | 1/5 | -0.000049 | -0.000244 | +0.000000 |
| `S1` | 4/5 | -0.005317 | -0.014509 | +0.000000 |
| `S2` | 5/5 | -0.003821 | -0.009045 | -0.000591 |
| `S3` | 1/5 | -0.000451 | -0.002255 | +0.000000 |
| `S4` | 5/5 | -0.008224 | -0.018214 | -0.001164 |

## Decision

- All primary splits beat ID341: `False`
- Primary split wins vs ID341: `4`
- Any target has three primary wins: `True`
- Source family has smoke signal: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Next action: If smoke signal is true, run a target-restricted v2 with geometry/transfer audit and no submission artifact. If false, keep sleep-boundary features as diagnostics and prioritize the Kaggle upstream rebuild.

