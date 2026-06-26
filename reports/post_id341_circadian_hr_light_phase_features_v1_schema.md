# Circadian HR/light feature matrix v1 schema — post ID341

Created: `2026-06-19 09:14:35 UTC+09:00`

This materializes only the registered label-free circadian HR/light feature matrix. It creates no model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Decision

- Feature matrix materialized: `True`
- Matrix written now: `True`
- Matrix path: `data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet`
- Matrix SHA256: `23ffdd124a46cd5f1e3d477f01433eff4123731030dd50ddefcdca565a7499e3`
- Source family: `circadian_hr_light_phase_shape_v1`
- Target focus: `Q3, S3, S2`
- Rows: `700`
- Feature columns: `48` / `48`
- Contract passed: `True`
- Allowed boundary passed: `True`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Feature matrix is registered as non-score infrastructure only. Do not create ID342, train a model, submit to DACON, or run Kaggle from this artifact without a separate approved contract.

## Validation checks

- row_count_exact: `True`
- row_alignment_exact: `True`
- duplicate_keys_absent: `True`
- train_rows_exact: `True`
- test_rows_exact: `True`
- key_columns_first: `True`
- feature_columns_exact_declared_order: `True`
- feature_column_count_exact: `True`
- feature_column_count_bounded: `True`
- no_target_or_prediction_columns: `True`
- all_feature_columns_numeric_or_bool: `True`
- no_missing_feature_values: `True`
- all_feature_values_finite: `True`
- split_values_expected: `True`
- coverage_passes: `True`

## Coverage

| split | rows | strong rate | weak rate | WHR covered |
|---|---:|---:|---:|---:|
| `train` | 450 | 0.5622 | 0.7489 | 0.8689 |
| `test` | 250 | 0.6120 | 0.8280 | 0.9800 |

## Feature columns

```json
[
  "phase_strong_day_flag",
  "phase_weak_day_flag",
  "whr_rows",
  "whr_hours",
  "wlight_hours",
  "mlight_hours",
  "wpedo_hours",
  "phase_missing_source_count",
  "hr_night_median",
  "hr_morning_median",
  "hr_afternoon_median",
  "hr_evening_median",
  "hr_day_night_amplitude",
  "hr_morning_rise_slope",
  "hr_evening_decay_slope",
  "hr_curve_roughness",
  "hr_rest_window_low_quantile",
  "hr_high_load_minutes_proxy",
  "wlight_first_nonzero_hour",
  "wlight_last_nonzero_hour",
  "wlight_morning_exposure_sum",
  "wlight_afternoon_exposure_sum",
  "wlight_evening_exposure_sum",
  "wlight_night_exposure_sum",
  "wlight_day_night_contrast",
  "wlight_evening_delay_ratio",
  "mlight_dark_minutes_proxy",
  "mlight_longest_dark_gap_hours",
  "mlight_night_light_burst_count",
  "mlight_evening_light_share",
  "mlight_morning_light_share",
  "mlight_wearable_light_disagreement",
  "steps_morning_sum",
  "steps_afternoon_sum",
  "steps_evening_sum",
  "steps_night_sum",
  "first_active_hour",
  "last_active_hour",
  "activity_fragmentation_count",
  "late_activity_ratio",
  "rest_window_length_proxy",
  "activity_phase_centroid_hour",
  "hr_light_phase_lag_proxy",
  "activity_light_phase_lag_proxy",
  "night_light_with_high_hr_minutes_proxy",
  "evening_activity_with_bright_light_proxy",
  "morning_light_without_activity_proxy",
  "strong_phase_cross_modal_consistency_score"
]
```

## Boundary

This is still not leaderboard evidence. It only parks a bounded, target-label-free feature matrix for later review; ID342/modeling/submission/Kaggle actions remain closed without a new explicit contract.

