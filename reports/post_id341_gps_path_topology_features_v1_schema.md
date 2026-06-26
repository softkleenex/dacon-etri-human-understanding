# GPS path-topology feature matrix v1 schema — post ID341

Created: `2026-06-19 05:47:55 UTC+09:00`

This materializes only the registered label-free GPS topology feature matrix. It creates no model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Decision

- Feature matrix materialized: `True`
- Matrix written now: `True`
- Matrix path: `data/processed/post_id341_gps_path_topology_features_v1.parquet`
- Matrix SHA256: `c3915ce60754f9da5395550644753cf9f35966a2a88357e459c3c33fb04a3ae9`
- Source family: `gps_path_topology_sequence_v1`
- Target focus: `Q3, S3`
- Rows: `700`
- Feature columns: `32` / `64`
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
- feature_column_count_bounded: `True`
- no_target_or_prediction_columns: `True`
- all_feature_columns_numeric_or_bool: `True`
- no_missing_feature_values: `True`
- all_feature_values_finite: `True`
- split_values_expected: `True`
- coverage_passes: `True`
- cardinality_capped: `True`

## Coverage

| split | rows | GPS-ready rate | place-ready rate | median GPS rows |
|---|---:|---:|---:|---:|
| `train` | 450 | 0.9333 | 0.9756 | 1371.0 |
| `test` | 250 | 0.9360 | 0.9840 | 1376.0 |

## Feature columns

```json
[
  "gps_rows",
  "gps_payload_points",
  "gps_coordinate_payload_rows",
  "gps_active_hours",
  "gps_evening_rows_18_23",
  "gps_night_rows_00_05",
  "gps_gap_count_gt15m",
  "gps_longest_gap_minutes",
  "gps_observed_span_minutes",
  "gps_speed_p50",
  "gps_speed_p90",
  "gps_speed_max",
  "gps_low_speed_payload_share",
  "gps_high_speed_payload_share",
  "gps_lat_span",
  "gps_lon_span",
  "gps_bbox_area_proxy",
  "gps_payload_quality_flag",
  "activity_rows",
  "activity_active_code_count",
  "activity_top_code_share",
  "activity_late_rows_18_23",
  "wifi_rows",
  "wifi_token_count_capped64",
  "wifi_top_token_share",
  "wifi_payload_points",
  "ble_rows",
  "ble_token_count_capped64",
  "ble_top_token_share",
  "ble_payload_points",
  "place_context_rows",
  "place_context_token_count_capped128"
]
```

## Boundary

This is still not leaderboard evidence. It only parks a bounded, target-label-free feature matrix for later review; ID342/modeling/submission/Kaggle actions remain closed without a new explicit contract.

