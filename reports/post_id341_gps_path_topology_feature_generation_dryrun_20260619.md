# GPS path-topology feature-generation dry run — post ID341

Created: `2026-06-19 04:41:02 UTC+09:00`

This is report-only. It creates no feature matrix/parquet, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Dry run passed: `True`
- Source family: `gps_path_topology_sequence_v1`
- Target focus: `Q3, S3`
- Row alignment exact: `True`
- Feature columns: `32` / `64`
- Train/test GPS topology ready rates: `0.9333` / `0.9360`
- Public `0.58` evidence: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `feature_matrix_smoke_contract_or_kaggle_upstream_rebuild`
- Next action: GPS topology can be materialized in memory with aligned rows and bounded columns, but this remains non-score evidence. Do not create ID342 or train locally; either write a separate feature-matrix smoke contract or use the explicit-approved clean upstream Kaggle rebuild for score-relevant work.

## Row alignment

```json
{
  "expected_rows": 700,
  "dryrun_rows": 700,
  "exact_key_alignment": true,
  "duplicate_keys": 0,
  "train_rows": 450,
  "test_rows": 250
}
```

## Coverage

| split | rows | GPS-ready rows | GPS-ready rate | place-ready rate | median GPS rows |
|---|---:|---:|---:|---:|---:|
| `train` | 450 | 420 | 0.9333 | 0.9756 | 1371.0 |
| `test` | 250 | 234 | 0.9360 | 0.9840 | 1376.0 |

## Feature columns

- Declared dry-run feature count: `32`
- Within first-matrix bound: `True`

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

## Cardinality bounds

```json
{
  "wifi_cardinality": {
    "source": "mWifi",
    "prefix": "wifi",
    "cap": 64,
    "subjects": 10,
    "max_raw_tokens_per_subject": 24373,
    "subjects_over_cap_before_capping": 10,
    "max_tokens_after_capping": 64
  },
  "ble_cardinality": {
    "source": "mBle",
    "prefix": "ble",
    "cap": 64,
    "subjects": 10,
    "max_raw_tokens_per_subject": 119814,
    "subjects_over_cap_before_capping": 10,
    "max_tokens_after_capping": 64
  }
}
```

## Safety contract

```json
{
  "feature_matrix_written": false,
  "feature_parquet_written": false,
  "predictions_created": false,
  "submission_csv_created": false,
  "submission_log_modified": false,
  "model_trained": false,
  "kaggle_run_executed": false,
  "dacon_submission_executed": false,
  "uses_target_labels": false,
  "uses_oof_predictions": false,
  "uses_public_scores": false
}
```

## Boundary

This proves only that a GPS topology feature family can be aligned and bounded in memory. It is not leaderboard evidence, does not open ID342, and does not change the explicit-approval requirement for Kaggle or DACON actions.
