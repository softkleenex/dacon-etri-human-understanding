# Post-ID342 subject-state raw schema contract

Created: `2026-06-24 09:26:45 UTC+09:00`

This report reads parquet metadata only. It creates no token cache, feature matrix, embedding, prediction, Kaggle run, submission artifact, log row, or DACON submission.

## Decision

- Raw schema contract ready: `True`
- Domain/token contract ready: `True`
- Metadata-only read: `True`
- Raw payload rows read: `0`
- Primary score targets: `['Q3', 'S3']`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Required phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`
- Next action: Raw schemas match the domain-token contract; implement no-submit tokenization only after exact approval.

## Schema Summary

```json
{
  "expected_modality_count": 12,
  "metadata_only_read": true,
  "missing_files": [],
  "present_modality_count": 12,
  "schema_failures": [],
  "schema_ok_count": 12,
  "total_rows_from_metadata_only": 6122179
}
```

## Raw Schema Rows

| modality | rows metadata | columns | schema ok | missing columns | token role |
|---|---:|---|---:|---|---|
| `mACStatus` | `939896` | `['subject_id', 'timestamp', 'm_charging']` | `True` | `[]` | `charge_bedside_routine` |
| `mActivity` | `961062` | `['subject_id', 'timestamp', 'm_activity']` | `True` | `[]` | `phone_activity_state` |
| `mAmbience` | `476577` | `['subject_id', 'timestamp', 'm_ambience']` | `True` | `[]` | `ambient_sound_context` |
| `mBle` | `21830` | `['subject_id', 'timestamp', 'm_ble']` | `True` | `[]` | `near_device_social_context` |
| `mGps` | `800611` | `['subject_id', 'timestamp', 'm_gps']` | `True` | `[]` | `mobility_place_routine` |
| `mLight` | `96258` | `['subject_id', 'timestamp', 'm_light']` | `True` | `[]` | `phone_light_context` |
| `mScreenStatus` | `939653` | `['subject_id', 'timestamp', 'm_screen_use']` | `True` | `[]` | `sleep_boundary_phone_use` |
| `mUsageStats` | `45197` | `['subject_id', 'timestamp', 'm_usage_stats']` | `True` | `[]` | `phone_app_load` |
| `mWifi` | `76336` | `['subject_id', 'timestamp', 'm_wifi']` | `True` | `[]` | `place_stability` |
| `wHr` | `382918` | `['subject_id', 'timestamp', 'heart_rate']` | `True` | `[]` | `wearable_recovery_arousal` |
| `wLight` | `633741` | `['subject_id', 'timestamp', 'w_light']` | `True` | `[]` | `watch_light_context` |
| `wPedo` | `748100` | `['subject_id', 'timestamp', 'step', 'step_frequency', 'running_step', 'walking_step', 'distance', 'speed', 'burned_calories']` | `True` | `[]` | `wearable_exertion_inactivity` |

## Forbidden Actions Now

```json
{
  "create_predictions": false,
  "create_submission_artifact": false,
  "fit_embedding": false,
  "modify_submission_log": false,
  "push_kaggle": false,
  "read_parquet_row_payloads": false,
  "run_kaggle": false,
  "submit_dacon": false,
  "train_downstream_model": false,
  "write_feature_matrix": false,
  "write_token_cache": false
}
```
