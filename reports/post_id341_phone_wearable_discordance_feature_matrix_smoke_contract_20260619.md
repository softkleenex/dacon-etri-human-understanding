# Post-ID341 phone/wearable discordance feature-matrix smoke contract

Created: `2026-06-23 18:17:15 UTC+09:00`

This is report-only. It creates no feature matrix/parquet, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Contract ready: `True`
- Source family: `phone_wearable_discordance_v1`
- Target focus: `['Q2', 'Q3', 'S3']`
- Public `0.58` evidence: `False`
- Feature matrix write allowed now: `False`
- Feature matrix write allowed next under contract: `False`
- Feature matrix already materialized under contract: `True`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval`
- Next action: The registered matrix already exists; validate schema/report and stop unless explicit Kaggle approval is given.

## Registered paths

- Feature matrix path: `data/processed/post_id341_phone_wearable_discordance_features_v1.parquet`
- Schema report path: `reports/post_id341_phone_wearable_discordance_features_v1_schema.json`
- Feature matrix exists now: `True`
- Schema report exists now: `True`

## Dry-run basis

- Rows train/test/all: `450` / `250` / `700`
- Feature columns: `85`
- Finite values: `True`
- NaN count: `0`
- Zero-variance feature count: `0`

## Future writer boundary

```json
{
  "writer_allowed_next_under_contract": false,
  "allowed_feature_matrix_path": "data/processed/post_id341_phone_wearable_discordance_features_v1.parquet",
  "allowed_schema_report_path": "reports/post_id341_phone_wearable_discordance_features_v1_schema.json",
  "expected_row_count": 700,
  "expected_train_rows": 450,
  "expected_test_rows": 250,
  "expected_feature_column_count": 85,
  "required_key_columns": [
    "subject_id",
    "sleep_date",
    "lifelog_date",
    "split"
  ],
  "forbidden_columns": [
    "Q1",
    "Q2",
    "Q3",
    "S1",
    "S2",
    "S3",
    "S4",
    "prediction",
    "oof",
    "submission"
  ],
  "writer_must_stop_after_schema_report": true,
  "model_training_allowed": false,
  "submission_artifact_allowed": false,
  "kaggle_or_dacon_allowed": false
}
```

Passing this contract does not create ID342 and does not authorize modeling or submission. It only makes a later single registered label-free matrix writer reviewable under this exact path and schema boundary.

