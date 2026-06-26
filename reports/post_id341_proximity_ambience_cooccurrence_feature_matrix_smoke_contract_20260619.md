# Post-ID341 proximity/ambience co-occurrence feature-matrix smoke contract

Created: `2026-06-23 18:16:30 UTC+09:00`

This is report-only. It writes no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Contract ready: `True`
- Source family: `proximity_ambience_cooccurrence_v1`
- Target focus: `Q3,S3`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix write allowed now: `False`
- Feature matrix write allowed next under contract: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `validate_proximity_ambience_cooccurrence_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval`
- Next action: The registered matrix already exists; validate schema/report and stop unless explicit Kaggle approval is given.

## Registered paths

- feature_matrix_path: `data/processed/post_id341_proximity_ambience_cooccurrence_features_v1.parquet`
- schema_report_path: `reports/post_id341_proximity_ambience_cooccurrence_features_v1_schema.json`
- feature_matrix_exists_now: `True`
- schema_report_exists_now: `True`

## Dry-run summary

- row_count: `700`
- train_rows: `450`
- test_rows: `250`
- column_count_including_keys: `84`
- feature_column_count: `80`
- expected_feature_column_count: `80`
- feature_columns_match_spec: `True`
- duplicate_key_rows: `0`
- finite_all_values: `True`
- nan_count: `0`
- feature_min: `0.0`
- feature_max: `5413.0`
- zero_variance_feature_count: `1`
- nonzero_feature_count: `80`

## Checks

- spec_contract_ready: `True`
- dryrun_contract_ready: `True`
- dryrun_schema_ready: `True`
- queue_contract_ready: `True`
- current_queue_contract_ready: `False`
- queue_records_contract_ready: `False`
- queue_records_matrix_materialized: `False`
- queue_records_lane_closed: `True`
- privacy_contract_ready: `True`
- registered_matrix_absent_before_write: `False`
- registered_schema_absent_before_write: `False`
- feature_matrix_write_allowed_next_under_contract: `False`
- does_not_write_feature_matrix: `True`
- does_not_train_model: `True`
- does_not_create_submission_artifact: `True`

## Future writer contract

- writer_allowed_next_under_contract: `False`
- allowed_feature_matrix_path: `data/processed/post_id341_proximity_ambience_cooccurrence_features_v1.parquet`
- allowed_schema_report_path: `reports/post_id341_proximity_ambience_cooccurrence_features_v1_schema.json`
- expected_row_count: `700`
- expected_train_rows: `450`
- expected_test_rows: `250`
- expected_feature_column_count: `80`
- required_key_columns: `['subject_id', 'sleep_date', 'lifelog_date', 'split']`
- forbidden_columns: `['Q1', 'Q2', 'Q3', 'S1', 'S2', 'S3', 'S4', 'prediction', 'oof', 'submission']`
- writer_must_stop_after_schema_report: `True`
- model_training_allowed: `False`
- submission_artifact_allowed: `False`
- kaggle_or_dacon_allowed: `False`
