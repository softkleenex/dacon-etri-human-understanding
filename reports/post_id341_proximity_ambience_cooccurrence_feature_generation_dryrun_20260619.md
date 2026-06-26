# Post-ID341 proximity/ambience co-occurrence feature-generation dry run

Created: `2026-06-23 18:16:30 UTC+09:00`

This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Dry run passed: `True`
- Source family: `proximity_ambience_cooccurrence_v1`
- Target focus: `Q3,S3`
- Feature count: `80`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_proximity_ambience_cooccurrence_feature_matrix_smoke_contract`
- Next action: Dry run passed. A future no-approval step may only write a report-only feature-matrix smoke contract; do not write a feature matrix, train, create ID342, run Kaggle, or submit.

## Feature-frame summary

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
- row_alignment_passed: `True`
- schema_passed: `True`
- values_passed: `True`
- privacy_guard_passed: `True`
- does_not_write_feature_matrix: `True`
- does_not_train_model: `True`
- does_not_create_submission_artifact: `True`

## Privacy/leakage contract

- labels_loaded: `False`
- oof_or_test_predictions_loaded: `False`
- public_scores_used: `False`
- raw_proximity_identifier_columns_created: `False`
- raw_proximity_identifier_values_persisted: `False`
- feature_frame_persisted: `False`
- feature_matrix_written: `False`
- all_missing_values_filled_with_zero_or_flags: `True`
