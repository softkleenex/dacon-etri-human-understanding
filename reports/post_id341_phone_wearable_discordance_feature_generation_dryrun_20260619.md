# Post-ID341 phone/wearable discordance feature-generation dry run

Created: `2026-06-23 18:17:15 UTC+09:00`

This is report-only and in-memory. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Dry run passed: `True`
- Source family: `phone_wearable_discordance_v1`
- Target focus: `Q2,Q3,S3`
- Feature count: `85`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_phone_wearable_discordance_feature_matrix_smoke_contract`
- Next action: Dry run passed. A future no-approval step may only write a report-only feature-matrix smoke contract; do not write a feature matrix, train a model, create ID342, run Kaggle, or submit.

## Feature frame summary

- row_count: `700`
- train_rows: `450`
- test_rows: `250`
- column_count_including_keys: `89`
- feature_column_count: `85`
- expected_feature_column_count: `85`
- feature_columns_match_spec: `True`
- duplicate_key_rows: `0`
- finite_all_values: `True`
- nan_count: `0`
- feature_min: `0.0`
- feature_max: `24.0`
- zero_variance_feature_count: `0`
- nonzero_feature_count: `85`

## Checks

- spec_contract_ready: `True`
- queue_contract_ready: `True`
- frame_contract_ready: `True`
- feature_count_within_bound: `True`
- all_features_prefixed: `True`
- no_feature_matrix_written: `True`
- no_model_boundary: `True`

## Feature groups

- `phone_rest_state`: metrics `4`, windowed features `20`
- `wearable_activation_state`: metrics `4`, windowed features `20`
- `discordance_state`: metrics `4`, windowed features `20`
- `transition_hysteresis`: metrics `3`, windowed features `15`

