# Post-ID341 phone/wearable discordance feature matrix schema

Created: `2026-06-23 18:17:20 UTC+09:00`

This registers one label-free feature matrix only. It creates no model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Feature matrix materialized: `True`
- Source family: `phone_wearable_discordance_v1`
- Target focus: `Q2,Q3,S3`
- Row count: `700`
- Feature columns: `85`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `validate_phone_wearable_discordance_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval`
- Next action: Feature matrix materialized and schema-validated. Next local step may only validate deterministic rebuild/schema and stop; do not train, create ID342, run Kaggle, or submit.

## Artifact

- matrix_path: `data/processed/post_id341_phone_wearable_discordance_features_v1.parquet`
- schema_report_path: `reports/post_id341_phone_wearable_discordance_features_v1_schema.json`
- matrix_exists: `True`
- matrix_written_now: `False`
- matrix_size_bytes: `149048`
- matrix_sha256: `cc4eb270931c39da215d763c20e23d57d5d7f6b0a18c0f9834a1f9c8bc5b85e9`

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
- feature_min_nonnegative: `True`
- feature_max_bounded: `True`
- split_values_expected: `True`
- zero_variance_guard_bounded: `True`
- nonzero_feature_guard: `True`

## Contract checks

- contract_report_parse_ok: `True`
- contract_ready_or_already_materialized: `True`
- source_family_matches: `True`
- target_focus_matches: `True`
- matrix_path_matches_contract: `True`
- schema_path_matches_contract: `True`
- row_bound_matches_contract: `True`
- train_bound_matches_contract: `True`
- test_bound_matches_contract: `True`
- column_exact_matches_contract: `True`
- key_columns_match_contract: `True`
- writer_must_stop_after_schema_report: `True`
- no_model_allowed_by_contract: `True`
- no_submission_allowed_by_contract: `True`
- no_kaggle_or_dacon_allowed_by_contract: `True`
- decision_keeps_score_route_closed: `True`
- decision_keeps_forbidden_actions_closed: `True`
