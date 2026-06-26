# Circadian HR/light feature-matrix smoke contract — post ID341

Created: `2026-06-19 09:28:01 UTC+09:00`

This is report-only. It writes no feature matrix/parquet, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model, or DACON submission.

## Decision

- Feature-matrix smoke contract ready: `True`
- Source family: `circadian_hr_light_phase_shape_v1`
- Target focus: `Q3, S3, S2`
- Feature matrix write allowed now: `False`
- Feature matrix write allowed next under contract: `False`
- Feature matrix already materialized under contract: `True`
- Allowed future matrix path: `data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `validate_circadian_feature_matrix_then_stop_or_wait_for_explicit_kaggle_approval`
- Next action: The registered circadian feature matrix already exists and passed schema validation. Stop here unless a separate model-smoke contract is opened.

## Contract checks

### input_contract

- circadian_feature_generation_dryrun: `True`
- no_approval_frontier: `True`
- local_smoke_freshness: `True`
- prelaunch_gate: `True`

### dryrun_contract

- dryrun_passed: `True`
- source_family_matches: `True`
- target_focus_matches: `True`
- row_alignment_exact: `True`
- feature_columns_bounded: `True`
- coverage_passes: `True`
- dryrun_wrote_no_artifacts: `True`
- dryrun_all_action_flags_false: `True`

### frontier_contract

- frontier_ready: `True`
- local_score_routes_closed: `True`
- no_approval_score_route_absent: `True`
- score_route_requires_kaggle_approval: `True`
- public_058_not_reviewable: `True`
- frontier_all_action_flags_false: `True`

### smoke_contract

- local_smoke_fresh: `True`
- local_smoke_rerun_not_required: `True`
- prelaunch_ready_for_explicit_approval: `True`
- prelaunch_still_requires_approval: `True`
- prelaunch_artifact_review_closed: `True`
- smoke_and_prelaunch_action_flags_false: `True`

### existing_registered_output_contract

- schema_report_parse_ok: `True`
- schema_diagnostic_matches: `True`
- schema_source_family_matches: `True`
- schema_target_focus_matches: `True`
- schema_matrix_path_matches: `True`
- schema_matrix_exists: `True`
- schema_feature_matrix_materialized: `True`
- schema_allowed_boundary_passed: `True`
- schema_row_and_column_bounds_match: `True`
- schema_score_actions_closed: `True`
- schema_safety_flags_closed: `True`

### path_contract

- proposed_matrix_path_is_unique: `True`
- proposed_schema_path_is_unique: `True`
- proposed_matrix_absent_or_registered_existing: `True`
- proposed_schema_absent_or_registered_existing: `True`
- proposed_matrix_under_data_processed: `True`
- proposed_schema_under_reports: `True`
- proposed_matrix_name_is_source_specific: `True`
- proposed_schema_name_is_source_specific: `True`

## Allowed future write boundary

```json
{
  "allowed_matrix_path": "data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet",
  "allowed_schema_report_path": "reports/post_id341_circadian_hr_light_phase_features_v1_schema.json",
  "allowed_rows_exact": 700,
  "allowed_feature_columns_exact": 48,
  "allowed_feature_columns_max": 56,
  "required_key_columns": [
    "subject_id",
    "lifelog_date",
    "split"
  ],
  "required_feature_family": "circadian_hr_light_phase_shape_v1",
  "required_target_focus": [
    "Q3",
    "S3",
    "S2"
  ],
  "must_revalidate_after_write": true,
  "must_not_include_target_labels": true,
  "must_not_include_oof_predictions": true,
  "must_not_include_submission_predictions": true,
  "must_not_train_model": true,
  "must_not_create_submission_artifact": true,
  "must_not_modify_submission_log": true,
  "must_not_run_kaggle": true,
  "must_not_submit_dacon": true
}
```

## Boundary

Passing this contract only permits a future no-model feature-matrix writer for the registered circadian HR/light file. It is not leaderboard evidence, does not create ID342, and does not bypass the exact Kaggle approval phrase.

