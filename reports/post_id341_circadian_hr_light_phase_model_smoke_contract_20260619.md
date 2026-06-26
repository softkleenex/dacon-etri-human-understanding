# Circadian HR/light model-smoke contract — post ID341

Created: `2026-06-19 11:38:23 UTC+09:00`

This is report-only. It trains no model and creates no predictions, OOF file, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Model-smoke contract ready: `True`
- Source family: `circadian_hr_light_phase_shape_v1`
- Target focus: `Q3, S3, S2`
- Model-smoke run allowed now: `False`
- Model-smoke run allowed next under contract: `False`
- Model-smoke already completed under contract: `True`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `close_circadian_phase_model_smoke_or_continue_usage_design`
- Next action: The bounded in-memory circadian model smoke already ran under contract. Do not rerun by default; continue only with the next queued report-only source design.

## Contract checks

### schema_contract

- schema_exists_and_parse_ok: `True`
- schema_diagnostic_matches: `True`
- schema_source_family_matches: `True`
- schema_target_focus_matches: `True`
- schema_matrix_materialized: `True`
- schema_boundary_passed: `True`
- schema_row_count_exact: `True`
- schema_feature_column_count_exact: `True`
- schema_matrix_path_matches: `True`
- schema_matrix_exists: `True`
- schema_score_actions_closed: `True`
- schema_safety_flags_closed: `True`

### feature_matrix_contract

- feature_matrix_contract_exists_and_parse_ok: `True`
- feature_matrix_contract_diagnostic_matches: `True`
- feature_matrix_contract_ready: `True`
- feature_matrix_materialized_under_contract: `True`
- feature_matrix_write_closed_now: `True`
- feature_matrix_contract_source_matches: `True`
- feature_matrix_contract_target_matches: `True`
- feature_matrix_contract_actions_closed: `True`

### queue_contract

- queue_exists_and_parse_ok: `True`
- queue_ready: `True`
- queue_contains_circadian_source: `True`
- queue_circadian_status_valid: `True`
- queue_step_allows_contract_or_future_smoke: `True`
- queue_no_local_score_route: `True`
- queue_score_route_requires_explicit_kaggle: `True`
- queue_public_058_false: `True`
- queue_actions_closed: `True`

### frontier_contract

- frontier_exists_and_parse_ok: `True`
- frontier_ready: `True`
- frontier_local_score_routes_closed: `True`
- frontier_no_approval_score_route_absent: `True`
- frontier_only_score_route_is_explicit_kaggle: `True`
- frontier_public_058_not_reviewable: `True`
- frontier_actions_closed: `True`

### public_gap_contract

- public_gap_exists_and_parse_ok: `True`
- public_gap_ready: `True`
- no_watch_candidate_passes_floor: `True`
- no_watch_candidate_passes_stress: `True`
- public_gap_actions_closed: `True`

### local_smoke_contract

- local_smoke_exists_and_parse_ok: `True`
- local_runner_smoke_fresh: `True`
- local_runner_smoke_rerun_not_required: `True`
- local_smoke_actions_closed: `True`

### existing_model_smoke_contract

- model_smoke_exists_and_parse_ok: `True`
- model_smoke_diagnostic_matches: `True`
- model_smoke_completed: `True`
- model_smoke_source_family_matches: `True`
- model_smoke_target_focus_matches: `True`
- model_smoke_actions_closed: `True`
- model_smoke_safety_flags_closed: `True`

## Allowed future model-smoke boundary

```json
{
  "allowed_future_script": "scripts/diagnose_post_id341_circadian_hr_light_phase_model_smoke.py",
  "allowed_matrix_path": "data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet",
  "allowed_schema_report_path": "reports/post_id341_circadian_hr_light_phase_features_v1_schema.json",
  "allowed_targets": [
    "Q3",
    "S3",
    "S2"
  ],
  "allowed_anchor_ids": [
    178,
    341
  ],
  "allowed_train_rows_exact": 450,
  "allowed_test_rows_present_but_unused": 250,
  "allowed_feature_columns_exact": 48,
  "allowed_model_scope": "tiny_fold_internal_in_memory_only",
  "allowed_model_families": [
    "fixed_logistic_or_ridge_like_linear_smoke",
    "fixed_hist_gradient_boosting_smoke"
  ],
  "must_use_predeclared_splits": true,
  "must_fit_preprocessing_inside_each_outer_train_fold": true,
  "must_compare_against_id178_and_id341_anchors": true,
  "must_report_targetwise_split_losses": true,
  "must_not_use_test_labels": true,
  "must_not_create_full_test_predictions": true,
  "must_not_write_oof_predictions": true,
  "must_not_persist_model": true,
  "must_not_create_submission_artifact": true,
  "must_not_modify_submission_log": true,
  "must_not_run_kaggle": true,
  "must_not_submit_dacon": true,
  "promotion_gate": "A later smoke may select targets only if it beats ID178/ID341 under frozen primary splits without opening a local-CV-only Public 0.58 claim; any submission artifact still requires a separate contract."
}
```

## Boundary

Passing this contract does not create ID342 and does not authorize a submission. It only allows a later bounded, in-memory fold-smoke report over the registered label-free circadian matrix.

