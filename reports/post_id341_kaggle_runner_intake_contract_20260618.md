# Post-ID341 Kaggle runner ↔ intake contract

Created: `2026-06-23 22:01:21 UTC+09:00`

This report is local and no-submit. It compares the runner notebook's remote report filenames with the local output-intake gate.

## Decision

- Runner/intake contract pass: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Runner staging matches the intake gate; explicit user approval is still required for Kaggle push/run.

## Counts

- intake_expected: `25`
- runner_generated: `25`
- runner_staged: `25`
- expected_runner_manifest: `1`
- path_assignments: `25`

## Runner manifest contract

- Expected manifest: `post_id341_upstream_rebuild_runner_manifest.json`
- Runner writes expected manifest: `True`
- Safety flags present: `True`
- Staged-files key present: `True`

## Notebook report-only safety contract

- No forbidden actions: `True`
- Required safety phrases present: `True`
- Report-only contract pass: `True`
- Forbidden matches: `{'csv_write_call': [], 'submission_or_base_output_path': [], 'submission_log_write': [], 'dacon_submit_api': [], 'kaggle_network_cli': [], 'forbidden_true_decision_flag': [], 'base_prediction_write': []}`
- Required safety phrases: `{'manifest_no_submission_created_true': True, 'manifest_no_dacon_submission_made_true': True, 'manifest_next_action_forbids_csv_submit': True, 'staging_uses_report_roots': True}`

## Q3/S3 negative-map handoff contract

- Remote negative map: `remote_post_id341_q3s3_negative_evidence_map_20260618.json`
- Remote remaining inventory: `remote_post_id341_q3s3_remaining_route_inventory_20260618.json`
- Negative map before inventory: `True`
- Inventory uses negative-map arg: `True`
- Negative map staged in report roots: `True`
- Asserts 8 closed lanes: `True`
- Handoff pass: `True`

## Mismatches

- missing_from_generation: `[]`
- missing_from_staging: `[]`
- unexpected_generated_reports: `[]`
- unexpected_staged_reports: `[]`
- unresolved_report_root_variables: `[]`
- staged_submission_like_files: `[]`

## Expected remote reports

- `remote_combined_sleep_evening_external_prior_gate_post_id341_20260618.json`
- `remote_external_etri_prior_smoke_post_id341_20260618.json`
- `remote_phase2_block_A_q1_proxy_gap_20260618.json`
- `remote_phase2_block_A_q1_s4_sleep_residual_bridge_20260618.json`
- `remote_phase2_block_B_s2_sleep_recovery_retry_20260618.json`
- `remote_phase2_raw_multimodal_execution_plan_post_id341_20260618.json`
- `remote_phase3_usage_temporal_microstructure_smoke_20260618.json`
- `remote_post_id341_clean_backbone_lane_closure_20260618.json`
- `remote_post_id341_clean_backbone_next_step_decision_20260618.json`
- `remote_post_id341_clean_backbone_rebuild_design_20260618.json`
- `remote_post_id341_clean_id221_feature_similarity_smoke_20260618.json`
- `remote_post_id341_clean_id221_inner_selection_probe_20260618.json`
- `remote_post_id341_clean_id226_calibration_smoke_20260618.json`
- `remote_post_id341_phase2_raw_source_execution_spec_20260618.json`
- `remote_post_id341_q2q3s3_local_fallback_chain_contract_20260618.json`
- `remote_post_id341_q2q3s3_source_hypothesis_matrix_20260618.json`
- `remote_post_id341_q3s3_fold_nested_sibling_bridge_smoke_20260619.json`
- `remote_post_id341_q3s3_negative_evidence_map_20260618.json`
- `remote_post_id341_q3s3_remaining_route_inventory_20260618.json`
- `remote_post_id341_self_supervised_subject_state_embedding_design_20260619.json`
- `remote_post_id341_transfer_gate_v2_joint_risk_20260618.json`
- `remote_post_id341_upstream_rebuild_blueprint_20260618.json`
- `remote_post_id341_upstream_rebuild_phase1_readiness_20260618.json`
- `remote_post_id341_upstream_source_priority_matrix_20260618.json`
- `remote_post_id341_usage_semantic_burst_markov_feature_matrix_validation_stop_20260619.json`

