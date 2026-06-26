# Post-ID342 self-supervised subject-state remote intake

Created: `2026-06-24 20:46:04 UTC+09:00`

This is report-only. It checks pulled Kaggle no-submit outputs and refuses any submission-like artifact.

## Decision

- Remote intake ready: `True`
- Output present: `True`
- Remote report OK: `True`
- Feature gate report present: `True`
- Feature gate report OK: `True`
- Sequence gate report present: `True`
- Sequence gate report OK: `True`
- Runner manifest OK: `True`
- Unexpected submission-like artifacts present: `False`
- Create submission artifact: `False`
- DACON submit now: `False`
- Next action: Use this as no-submit research evidence only. Do not create a submission artifact; derive a separate supervised candidate only after a new artifact-contract review.

## Output Files

- `build/kaggle_outputs/self_supervised_subject_state_post_id342/dacon-etri-ss-subject-state-id342.log` (10582 bytes)
- `build/kaggle_outputs/self_supervised_subject_state_post_id342/kaggle_outputs/post_id342_self_supervised_subject_state_runner_manifest.json` (1247 bytes)
- `build/kaggle_outputs/self_supervised_subject_state_post_id342/kaggle_outputs/remote_post_id342_raw_subject_state_feature_gate_20260624.json` (142470 bytes)
- `build/kaggle_outputs/self_supervised_subject_state_post_id342/kaggle_outputs/remote_post_id342_raw_subject_state_feature_gate_20260624.md` (6051 bytes)
- `build/kaggle_outputs/self_supervised_subject_state_post_id342/kaggle_outputs/remote_post_id342_s3q3_raw_sequence_representation_gate_20260624.json` (191707 bytes)
- `build/kaggle_outputs/self_supervised_subject_state_post_id342/kaggle_outputs/remote_post_id342_s3q3_raw_sequence_representation_gate_20260624.md` (6155 bytes)
- `build/kaggle_outputs/self_supervised_subject_state_post_id342/kaggle_outputs/remote_post_id342_self_supervised_subject_state_no_submit_smoke_20260624.json` (8440 bytes)
- `build/kaggle_outputs/self_supervised_subject_state_post_id342/kaggle_outputs/remote_post_id342_self_supervised_subject_state_no_submit_smoke_20260624.md` (2357 bytes)

## Remote Report Checks

- smoke_passed: `True`
- no_submission_artifact_decision: `True`
- no_submission_log_decision: `True`
- no_dacon_submit_decision: `True`
- no_submission_created: `True`
- no_submission_artifact_created: `True`
- no_submission_log_row_created: `True`
- no_dacon_submission_made: `True`

## Feature Gate Report Checks

- reviewed_candidates_present: `True`
- no_feature_matrix_artifact_decision: `True`
- no_prediction_artifact_decision: `True`
- no_submission_artifact_decision: `True`
- no_submission_log_decision: `True`
- no_dacon_submit_decision: `True`
- no_feature_matrix_created: `True`
- no_predictions_created: `True`
- no_submission_created: `True`
- no_submission_log_row_created: `True`
- no_dacon_submission_made: `True`

## Sequence Gate Report Checks

- reviewed_candidates_present: `True`
- no_feature_matrix_artifact_decision: `True`
- no_prediction_artifact_decision: `True`
- no_submission_artifact_decision: `True`
- no_submission_log_decision: `True`
- no_dacon_submit_decision: `True`
- no_feature_matrix_created: `True`
- no_predictions_created: `True`
- no_submission_created: `True`
- no_submission_log_row_created: `True`
- no_dacon_submission_made: `True`

## Runner Manifest Checks

- remote_no_submit_mode: `True`
- no_submission_created: `True`
- no_submission_artifact_created: `True`
- no_submission_log_row_created: `True`
- no_dacon_submission_made: `True`
- output_file_list_has_no_forbidden_names: `True`

## Guardrails

- no_predictions_created: `True`
- no_submission_created: `True`
- no_base_prediction_artifact_created: `True`
- no_submission_log_row_created: `True`
- no_kaggle_action_made_by_this_script: `True`
- no_dacon_submission_made: `True`
