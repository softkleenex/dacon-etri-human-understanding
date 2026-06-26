# Post-ID341 phone/wearable discordance feature spec

Created: `2026-06-23 18:17:10 UTC+09:00`

This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Feature spec ready: `True`
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
- Next local allowed step: `report_only_phone_wearable_discordance_feature_generation_dryrun_no_matrix`
- Next action: Feature spec is ready. A future no-approval step may only run an in-memory feature-generation dry run that checks 700-row alignment and bounded finite columns; do not write a feature matrix, train a model, create ID342, run Kaggle, or submit.

## Row contract

- key_columns: `['subject_id', 'sleep_date', 'lifelog_date']`
- expected_train_rows: `450`
- expected_test_rows: `250`
- expected_total_rows: `700`
- label_columns_allowed: `False`
- prediction_columns_allowed: `False`
- submission_columns_allowed: `False`
- raw_timestamp_columns_allowed: `False`

## Feature groups

- `phone_rest_state`: `screen_off_rate, charging_rate, phone_still_rate, screen_off_charging_bins`
- `wearable_activation_state`: `step_active_rate, step_sum_log1p, hr_observed_rate, wearable_light_log1p`
- `discordance_state`: `screen_off_step_active_bins, charging_step_active_bins, phone_rest_wearable_active_bins, phone_dark_wearable_light_bins`
- `transition_hysteresis`: `last_phone_active_hour, first_wearable_active_hour_after_phone_rest, wearable_after_phone_rest_gap_hours`

## Global features

- `pwd_global_day_source_coverage_index`
- `pwd_global_phone_rest_total_hour_bins`
- `pwd_global_wearable_active_total_hour_bins`
- `pwd_global_discordance_total_hour_bins`
- `pwd_global_discordance_to_phone_rest_ratio`
- `pwd_global_late_discordance_share`
- `pwd_global_first_discordance_hour`
- `pwd_global_last_discordance_hour`
- `pwd_global_max_contiguous_discordance_hours`
- `pwd_global_source_missingness_guard_count`

## Schema bounds

- feature_prefix: `pwd_`
- max_feature_columns: `96`
- actual_feature_columns: `85`
- all_features_prefixed: `True`
- duplicate_feature_count: `0`
- forbidden_feature_token_hits: `[]`
- raw_identifier_columns_allowed: `False`
- raw_timestamp_columns_allowed: `False`
- fold_internal_preprocessing_required_for_any_model: `True`

## Aggregation contract

- join_grain: subject_id + lifelog_date + fixed day window
- hour_grid: 24 fixed hourly bins per row; windows are deterministic and predeclared.
- phone_rest_definition: screen-off, charging, or still-phone state from raw phone sensors only.
- wearable_activation_definition: positive steps, observed heart-rate, or wearable light from raw wearable sensors only.
- discordance_definition: fixed cross-device disagreement masks; no threshold fitted to labels.
- missing_value_policy: write explicit source coverage/missingness guards; later imputation must be fold-internal.

## Leakage guards

- Build future features only from each row's lifelog_date raw sensor timestamps.
- Do not use target labels, public LB scores, OOF predictions, submission files, or base prediction artifacts.
- Keep phone/wearable thresholds fixed from the design report; do not tune them on validation residuals.
- Any later feature-generation dry run may stay in memory only unless a separate matrix contract is written.
- Any future model smoke must fit preprocessing, imputation, selection, and calibration inside each outer fold.
- This spec cannot create ID342, a notebook-history artifact, a base prediction artifact, or a submission-log row.

