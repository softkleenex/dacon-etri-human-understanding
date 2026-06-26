# Post-ID341 phone/wearable discordance design

Created: `2026-06-23 18:17:10 UTC+09:00`

This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Design ready: `True`
- Source family: `phone_wearable_discordance_v1`
- Target focus: `Q2,Q3,S3`
- Coverage ready: `True`
- Discordance observed: `True`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_phone_wearable_discordance_feature_spec_no_matrix`
- Next action: Treat this as a new report-only source design for the approved upstream rebuild packet. A later step may write only a feature spec; do not create a matrix, train, create ID342, run Kaggle, or submit.

## Context checks

- all_context_reports_present: `True`
- local_score_routes_closed_before_design: `True`
- sensor_adherence_standalone_closed: `True`
- sleep_boundary_is_lead_not_artifact: `True`
- circadian_matrix_model_closed: `True`
- usage_matrix_validation_stopped: `True`
- proximity_matrix_validation_stopped: `True`
- operator_handoff_waiting_for_approval: `True`

## Source coverage

- `screen`: train day rate `1.0000`, test day rate `1.0000`, observed hour bins `16214`
- `charge`: train day rate `1.0000`, test day rate `1.0000`, observed hour bins `16214`
- `activity`: train day rate `1.0000`, test day rate `1.0000`, observed hour bins `16214`
- `pedo`: train day rate `0.8978`, test day rate `0.9960`, observed hour bins `13499`
- `hr`: train day rate `0.8689`, test day rate `0.9800`, observed hour bins `8098`
- `wearable_light`: train day rate `0.9200`, test day rate `1.0000`, observed hour bins `14003`
- `phone_light`: train day rate `1.0000`, test day rate `1.0000`, observed hour bins `16184`

## Discordance feasibility

- Phone-rest hour bins: `14862`
- Wearable-active hour bins: `14038`
- `screen_off_step_active`: bins `1090`, train rate `0.0625`, test rate `0.0692`
- `charging_step_active`: bins `783`, train rate `0.0460`, test rate `0.0477`
- `phone_still_hr_present`: bins `6596`, train rate `0.3790`, test rate `0.4172`
- `phone_rest_wearable_active`: bins `12600`, train rate `0.7223`, test rate `0.7998`
- `screen_off_wearable_light_present`: bins `5529`, train rate `0.3143`, test rate `0.3558`

## Feature groups

- `phone_rest_state`: `screen_off_rate, charging_rate, phone_still_rate, screen_off_charging_bins`
- `wearable_activation_state`: `step_active_rate, step_sum_log1p, hr_observed_rate, wearable_light_log1p`
- `discordance_state`: `screen_off_step_active_bins, charging_step_active_bins, phone_rest_wearable_active_bins, phone_dark_wearable_light_bins`
- `transition_hysteresis`: `last_phone_active_hour, first_wearable_active_hour_after_phone_rest, wearable_after_phone_rest_gap_hours`

## Leakage guards

- Build future features only from each row's lifelog_date timestamps, never from target labels or public scores.
- Use fixed discordance definitions; do not select phone/wearable thresholds from outer validation rows.
- Any model smoke must learn imputation/scaling/selection inside outer folds only.
- This design cannot create ID342, a base prediction artifact, a notebook-history artifact, or a submission-log row.

