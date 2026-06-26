# Post-ID341 usage semantic burst/Markov feature spec

Created: `2026-06-23 18:15:21 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Feature spec ready: `True`
- Source family: `usage_semantic_burst_markov_v1`
- Target focus: `['Q2', 'Q3']`
- Public `0.58` evidence: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_usage_semantic_burst_markov_feature_generation_dryrun_no_matrix`
- Next action: A future no-approval step may only run a report-only feature-generation dry run in memory. Do not write a feature matrix/parquet, train a model, create ID342, run Kaggle, or submit.

## State space and feature bound

- Predeclared app categories: `['communication', 'media_search', 'finance_shop', 'system_home', 'routine_health', 'mobility_context', 'other']`
- Pair-space size: `49`
- Raw app-name features allowed: `False`
- Target-guided vocabulary allowed: `False`
- Feature groups / columns / bound: `7` / `77` / `96`

## Feature groups

### `coverage_quality_flags`

- Sources: `['mUsageStats', 'mScreenStatus', 'mAmbience']`
- Column count: `7`
- Leakage guard: Only source-presence and match rates over the current lifelog day; no target labels or predictions.
- Columns:
  - `usage_row_count`
  - `usage_payload_row_rate`
  - `usage_missing_flag`
  - `screen_match_rate_5min`
  - `screen_missing_flag`
  - `ambience_match_rate_10min`
  - `source_coverage_score`

### `app_burst_session_shape`

- Sources: `['mUsageStats']`
- Column count: `14`
- Leakage guard: Sessions are cut by subject and lifelog_date; rolling variants may use only prior lifelog_date rows.
- Columns:
  - `usage_rows_total`
  - `usage_rows_evening`
  - `usage_rows_late`
  - `payload_event_count_total`
  - `usage_total_time_sum`
  - `usage_total_time_evening_sum`
  - `usage_total_time_late_sum`
  - `session_count`
  - `session_rows_median`
  - `session_rows_p90`
  - `max_session_rows`
  - `multi_category_session_count`
  - `evening_burst_rows_ge3`
  - `late_reactivation_rows_ge1`

### `semantic_category_distribution`

- Sources: `['mUsageStats']`
- Column count: `21`
- Leakage guard: Raw app names are mapped only through the fixed hand-written category function; no learned app vocabulary.
- Columns:
  - `category_time_share_communication`
  - `category_time_share_media_search`
  - `category_time_share_finance_shop`
  - `category_time_share_system_home`
  - `category_time_share_routine_health`
  - `category_time_share_mobility_context`
  - `category_time_share_other`
  - `category_row_presence_count_communication`
  - `category_row_presence_count_media_search`
  - `category_row_presence_count_finance_shop`
  - `category_row_presence_count_system_home`
  - `category_row_presence_count_routine_health`
  - `category_row_presence_count_mobility_context`
  - `category_row_presence_count_other`
  - `category_dominant_count_communication`
  - `category_dominant_count_media_search`
  - `category_dominant_count_finance_shop`
  - `category_dominant_count_system_home`
  - `category_dominant_count_routine_health`
  - `category_dominant_count_mobility_context`
  - `category_dominant_count_other`

### `markov_transition_shape`

- Sources: `['mUsageStats']`
- Column count: `10`
- Leakage guard: Transition pairs are over fixed categories only; subject baselines are fitted inside each outer fold.
- Columns:
  - `transition_count`
  - `unique_transition_pair_count`
  - `transition_entropy`
  - `evening_transition_count`
  - `evening_unique_transition_pair_count`
  - `self_transition_rate`
  - `top_pair_concentration`
  - `rare_pair_count`
  - `category_switch_rate`
  - `markov_surprise_vs_subject_baseline`

### `screen_coupled_bursts`

- Sources: `['mUsageStats', 'mScreenStatus']`
- Column count: `10`
- Leakage guard: Screen coupling is computed from raw timestamps only and must not condition on target or public score.
- Columns:
  - `screen_on_at_usage_rate`
  - `screen_state_change_count_near_usage`
  - `screen_on_burst_count`
  - `screen_off_burst_count`
  - `late_screen_on_usage_rate`
  - `evening_screen_on_usage_rate`
  - `usage_after_screen_on_count`
  - `usage_before_screen_off_count`
  - `screen_coupling_missing_rate`
  - `screen_reactivation_after_idle_count`

### `ambience_context_guards`

- Sources: `['mUsageStats', 'mAmbience']`
- Column count: `8`
- Leakage guard: Ambience features may gate burst context but cannot reopen a standalone ambience model lane.
- Columns:
  - `ambience_events_near_burst_count`
  - `ambience_unique_element_count_near_burst`
  - `quiet_high_switching_flag`
  - `noisy_late_reactivation_flag`
  - `ambience_missing_rate_near_usage`
  - `ambience_evening_context_count`
  - `ambience_late_context_count`
  - `ambience_guard_conflict_score`

### `subject_relative_rolling_baseline`

- Sources: `['mUsageStats', 'mScreenStatus']`
- Column count: `7`
- Leakage guard: In validation, rolling/baseline parameters are learned from the outer-train fold only; never from validation labels.
- Columns:
  - `rolling_7d_burst_count_z`
  - `rolling_7d_late_reactivation_z`
  - `rolling_7d_transition_entropy_z`
  - `rolling_7d_screen_on_rate_z`
  - `subject_history_day_count`
  - `subject_history_missing_flag`
  - `subject_relative_usage_intensity_rank`

## Validation/leakage contract

- Outer split: existing subject-hole/fold-internal evaluation only; do not pick candidates on full OOF rows
- Fit scope: fit any scaling/imputation/baseline statistics inside each outer-train fold only
- Rolling scope: subject-relative rolling features use only prior lifelog_date rows within the available raw stream
- Missingness policy: preserve missingness with explicit flags; do not drop expected train/test keys
- Forbidden inputs:
  - target labels
  - OOF/test predictions
  - public leaderboard score
  - raw app-name one-hot
  - learned app vocabulary
  - same-row sibling target labels

## Boundary

Passing this feature spec does not create ID342 and does not authorize feature generation beyond a future report-only in-memory dry run. No feature matrix, model, submission CSV, submission-log row, Kaggle run, DACON submission, or Public `0.58` claim is authorized.

