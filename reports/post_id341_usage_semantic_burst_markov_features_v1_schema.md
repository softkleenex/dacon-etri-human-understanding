# Usage semantic burst/Markov feature matrix v1 schema — post ID341

Created: `2026-06-23 18:15:43 UTC+09:00`

This materializes only the registered label-free usage semantic burst/Markov feature matrix. It creates no model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Decision

- Feature matrix materialized: `True`
- Matrix written now: `False`
- Matrix path: `data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet`
- Matrix SHA256: `4782da045a9af4c688d47faf3bc9d8c0c926a3ce05116fc967e9e028af755251`
- Source family: `usage_semantic_burst_markov_v1`
- Target focus: `Q2, Q3`
- Rows: `700`
- Feature columns: `77` / `77`
- Contract passed: `True`
- Allowed boundary passed: `True`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Feature matrix is registered as non-score infrastructure only. Do not create ID342, train a model, submit to DACON, or run Kaggle from this artifact without a separate approved contract.

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
- no_raw_app_name_columns: `True`
- all_feature_columns_numeric_or_bool: `True`
- no_missing_feature_values: `True`
- all_feature_values_finite: `True`
- split_values_expected: `True`
- zero_variance_guard_bounded: `True`
- usage_coverage_passes: `True`
- screen_coupling_passes: `True`
- ambience_coupling_passes: `True`
- predeclared_category_space_used: `True`

## Coverage

| split | rows | usage present | screen match | ambience match | source coverage | late reactivation | evening burst |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train` | 450 | 0.9778 | 0.9759 | 0.9778 | 0.9852 | 0.9622 | 0.9556 |
| `test` | 250 | 1.0000 | 0.9987 | 1.0000 | 1.0000 | 0.9880 | 0.9960 |

## Feature columns

```json
[
  "usage_row_count",
  "usage_payload_row_rate",
  "usage_missing_flag",
  "screen_match_rate_5min",
  "screen_missing_flag",
  "ambience_match_rate_10min",
  "source_coverage_score",
  "usage_rows_total",
  "usage_rows_evening",
  "usage_rows_late",
  "payload_event_count_total",
  "usage_total_time_sum",
  "usage_total_time_evening_sum",
  "usage_total_time_late_sum",
  "session_count",
  "session_rows_median",
  "session_rows_p90",
  "max_session_rows",
  "multi_category_session_count",
  "evening_burst_rows_ge3",
  "late_reactivation_rows_ge1",
  "category_time_share_communication",
  "category_time_share_media_search",
  "category_time_share_finance_shop",
  "category_time_share_system_home",
  "category_time_share_routine_health",
  "category_time_share_mobility_context",
  "category_time_share_other",
  "category_row_presence_count_communication",
  "category_row_presence_count_media_search",
  "category_row_presence_count_finance_shop",
  "category_row_presence_count_system_home",
  "category_row_presence_count_routine_health",
  "category_row_presence_count_mobility_context",
  "category_row_presence_count_other",
  "category_dominant_count_communication",
  "category_dominant_count_media_search",
  "category_dominant_count_finance_shop",
  "category_dominant_count_system_home",
  "category_dominant_count_routine_health",
  "category_dominant_count_mobility_context",
  "category_dominant_count_other",
  "transition_count",
  "unique_transition_pair_count",
  "transition_entropy",
  "evening_transition_count",
  "evening_unique_transition_pair_count",
  "self_transition_rate",
  "top_pair_concentration",
  "rare_pair_count",
  "category_switch_rate",
  "markov_surprise_vs_subject_baseline",
  "screen_on_at_usage_rate",
  "screen_state_change_count_near_usage",
  "screen_on_burst_count",
  "screen_off_burst_count",
  "late_screen_on_usage_rate",
  "evening_screen_on_usage_rate",
  "usage_after_screen_on_count",
  "usage_before_screen_off_count",
  "screen_coupling_missing_rate",
  "screen_reactivation_after_idle_count",
  "ambience_events_near_burst_count",
  "ambience_unique_element_count_near_burst",
  "quiet_high_switching_flag",
  "noisy_late_reactivation_flag",
  "ambience_missing_rate_near_usage",
  "ambience_evening_context_count",
  "ambience_late_context_count",
  "ambience_guard_conflict_score",
  "rolling_7d_burst_count_z",
  "rolling_7d_late_reactivation_z",
  "rolling_7d_transition_entropy_z",
  "rolling_7d_screen_on_rate_z",
  "subject_history_day_count",
  "subject_history_missing_flag",
  "subject_relative_usage_intensity_rank"
]
```

## Boundary

This is still not leaderboard evidence. It only parks a bounded, target-label-free feature matrix for later review; ID342/modeling/submission/Kaggle actions remain closed without a new explicit contract.

