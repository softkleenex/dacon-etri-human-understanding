# Post-ID341 Q2/Q3/S3 env-social-routine v2 design

Created: `2026-06-23 18:07:58 UTC+09:00`

This is report-only. It trains no model and creates no predictions, submission CSV, notebook/history artifact, Kaggle run, or DACON submission.

## Evidence read

- route_primary: `explicit_approved_clean_upstream_id221_to_id226`
- route_fallback: `local_or_kaggle_q2q3s3_env_social_routine_v2_design`
- missing_targets: `['Q2', 'Q3', 'S3']`
- q1_s4_only_is_insufficient: `True`
- local_cv_needed_for_public_058_recent_floor: `0.517445`
- local_cv_needed_for_public_058_recent_stress: `0.494927`
- local_source_stack_closed: `True`
- usage_temporal_selected_targets: `[]`
- place_social_selected_targets: `[]`
- sensor_adherence_selected_targets: `[]`
- data_source_coverage_ready: `True`

## Raw source profiles

| source | rows | subjects | dates | size MB | cardinality highlights |
|---|---:|---:|---:|---:|---|
| `mWifi` | 76336 | 10 | 166 | 3.799 | `m_wifi(bssid=125239, rssi=99)` |
| `mBle` | 21830 | 10 | 166 | 3.878 | `m_ble(address=350456, device_class=18, rssi=103)` |
| `mGps` | 800611 | 10 | 166 | 68.592 | `` |
| `mUsageStats` | 45197 | 10 | 160 | 0.955 | `m_usage_stats(app_name=586, total_time=102266)` |
| `mAmbience` | 476577 | 10 | 166 | 14.155 | `m_ambience(labels=517)` |
| `mActivity` | 961062 | 10 | 166 | 4.427 | `m_activity=6` |
| `mScreenStatus` | 939653 | 10 | 166 | 4.359 | `m_screen_use=2` |

## Feature blocks

### `routine_stability_clock_v2`

- Target focus: `Q2, Q3, S3`
- Raw inputs: `mScreenStatus, mActivity, mUsageStats`
- Why Q2/Q3/S3: Q2/Q3/S3 need behavioral regularity and mental-state context rather than another sleep-boundary overlay; these features model routine stability without borrowing label-derived OOF movement.

Feature families:
- hourly screen-on/off rhythm distance from subject training-fold routine
- activity-state transition entropy by morning/day/evening/night bins
- weekday-adjusted app-use total-time rhythm without direct app-name model columns
- rolling 3-day and 7-day routine deviation fitted inside each outer fold

### `place_social_novelty_v2`

- Target focus: `Q2, Q3, S3`
- Raw inputs: `mWifi, mBle, mGps`
- Why Q2/Q3/S3: Previous place/social standalone smoke selected no target, so v2 must combine novelty, regularity, and daypart transitions instead of direct BSSID/address memorization.

Feature families:
- Wi-Fi/BLE novelty counts using fold-trained top-k or hashing buckets only
- home-like anchor consistency inferred from within-subject frequent place buckets
- GPS movement radius, speed quantiles, and stationary ratio by daypart
- place-transition entropy and next-day smoothness features available without labels

### `ambient_load_recovery_context_v2`

- Target focus: `Q2, Q3`
- Raw inputs: `mAmbience, mUsageStats, mActivity`
- Why Q2/Q3/S3: Q2/Q3 are the missing questionnaire targets most plausibly tied to social/context load. This block is deliberately Q2/Q3-first and should not be evaluated as a generic all-target tuner.

Feature families:
- speech/music/silence ratios by daypart from ambience element
- late-day app-use load and app category entropy using fold-local category maps
- ambient-context change points before sleep-date boundary
- interaction-free single-family summaries first; no broad stacker until split evidence exists

## Validation contract

- Outer splits required: `subject_hole_5, actual_test_band_holdout, test_boundary_neighbor2_holdout, test_boundary_neighbor5_holdout, loso_10`
- Fit inside fold only:
  - top-k Wi-Fi/BLE/app categories
  - hash bucket dictionaries when used
  - subject routine baselines
  - scalers/imputers
  - feature selection and target selection
- Forbidden shortcuts:
  - direct BSSID/address/app-name one-hot columns fitted on full train+test
  - full-OOF target selection
  - public-LB-guided thresholding
  - external row concatenation
  - creating ID342 from a local-only feature design

## Decision

- Design ready: `True`
- Target focus: `Q2, Q3, S3`
- Next local allowed step: `feature_generation_metadata_smoke_only`
- Next Kaggle allowed step: `only_after_explicit_no_submit_approval`
- Promotion target set must include: `Q2, Q3, S3`
- Projected proxy must beat ID178 public: `0.6014113009`
- Public-0.58 review cutoff: `0.59`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Implement only a Q2/Q3/S3 env-social-routine feature-generation smoke next, or run the approved clean upstream rebuild if the user gives the exact Kaggle no-submit approval phrase. Do not create a submission artifact from this design.

