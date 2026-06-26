# Phase-2 raw multimodal execution plan — post ID341

Created: `2026-06-18 07:43:21 UTC+09:00`

This is report-only. It trains no model, creates no predictions or submission artifacts, and makes no Kaggle/DACON external action.

## Evidence read

```json
{
  "transfer_gate_closed": true,
  "sleep_boundary_low_movement_lead": true,
  "sleep_boundary_promotable": false,
  "combined_weak_best_proxy": 0.6014904915736343,
  "combined_weak_public_058": false,
  "residual_selected_targets": [
    "Q1",
    "S4"
  ],
  "residual_primary_wins": 3,
  "residual_projected_proxy": 0.6037839111413839,
  "residual_public_058": false,
  "place_selected_targets": [],
  "sensor_selected_targets": [],
  "phase2_ready": true
}
```

## Execution blocks

### 1. `A_q1_s4_sleep_residual_bridge`

The only repeated local raw/source signals after ID341 are sleep-boundary structure plus residual Q1/S4 drift. This block tests their common raw cause instead of blending their OOF outputs.

- Target focus: `Q1, S4`
- Raw inputs: `mScreenStatus, mActivity, mACStatus, mLight, wLight, wHr, wPedo`

Feature families:
- 12h-36h relative-night bins for screen/activity/charge/light/hr/pedo
- longest quiet interval and interruption count from screen-on + active + step spikes
- HR dip timing, morning-minus-night HR, and light/dark transitions
- fold-internal subject residuals of inferred onset/wake/recovery features
- Q1/S4-only compact model heads; all other targets stay anchored during this block

Selection gate:

```json
{
  "target_config_win_floor": 4,
  "assembled_primary_wins_floor": 5,
  "max_primary_delta_vs_id341_ceiling": 0.0,
  "projected_proxy_must_be_below": 0.6014113009,
  "public_058_review_cutoff": 0.59
}
```

Expected failure mode: Current residual gate wins only 3/5 primary splits and fails subject_hole/loso; if this raw bridge does not fix those two lenses, stop immediately.

### 2. `B_sleep_boundary_s2_recovery_retry`

S2 had residual smoke but failed the stricter target-restricted gate. Test whether raw recovery features explain S2 without borrowing Q1/S4 movement.

- Target focus: `S2`
- Raw inputs: `wHr, wPedo, mActivity, mLight, wLight, mScreenStatus`

Feature families:
- night HR volatility and min-HR timing
- low-step and stillness segment duration
- morning recovery slope and light exposure
- fold-internal subject residuals of recovery features

Selection gate:

```json
{
  "target_config_win_floor": 4,
  "mean_delta_vs_id341_must_be_negative": true,
  "assembled_primary_wins_floor": 5,
  "projected_proxy_must_be_below": 0.6014113009
}
```

Expected failure mode: If S2 remains 2/5 or worse, close S2 residual/recovery locally.

### 3. `C_env_social_routine_rebuild_only_if_A_or_B_passes`

Place/social novelty and sensor-adherence standalone lanes selected no targets, so this should not consume compute unless a sleep/residual block creates a stronger anchor first.

- Target focus: `Q2, Q3`
- Raw inputs: `mWifi, mBle, mGps, mUsageStats, mAmbience`

Feature families:
- home/routine entropy with hashed top-k buckets only
- app-category evening novelty, not direct app-name memorization
- ambience context ratios as weak modifiers

Selection gate:

```json
{
  "run_condition": "only after block A or B passes all-primary transfer pre-gate",
  "target_config_win_floor": 4,
  "no_high_joint_risk_warning": true
}
```

Expected failure mode: Do not widen local place/social scans; previous fixed-list parser smoke already failed.

## Remote runner contract

```json
{
  "compute": "Kaggle CPU only; GPU/TPU disabled",
  "max_runtime_plan": "single notebook revision, report-only, stop after block A if it fails all-primary gate",
  "allowed_outputs": [
    "reports/remote_phase2_raw_multimodal_execution_plan_post_id341_20260618.json/.md",
    "reports/remote_phase2_block_A_q1_s4_sleep_residual_bridge_20260618.json/.md",
    "reports/remote_phase2_block_B_s2_recovery_retry_20260618.json/.md only if block A has non-catastrophic transfer",
    "kaggle_outputs/phase2_raw_multimodal_manifest.json"
  ],
  "forbidden_outputs": [
    "submissions/*.csv",
    "artifacts/base_predictions/base_oof_id_*.csv",
    "artifacts/base_predictions/base_test_id_*.csv",
    "notebooks/history/submission_*.ipynb",
    "submission_log.csv row changes",
    "DACON submission"
  ],
  "stop_rules": [
    "Stop if block A does not beat ID341 on all primary splits.",
    "Stop if projected proxy remains above actual ID178 public 0.6014113009.",
    "Stop if any temporary source resembles ID287 high-movement/high-joint-risk geometry.",
    "Stop before creating ID342; artifact-contract review is separate even if gates pass."
  ],
  "block_order": [
    "A_q1_s4_sleep_residual_bridge",
    "B_sleep_boundary_s2_recovery_retry",
    "C_env_social_routine_rebuild_only_if_A_or_B_passes"
  ]
}
```

## Decision

- Phase-2 raw multimodal plan ready: `True`
- Recommended first block: `A_q1_s4_sleep_residual_bridge`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: Wire this plan into the no-submit Kaggle upstream runner only after explicit approval. Locally, do not create ID342 from residual/sleep/evening micro leads; use them only to implement block A.

