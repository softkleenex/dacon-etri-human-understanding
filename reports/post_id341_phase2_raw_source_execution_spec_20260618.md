# Post-ID341 phase-2 raw/sequence source execution spec

Created: `2026-06-18 08:43:23 UTC+09:00`

This is report-only. It trains no model, creates no predictions or submission artifacts, and makes no Kaggle/DACON external action.

## Closure read

```json
{
  "phase1_ready": true,
  "local_new_source_stack_closed": true,
  "existing_artifact_promotable": false,
  "clean_backbone_closed": true,
  "clean_backbone_next_lane": "stronger_raw_or_sequence_source_model",
  "transfer_v2_public_058_candidate": false,
  "combined_external_best_proxy": 0.6014904915736343,
  "combined_external_public_058": false,
  "new_source_blueprint_top_family": "sleep_boundary_transition_v1"
}
```

## Data readiness

- Processed train/test: `[450, 242]` / `[250, 242]`
- Raw files: `12`
- Raw total rows: `6122179`
- Raw total size MB: `122.832`

## Stage contract

## Optional local smoke evidence

```json
{
  "subject_residual_sequence_drift": {
    "path": "reports/subject_residual_sequence_drift_smoke_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": true,
      "selected_targets": null,
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": false,
      "primary_split_wins_vs_id341": 1,
      "primary_wins_vs_id341": null,
      "mean_primary_delta_vs_id341": -1.3681439806112471e-05,
      "projected_public_proxy_from_id341": 0.603816102060194,
      "public_058_reviewable": false,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "subject_residual_target_restricted_gate": {
    "path": "reports/subject_residual_target_restricted_gate_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": [
        "Q1",
        "S4"
      ],
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": false,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": 3,
      "mean_primary_delta_vs_id341": -4.587235861621686e-05,
      "projected_public_proxy_from_id341": 0.6037839111413839,
      "public_058_reviewable": false,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "phase2_raw_multimodal_execution_plan": {
    "path": "reports/phase2_raw_multimodal_execution_plan_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": null,
      "phase2_raw_multimodal_plan_ready": true,
      "recommended_first_block": "A_q1_s4_sleep_residual_bridge",
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": null,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": null,
      "mean_primary_delta_vs_id341": null,
      "projected_public_proxy_from_id341": null,
      "public_058_reviewable": null,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "phase2_block_a_q1_s4_sleep_residual_bridge": {
    "path": "reports/phase2_block_a_q1_s4_sleep_residual_bridge_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": [
        "Q1"
      ],
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": false,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": true,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": 5,
      "mean_primary_delta_vs_id341": -0.0002127085031439435,
      "projected_public_proxy_from_id341": 0.6036170749968561,
      "public_058_reviewable": false,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "phase2_block_a_q1_proxy_gap": {
    "path": "reports/phase2_block_a_q1_proxy_gap_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": null,
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": true,
      "q1_signal_sufficient_for_id178_proxy": false,
      "additional_independent_target_signal_required": true,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": null,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": null,
      "mean_primary_delta_vs_id341": null,
      "projected_public_proxy_from_id341": null,
      "public_058_reviewable": null,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "phase2_block_b_s2_sleep_recovery_retry": {
    "path": "reports/phase2_block_b_s2_sleep_recovery_retry_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": [],
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": false,
      "final_block_b_gate_pass": false,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": false,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": 0,
      "mean_primary_delta_vs_id341": 0.0,
      "projected_public_proxy_from_id341": 0.6038297835,
      "public_058_reviewable": false,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "phase3_usage_temporal_microstructure": {
    "path": "reports/phase3_usage_temporal_microstructure_smoke_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": [],
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": false,
      "final_usage_temporal_gate_pass": false,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": false,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": 0,
      "mean_primary_delta_vs_id341": 0.0,
      "projected_public_proxy_from_id341": 0.6038297835,
      "public_058_reviewable": false,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "phase3_ambience_context_closure": {
    "path": "reports/phase3_ambience_context_closure_post_id341_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": null,
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": false,
      "ambience_has_nested_positive_signal": false,
      "local_source_stack_closed_for_submission_artifact": null,
      "any_local_lane_promotable": null,
      "all_primary_splits_beat_id341": null,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": null,
      "mean_primary_delta_vs_id341": null,
      "projected_public_proxy_from_id341": null,
      "public_058_reviewable": false,
      "create_submission_artifact": false,
      "submit_now": false
    }
  },
  "post_id341_local_source_closure_synthesis": {
    "path": "reports/post_id341_local_source_closure_synthesis_20260618.json",
    "exists": true,
    "summary": {
      "source_family_has_smoke_signal": null,
      "selected_targets": null,
      "phase2_raw_multimodal_plan_ready": null,
      "recommended_first_block": null,
      "final_block_a_gate_pass": null,
      "q1_all_primary_signal_confirmed": null,
      "q1_signal_sufficient_for_id178_proxy": null,
      "additional_independent_target_signal_required": null,
      "s2_has_config_signal": null,
      "final_block_b_gate_pass": null,
      "usage_temporal_has_config_signal": null,
      "final_usage_temporal_gate_pass": null,
      "ambience_reopen_now": null,
      "ambience_has_nested_positive_signal": null,
      "local_source_stack_closed_for_submission_artifact": true,
      "any_local_lane_promotable": false,
      "all_primary_splits_beat_id341": null,
      "primary_split_wins_vs_id341": null,
      "primary_wins_vs_id341": null,
      "mean_primary_delta_vs_id341": null,
      "projected_public_proxy_from_id341": null,
      "public_058_reviewable": false,
      "create_submission_artifact": false,
      "submit_now": false
    }
  }
}
```

### `remote_latest_closure_parity`

Regenerate transfer-v2, external-prior, and combined weak-source closure reports in Kaggle runtime.

- Compute: Kaggle CPU
- Creates submission: `False`

Allowed outputs:
- `remote_post_id341_transfer_gate_v2_joint_risk_20260618.json/.md`
- `remote_external_etri_prior_smoke_post_id341_20260618.json/.md`
- `remote_combined_sleep_evening_external_prior_gate_post_id341_20260618.json/.md`

Stop if:
- any report creates or requests a submission artifact
- any report unexpectedly finds a Public 0.58 candidate; stop for manual review instead of auto-promoting

### `raw_multimodal_sequence_source_smoke`

Build compact sequence/source summaries from raw event streams rather than recombining existing OOF prediction artifacts.

- Compute: Kaggle CPU first; GPU/TPU disabled
- Creates submission: `False`

| source | raw inputs | target focus | representation |
|---|---|---|---|
| `nightly_multimodal_bins_v1` | `mScreenStatus, mActivity, mACStatus, mLight, wLight, wHr, wPedo` | `Q1, S1, S2, S3, S4` | 12h-36h relative-night bins with screen/activity/charge/light/hr/pedo transitions |
| `environment_social_routine_bins_v1` | `mWifi, mBle, mUsageStats, mGps, mAmbience` | `Q2, Q3, S4` | hourly novelty/routine entropy and app/env context summaries, no BSSID direct memorization |
| `subject_residual_sequence_drift_v1` | `processed numeric features, calendar covariates` | `Q1, Q2, Q3, S1, S2, S3, S4` | fold-internal per-subject residual trend and volatility summaries fitted without valid rows |

Allowed outputs:
- `reports/remote_phase2_raw_source_smoke_metrics_20260618.json/.md`
- `reports/remote_phase2_raw_source_candidate_geometry_20260618.json/.md`
- `kaggle_outputs/phase2_raw_source_manifest.json`

Forbidden outputs:
- `submissions/*.csv`
- `artifacts/base_predictions/base_oof_id_*.csv`
- `artifacts/base_predictions/base_test_id_*.csv`
- `notebooks/history/submission_*.ipynb`
- `submission_log.csv row changes`

### `transfer_and_joint_risk_gate`

Evaluate any temporary source predictions against ID178/287/329/341 and joint-risk metrics before artifact review.

- Compute: Kaggle CPU or local after pulled outputs
- Creates submission: `False`

## Gate invariants

```json
{
  "no_dacon_submission": true,
  "no_kaggle_push_without_explicit_user_approval": true,
  "no_submission_csv_before_artifact_contract": true,
  "unique_filename_required_for_any_future_artifact": true,
  "full_artifact_contract_required_before_submission_file_promotion": [
    "submission CSV",
    "notebook/history",
    "metadata JSON",
    "decision MD",
    "base_oof",
    "base_test",
    "submission_log row"
  ],
  "public_lb_not_used_for_selection": true,
  "outer_valid_rows_score_only": true
}
```

## Decision

- Phase-2 spec ready: `True`
- Recommended next revision: `remote_phase2_raw_multimodal_sequence_source_smoke`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: After explicit approval and pulled green phase-1 runner reports, extend the Kaggle runner only with the stage-contract raw/sequence source smoke. Do not create ID342 unless the pulled temporary source passes transfer and joint-risk gates plus a separate artifact-contract review.

