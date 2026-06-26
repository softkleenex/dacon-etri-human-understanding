# Post-ID341 upstream rebuild phase-1 readiness

Created: `2026-06-18 07:15:50 UTC+09:00`

This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.

## Required report checks

| path | exists | bytes |
|---|---|---:|
| `reports/post_id341_upstream_rebuild_blueprint_20260618.json` | `True` | 14865 |
| `reports/id340_validation_split_manifest_20260618.json` | `True` | 135846 |
| `reports/public_joint_distribution_risk_post_id341_20260618.json` | `True` | 37177 |
| `reports/joint_risk_candidate_rerank_post_id341_20260618.json` | `True` | 73137 |
| `reports/post_id341_family_priority_20260618.json` | `True` | 3919 |
| `reports/post_id341_a1_qfree_lane_closure_20260618.json` | `True` | 1726 |
| `reports/b1_pair_blend_source_dependency_audit_20260618.json` | `True` | 38379 |
| `reports/post_id341_public_transfer_audit_20260618.json` | `True` | 127464 |
| `reports/post_id341_transfer_gate_v2_joint_risk_20260618.json` | `True` | 122574 |
| `reports/post_id341_new_source_blueprint_20260618.json` | `True` | 13527 |
| `reports/sleep_boundary_prepromotion_gate_post_id341_20260618.json` | `True` | 9003 |
| `reports/evening_load_recovery_smoke_post_id341_20260618.json` | `True` | 88373 |
| `reports/combined_new_source_gate_post_id341_20260618.json` | `True` | 236127 |
| `reports/combined_sequence_sleep_evening_gate_post_id341_20260618.json` | `True` | 1236886 |
| `reports/routine_stability_place_entropy_smoke_post_id341_20260618.json` | `True` | 53347 |
| `reports/place_social_novelty_smoke_fixedlist_post_id341_20260618.json` | `True` | 52392 |
| `reports/sensor_adherence_dropout_smoke_post_id341_20260618.json` | `True` | 90975 |
| `reports/external_etri_prior_smoke_post_id341_20260618.json` | `True` | 37128 |
| `reports/combined_sleep_evening_external_prior_gate_post_id341_20260618.json` | `True` | 56491 |
| `reports/post_id341_phase2_raw_source_execution_spec_20260618.json` | `True` | 13253 |

## Split manifest

- Exists: `True`
- Train/test rows: `450` / `250`
- Total folds: `23`

## Source artifact readiness

### `upstream_model_layer`

| id | log | oof | test | status | local | public | ready |
|---:|---|---|---|---|---:|---:|---|
| 221 | `True` | `True` | `True` | `generated_unsubmitted` | 0.532990 |  | `True` |
| 226 | `True` | `True` | `True` | `generated_unsubmitted` | 0.491771 |  | `True` |
| 227 | `True` | `True` | `True` | `generated_unsubmitted` | 0.464409 |  | `True` |
| 228 | `True` | `True` | `True` | `generated_unsubmitted` | 0.482245 |  | `True` |
| 237 | `True` | `True` | `True` | `generated_unsubmitted` | 0.466691 |  | `True` |

### `a1_replay_layer`

| id | log | oof | test | status | local | public | ready |
|---:|---|---|---|---|---:|---:|---|
| 238 | `True` | `True` | `True` | `generated_unsubmitted` | 0.462365 |  | `True` |
| 239 | `True` | `True` | `True` | `generated_unsubmitted` | 0.437967 |  | `True` |
| 241 | `True` | `True` | `True` | `generated_unsubmitted` | 0.383459 |  | `True` |

### `b1_replay_layer`

| id | log | oof | test | status | local | public | ready |
|---:|---|---|---|---|---:|---:|---|
| 254 | `True` | `True` | `True` | `generated_unsubmitted` | 0.471985 |  | `True` |
| 255 | `True` | `True` | `True` | `generated_unsubmitted` | 0.469816 |  | `True` |
| 256 | `True` | `True` | `True` | `generated_unsubmitted` | 0.464121 |  | `True` |
| 257 | `True` | `True` | `True` | `generated_unsubmitted` | 0.433079 |  | `True` |
| 258 | `True` | `True` | `True` | `generated_unsubmitted` | 0.457647 |  | `True` |
| 266 | `True` | `True` | `True` | `generated_unsubmitted` | 0.457894 |  | `True` |

### `public_anchor_layer`

| id | log | oof | test | status | local | public | ready |
|---:|---|---|---|---|---:|---:|---|
| 178 | `True` | `True` | `True` | `cv_lb_gap_risk` | 0.538638 | 0.601411 | `True` |
| 287 | `True` | `True` | `True` | `cv_lb_gap_risk` | 0.378934 | 0.669879 | `True` |
| 329 | `True` | `True` | `True` | `cv_lb_gap_risk` | 0.516718 | 0.601791 | `True` |
| 341 | `True` | `True` | `True` | `generated_unsubmitted` | 0.538584 |  | `True` |

## Recent new-source read

```json
{
  "sleep_boundary": {
    "reviewable_feature_lead": true,
    "promote_submission_artifact": false,
    "proxy": 0.6016972511974908,
    "proxy_minus_public_id178": 0.00028595029749090006
  },
  "evening_load_recovery": {
    "selected_targets": [
      "S4"
    ],
    "best_proxy": 0.601506941674763,
    "reviewable": false
  },
  "combined_new_source": {
    "best_proxy": 0.6014904915736343,
    "proxy_below_id178_candidate_exists": false,
    "public_058_plausible": false
  },
  "combined_sequence_sleep_evening": {
    "best_proxy": 0.6014904915736343,
    "best_proxy_sequence_scale": 0.0,
    "proxy_below_id178_candidate_exists": false,
    "public_058_plausible": false
  },
  "routine_stability": {
    "selected_targets": [],
    "reviewable": false
  },
  "place_social_novelty": {
    "selected_targets": [],
    "reviewable": false,
    "best_proxy": null,
    "train_missing_mean": 0.019823232323232323
  },
  "sensor_adherence_dropout": {
    "selected_targets": [],
    "reviewable": false,
    "best_proxy": null,
    "train_missing_mean": 0.0
  },
  "external_etri_prior": {
    "targets_available": [
      "Q1",
      "Q2",
      "Q3",
      "S1",
      "S2",
      "S3",
      "S4"
    ],
    "best_proxy": 0.6015636731586781,
    "reviewable": false,
    "public_058_plausible_candidate_exists": false
  },
  "combined_sleep_evening_external_prior": {
    "best_proxy": 0.6014904915736343,
    "best_proxy_sleep_scale": 0.0,
    "best_proxy_evening_scale": 0.025,
    "best_proxy_external_group": "Q1",
    "best_proxy_external_scale": 0.0,
    "reviewable": false,
    "public_058_plausible_candidate_exists": false
  },
  "phase2_raw_source_execution_spec": {
    "phase2_spec_ready": true,
    "recommended_next_revision": "remote_phase2_raw_multimodal_sequence_source_smoke",
    "create_submission_artifact": false,
    "run_heavy_rebuild_now": false
  },
  "transfer_gate_v2_joint_risk": {
    "combined_rows": 244,
    "unscored_combined_rows": 241,
    "public_058_plausible_candidate_exists": false,
    "any_candidate_passes_combined_public_058_gate": false,
    "best_combined_pass_id": null
  },
  "local_new_source_stack_read": {
    "proxy_below_id178_found": false,
    "public_058_plausible_found": false,
    "force_submission_artifact": false
  }
}
```

## Next phase queue

| phase | ready now | compute | creates submission | purpose |
|---|---|---|---|---|
| `payload_and_split_smoke` | `True` | Kaggle CPU or local | `False` | Verify project payload, frozen split manifest, and source artifact availability. |
| `upstream_model_layer_rebuild_design` | `True` | Kaggle CPU recommended before any long run | `False` | Define model-level approximations for fresh upstream sources using frozen folds and no public-LB feedback. |
| `a1_replay_from_clean_sources` | `True` | Kaggle CPU after source rebuild design passes | `False` | Replay A1 qfree selectors only from rebuilt/clean sources, not contaminated full-OOF artifacts. |
| `b1_dependency_rebuild_plan` | `True` | Kaggle CPU after A1 read | `False` | Plan pair/logit source dependencies below IDs 254/255/256/257/258/266 without creating a candidate CSV. |
| `transfer_gate_only` | `True` | local or Kaggle CPU | `False` | Compare any rebuilt source lead against ID178/287/329/341 plus the joint-risk transfer gate v2 before artifact review. |

## Decision

- Phase-1 ready: `True`
- Local new-source stack closed: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: Use this refreshed readiness report inside the Kaggle upstream runner. Local new-source stacks did not beat the ID178 public anchor, so do not create ID342 locally; the next useful approved action is a CPU-only report-first upstream rebuild runner before any artifact promotion.

