# Post-ID342 subject-state heavy success thresholds

Created: `2026-06-24 09:35:57 UTC+09:00`

This report defines how a future approved no-submit heavy run will be judged. It creates no runner, token cache, feature matrix, embedding, prediction, Kaggle run, artifact, log row, or DACON submission.

## Decision

- Threshold contract ready: `True`
- Artifact-review public proxy threshold: `0.58`
- DACON-candidate public proxy threshold: `0.55`
- Create runner files now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Required phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`
- Next action: Use these thresholds to review a future approved no-submit heavy run; do not execute now.

## Current Anchor

```json
{
  "gap_to_public_055": 0.046083251899999955,
  "id": 342,
  "public_score": 0.5960832519,
  "target_public_score": 0.55
}
```

## Minimum Downstream Metrics

```json
{
  "failed_frontier_excluded_required": true,
  "failed_frontier_inclusive_report_required": true,
  "other_primary_target_delta_vs_id342_max": 0.00025,
  "primary_worst_split_delta_vs_id342_max": 0.0005,
  "q3_or_s3_mean_stable_lens_delta_vs_id342_max": -0.001,
  "recent_slate_order_preservation_required": true
}
```

## Threshold Tiers

| tier | proxy max | artifact review | DACON candidate | rules |
|---|---:|---:|---:|---|
| `reject` | `None` | `False` | `False` | Any embedding purity failure.; Any submission artifact/log row created by the no-submit runner.; No Q3/S3 stable-lens improvement versus ID342.; Improvement appears only in failed-frontier-inclusive lenses. |
| `research_lead_only` | `0.5960832519` | `False` | `False` | At least one of Q3/S3 beats ID342 on stable failed-frontier-excluded lenses.; Projected public proxy is below ID342 but still above 0.58.; Record as source evidence only; do not create artifacts or submit. |
| `artifact_review_candidate` | `0.58` | `True` | `False` | Q3 or S3 has material stable-lens gain with worst primary split non-degrading.; Other primary score target is neutral or improving.; Projected public proxy is at or below 0.58.; Remote report proves embedding purity, privacy guards, and artifact separation. |
| `dacon_candidate_requires_fresh_exact_file_memo_approval` | `0.55` | `True` | `True` | Projected public proxy is at or below the target 0.55 zone.; Movement is not a post-public ID342-neighborhood tweak.; File/ipynb/result/log matching can be proven for the exact artifact.; DACON still requires exact file and memo approval before submission. |

## Mandatory Remote Reports

- `post_id342_subject_state_remote_manifest.json`
- `post_id342_subject_state_embedding_purity.json`
- `post_id342_subject_state_token_coverage.json`
- `post_id342_subject_state_downstream_smoke.json`
- `post_id342_subject_state_failed_frontier_resistance.json`
- `post_id342_subject_state_public_gap_budget.json`
- `post_id342_subject_state_artifact_separation.json`

## Stop Rules

| rule | stop if |
|---|---|
| `embedding_purity` | Any target label, OOF/base prediction, public score, or test-label proxy is used during pretraining. |
| `privacy_guard` | Raw app names, BSSID, BLE device IDs, or GPS coordinates are emitted in review artifacts. |
| `id342_stable_lens_failure` | Q3 and S3 both fail to beat ID342 on stable failed-frontier-excluded lenses. |
| `failed_frontier_dependence` | The apparent gain is driven by ID287/ID329-inclusive lenses or failed-direction geometry. |
| `public_gap_not_material` | Projected public proxy stays in the ID342 0.596 band. |
| `artifact_leak` | No-submit runner creates submission CSVs, base-test predictions, notebook-history artifacts, or log rows. |

## Prerequisites

```json
{
  "all_input_reports_parse": true,
  "docs_audit_clean": true,
  "goal_not_achieved": true,
  "heavy_readiness_ready_but_execution_closed": true,
  "id342_anchor_current": true,
  "runner_spec_ready": true,
  "stable_lens_policy_ready": true
}
```

## Forbidden Actions Now

```json
{
  "create_predictions": false,
  "create_runner_files": false,
  "create_submission_artifact": false,
  "fit_embedding": false,
  "modify_submission_log": false,
  "push_kaggle": false,
  "read_raw_parquet_payloads": false,
  "run_kaggle": false,
  "submit_dacon": false,
  "train_downstream_model": false,
  "write_feature_matrix": false,
  "write_token_cache": false
}
```
