# GPS path-topology feature spec — post ID341

Created: `2026-06-19 07:02:38 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Feature spec ready: `True`
- Source family: `gps_path_topology_sequence_v1`
- Target focus: `Q3, S3`
- Metadata probe passed: `True`
- Strong topology train/test rates: `0.9333333333333333` / `0.936`
- Next local allowed step: `report_only_feature_generation_dryrun`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: The GPS topology design spec is ready. The next no-approval step may only be a report-only feature-generation dry run that checks row alignment, coverage, and cardinality; no model, feature parquet, ID342, Kaggle run, or DACON submission is allowed.

## Feature blocks

| block | raw inputs | fit scope | target focus | feature count |
|---|---|---|---|---:|
| `gps_payload_quality` | `mGps` | `row_local_no_fit` | `Q3, S3` | 7 |
| `path_motion_shape` | `mGps, mActivity` | `predeclared_numeric_thresholds_only` | `Q3, S3` | 7 |
| `dwell_segment_topology` | `mGps, mWifi` | `train_fold_subject_baseline_only` | `Q3, S3` | 7 |
| `route_novelty_and_regularization` | `mGps, mWifi, mBle` | `train_fold_subject_baseline_only` | `Q3, S3` | 6 |
| `place_social_context_alignment` | `mWifi, mBle, mActivity` | `token_hashing_or_train_fold_vocabulary_only` | `Q3, S3` | 7 |

## Leakage guards

- No target labels, OOF predictions, submission CSVs, public scores, or previous candidate predictions may enter feature generation.
- Any subject-level home/place baseline must be fit inside the training fold only for validation.
- For final test feature generation, place baselines may use current train raw logs only unless a separate report approves unlabeled-test baselines.
- All coordinate bucketing must be predeclared or train-fold-fitted; never tune bucket sizes on validation loss.
- BLE/Wi-Fi vocabularies must be hashed or train-fold-fitted with capped cardinality.
- Feature generation may write only reports in the next dry run; no feature parquet is allowed until a separate smoke contract opens it.
- A fold model is explicitly blocked until the dry-run report proves row alignment, coverage, and cardinality bounds.

## Resource bounds

```json
{
  "max_gps_rows_expected": 1000000,
  "max_daily_feature_columns_for_first_matrix": 64,
  "max_coordinate_bins_per_subject": 32,
  "max_wifi_tokens_per_subject": 64,
  "max_ble_tokens_per_subject": 64,
  "allowed_first_execution": "report_only_feature_generation_dryrun",
  "disallowed_first_execution": [
    "fold_model_training",
    "test_prediction_generation",
    "submission_csv_creation",
    "kaggle_remote_run_without_exact_approval"
  ]
}
```

## Validation lenses

| lens | required before model/artifact | pass rule |
|---|---|---|
| `row_alignment_dryrun` | `model` | 700 subject/lifelog rows align exactly with train+sample IDs and dates; no duplicate keys. |
| `coverage_cardinality_gate` | `model` | train/test coverage both remain >= 0.90 and feature columns <= 64 in the first dry run. |
| `subject_hole_5` | `artifact` | future fold-internal model must beat ID341 on Q3/S3 target slices without moving broad distribution risk upward. |
| `public_transfer_gate_v2` | `artifact` | future candidate must pass transfer and joint-risk gates before any artifact-contract review. |

## Boundary

This spec turns a raw-source idea into a controlled dry-run plan. It is not leaderboard evidence, does not allow a feature parquet yet, and does not change the fact that the only current score-relevant open route is the explicit-approved clean upstream ID221→ID226 rebuild.
