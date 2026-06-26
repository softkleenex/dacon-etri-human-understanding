# Circadian HR/light phase-shape feature spec — post ID341

Created: `2026-06-19 09:47:52 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Feature spec ready: `True`
- Source family: `circadian_hr_light_phase_shape_v1`
- Target focus: `Q3, S3, S2`
- Metadata probe passed: `True`
- Strong phase-shape train/test rates: `0.5622222222222222` / `0.612`
- Weak phase-shape train/test rates: `0.7488888888888889` / `0.828`
- Next local allowed step: `report_only_feature_generation_dryrun_no_matrix`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: The circadian HR/light phase-shape spec is ready. The next no-approval step may only be a report-only feature-generation dry run that checks row alignment, coverage, and schema bounds; no feature parquet, model, ID342, Kaggle run, or DACON submission is allowed.

## Feature blocks

| block | raw inputs | fit scope | target focus | feature count |
|---|---|---|---|---:|
| `phase_source_quality` | `wHr, wLight, mLight, wPedo` | `row_local_no_fit` | `Q3, S3, S2` | 8 |
| `wearable_hr_phase_shape` | `wHr` | `predeclared_hour_bins_only` | `Q3, S3` | 10 |
| `wearable_light_phase_shape` | `wLight` | `row_local_no_fit` | `Q3, S3, S2` | 8 |
| `phone_light_dark_continuity` | `mLight` | `row_local_no_fit` | `Q3, S3, S2` | 6 |
| `rest_activity_phase_shape` | `wPedo` | `predeclared_activity_thresholds_only` | `Q3, S3` | 10 |
| `cross_modal_phase_alignment` | `wHr, wLight, mLight, wPedo` | `row_local_no_fit` | `Q3, S3, S2` | 6 |

## Leakage guards

- No target labels, OOF predictions, submission CSVs, public scores, or prior candidate predictions may enter feature generation.
- All phase windows must be predeclared hour bins; do not tune time windows on validation loss.
- Subject-relative baselines, if added later, must be fit inside each training fold only.
- Final test feature generation may use row-local raw sensor values, but must not fit subject baselines from unlabeled test rows without a separate report.
- Missingness and coverage flags are allowed only as raw-source reliability features, not as public-score-calibrated switches.
- The first execution after this spec may only be a report-only dry run checking row alignment, feature counts, and train/test coverage.
- No feature parquet, fold model, base prediction artifact, notebook/history artifact, submission-log row, ID342, Kaggle run, or DACON submission is allowed by this spec.

## Resource bounds

```json
{
  "max_daily_feature_columns_for_first_dryrun": 56,
  "planned_first_pass_feature_columns": 48,
  "max_hour_bins": 24,
  "max_phase_summary_bins": 4,
  "max_rows_expected": 700,
  "allowed_first_execution": "report_only_feature_generation_dryrun_no_matrix",
  "disallowed_first_execution": [
    "feature_parquet_write",
    "fold_model_training",
    "test_prediction_generation",
    "submission_csv_creation",
    "kaggle_remote_run_without_exact_approval",
    "dacon_submission"
  ]
}
```

## Validation lenses

| lens | required before | pass rule |
|---|---|---|
| `row_alignment_dryrun` | `matrix` | The dry run must align exactly to 450 train + 250 test subject/lifelog rows with no duplicate keys. |
| `coverage_balance_gate` | `matrix` | Strong phase-shape rates must remain >=0.50 in train and test; weak rates must remain >=0.70. |
| `feature_count_gate` | `matrix` | The first-pass schema must stay within 56 feature columns plus keys and must contain only bounded numeric/boolean features. |
| `subject_hole_fit_scope` | `model` | Any later subject-relative phase baseline must be fitted inside each outer train fold only. |
| `target_specific_promotion_gate` | `artifact` | A future fold-internal model must beat the ID341/ID178 anchor on Q3/S3/S2 slices without worsening public-transfer risk. |
| `public_transfer_gate_v2` | `artifact` | No candidate can be promoted unless the conservative public proxy is reviewable below the current ID178/ID329 plateau. |

## Boundary

This spec turns a feasible raw-source family into a controlled dry-run plan. It is not leaderboard evidence, does not allow a feature parquet yet, and does not change the fact that the only current score-relevant open route is the explicit-approved clean upstream ID221→ID226 rebuild.

