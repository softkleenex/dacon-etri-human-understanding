# Circadian HR/light phase-shape feature-generation dry run — post ID341

Created: `2026-06-19 09:17:00 UTC+09:00`

This is report-only. It creates no feature matrix/parquet, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Dry run passed: `True`
- Source family: `circadian_hr_light_phase_shape_v1`
- Target focus: `Q3, S3, S2`
- Row alignment exact: `True`
- Feature column count: `48`
- Feature column bound passed: `True`
- Strong phase-shape train/test rates: `0.5622` / `0.6120`
- Weak phase-shape train/test rates: `0.7489` / `0.8280`
- Next local allowed step: `report_only_feature_matrix_smoke_contract`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Circadian feature dry run passed. The next no-approval step may only be a separate report-only feature-matrix smoke contract; no matrix write, model, ID342, Kaggle run, or DACON submission is allowed.

## Row alignment

```json
{
  "expected_rows": 700,
  "dryrun_rows": 700,
  "exact_key_alignment": true,
  "duplicate_keys": 0,
  "train_rows": 450,
  "test_rows": 250
}
```

## Coverage

| split | rows | strong rows | strong rate | weak rows | weak rate | HR cov | wLight cov | mLight cov | pedo cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `test` | 250 | 153 | 0.6120 | 207 | 0.8280 | 0.9800 | 1.0000 | 1.0000 | 0.9960 |
| `train` | 450 | 253 | 0.5622 | 337 | 0.7489 | 0.8689 | 0.9200 | 1.0000 | 0.8978 |

## Feature columns

- Feature column count: `48`
- Max allowed first dry-run columns: `56`
- Missing declared features: `[]`
- Extra features: `[]`
- All values finite: `True`

## Safety contract

```json
{
  "feature_matrix_written": false,
  "feature_parquet_written": false,
  "predictions_created": false,
  "submission_csv_created": false,
  "submission_log_modified": false,
  "model_trained": false,
  "kaggle_run_executed": false,
  "dacon_submission_executed": false,
  "uses_target_labels": false,
  "uses_oof_predictions": false,
  "uses_public_scores": false
}
```

## Boundary

This dry run proves only that the predeclared circadian feature design can be aligned in memory. It is not leaderboard evidence, does not open ID342, and does not allow a feature parquet/model without a separate contract.

