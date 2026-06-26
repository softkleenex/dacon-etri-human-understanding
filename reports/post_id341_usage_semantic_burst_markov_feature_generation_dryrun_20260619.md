# Post-ID341 usage semantic burst/Markov feature-generation dry run

Created: `2026-06-23 18:15:32 UTC+09:00`

This is report-only. It creates no feature matrix/parquet, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Dry run passed: `True`
- Source family: `usage_semantic_burst_markov_v1`
- Target focus: `['Q2', 'Q3']`
- Public `0.58` evidence: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_usage_semantic_burst_markov_feature_matrix_smoke_contract`
- Next action: A future no-approval step may only write a report-only feature-matrix smoke contract. Do not write the feature matrix/parquet, train a model, create ID342, run Kaggle, or submit.

## In-memory feature frame summary

- Rows train/test/all: `450` / `250` / `700`
- Feature columns: `77`
- Finite values: `True`
- NaN count: `0`
- Zero-variance feature count: `5`
- Max absolute value: `72527472.000000`
- Mean abs train/test feature-mean delta: `17769.686642`

## Feature groups

| group | columns |
|---|---:|
| `coverage_quality_flags` | `7` |
| `app_burst_session_shape` | `14` |
| `semantic_category_distribution` | `21` |
| `markov_transition_shape` | `10` |
| `screen_coupled_bursts` | `10` |
| `ambience_context_guards` | `8` |
| `subject_relative_rolling_baseline` | `7` |

## Leakage and persistence boundary

```json
{
  "labels_loaded": false,
  "oof_or_test_predictions_loaded": false,
  "public_scores_used": false,
  "raw_app_name_feature_columns_created": false,
  "target_guided_vocabulary_used": false,
  "feature_frame_persisted": false,
  "all_missing_values_filled_with_flags_or_zero": true
}
```

Passing this dry run does not create ID342 and does not authorize feature-matrix persistence, model training, a submission CSV, submission-log changes, Kaggle execution, DACON submission, or a Public `0.58` claim.

