# Q2/Q3/S3 feature-generation metadata smoke — post ID341

Created: `2026-06-23 18:08:23 UTC+09:00`

This is report-only. It creates no submission CSV, no base prediction artifact, no notebook/history artifact, no submission-log row, no Kaggle push/run, and no DACON submission.

## Row coverage

- Rows: `700` total / `450` train / `250` test
- Expected rows OK: `True`
- Duplicate key rows: `0`
- Subjects: `10`, date range `2024-06-03` → `2024-11-19`

## Feature summary

- Raw feature columns: `247`
- Missing-indicator columns: `247`
- Total feature columns: `494`
- All-null raw feature count: `0`

| block | raw cols | all cols | train coverage | test coverage | train missing | test missing |
|---|---:|---:|---:|---:|---:|---:|
| `routine_stability_clock_v2` | 188 | 376 | 0.998 | 1.000 | 0.046 | 0.029 |
| `place_social_novelty_v2` | 41 | 82 | 0.978 | 0.988 | 0.049 | 0.042 |
| `ambient_load_recovery_context_v2` | 18 | 36 | 0.998 | 1.000 | 0.010 | 0.007 |

## Source metadata

| source | raw rows | daily rows | nested/items | feature cols |
|---|---:|---:|---:|---:|
| `wifi` | 76336 | 792 | 1270006 | 15 |
| `ble` | 21830 | 754 | 437054 | 15 |
| `usage` | 45197 | 800 | 158426 | 20 |
| `gps` | 800611 | 766 | 8546932 | 11 |
| `screen` | 939653 | 810 |  | 9 |
| `activity` | 961062 | 810 |  | 14 |
| `ambience` | 476577 | 809 | 4765770 | 14 |

## Leakage guards

- Forbidden direct identifier columns: `[]`
- Target columns present in feature frame: `[]`
- Direct high-cardinality identifier columns exported: `False`
- Model training performed: `False`
- Leakage guard OK: `True`

## Decision

- Metadata smoke passed: `True`
- Target focus: `Q2, Q3, S3`
- Next local allowed step: `fold_internal_q2q3s3_model_smoke_only`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If this metadata smoke passes, the next safe step is a fold-internal Q2/Q3/S3-only model smoke against ID178/ID341 anchors; still no submission artifact without a separate promotion gate.

