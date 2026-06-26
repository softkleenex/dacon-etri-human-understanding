# Post-ID341 clean backbone rebuild design

Created: `2026-06-18 05:18:20 UTC+09:00`

This is report-only. It trains no model, creates no prediction artifact, and makes no DACON submission.

## Data readiness

- Train/test shape: `[450, 242]` / `[250, 242]`
- Numeric feature count: `230`
- All-null numeric columns: `10`

Feature group counts:

```json
{
  "activity": 25,
  "ambience": 16,
  "ble": 7,
  "calendar": 4,
  "demographic": 10,
  "gps": 13,
  "heart_rate": 38,
  "id_calendar": 4,
  "light": 33,
  "other": 6,
  "pedo": 27,
  "screen": 17,
  "usage": 27,
  "wifi": 8
}
```

## Target priority from existing ID226 signal

| target | ID226 delta vs ID178 | priority |
|---|---:|---|
| `Q2` | -0.064994 | `high` |
| `Q1` | -0.054903 | `high` |
| `S2` | -0.052389 | `high` |
| `Q3` | -0.051075 | `high` |
| `S4` | -0.040712 | `medium` |
| `S1` | -0.038218 | `medium` |
| `S3` | -0.025778 | `medium` |

## Stage contract

### `clean_id221_feature_similarity_root`

Rebuild the ID221-like feature-similarity signal without public-LB or full-OOF target leakage.

- Selection rule: Select per-target family/k/date/blend configs using only inner training-fold validation. Outer valid rows are score-only.
- Must output: `clean_id221_oof.parquet, clean_id221_test.parquet, clean_id221_metrics.json/.md`
- Promotion gate: Must improve ID178 or ID341 on frozen primary lenses with low test-geometry movement; otherwise hold as diagnostic only.

### `clean_id226_targetwise_convex_backbone`

Rebuild the ID226-like convex backbone without reusing the existing ID172 extreme full-OOF calibration artifact.

- Selection rule: For each target, fit calibrators and choose convex weights inside inner folds only. No target may use the same row for fitting a calibrator, selecting a weight, and reporting the outer score.
- Must output: `clean_id226_oof.parquet, clean_id226_test.parquet, clean_id226_metrics.json/.md`
- Promotion gate: Must beat ID341 proxy after transfer-gate geometry before any local artifact contract is considered.

### `hold_calibration_and_replay_layers`

Prevent the old failure mode from re-entering before a clean backbone exists.

- Selection rule: No replay or calibration derivative is allowed until clean ID221 and ID226 reports are green.
- Must output: `hold_decision.json/.md`
- Promotion gate: Always no-submit at this stage.

## Decision

- Clean backbone design ready: `True`
- First heavy compute focus: `clean_id221_then_clean_id226`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: If explicitly approved, push/run the CPU-only Kaggle runner in report-first mode. The next notebook revision after pulled green reports should implement only clean ID221/ID226 rebuild cells, not ID227/228/237 or A1/B1 replay.

