# Fold-internal failed-direction rebuild — post ID335

Created: `2026-06-17 22:44:17 UTC+09:00`

This is report-only. It recomputes beta/projection/choice inside each outer fold, but still uses existing source OOF artifacts.

## Results

| split | setting | overall | robust | worst target | worst subject | selected failed-source ids |
|---|---|---:|---:|---|---|---|
| `subject_hole_fold_internal` | `id311_like_cap100_lam160` | 0.524548 | 0.550487 | S4 `0.589554` | id10 `0.624401` | `[249, 252, 256, 268, 276, 278]` |
| `subject_hole_fold_internal` | `id310_like_cap120_lam160` | 0.522506 | 0.548380 | S4 `0.582843` | id10 `0.622975` | `[242, 243, 249, 252, 268, 276, 278]` |
| `subject_hole_fold_internal` | `tighter_cap060_lam120` | 0.525783 | 0.552088 | S4 `0.594354` | id10 `0.623844` | `[256, 268, 276, 278]` |
| `loso_fold_internal` | `id311_like_cap100_lam160` | 0.524831 | 0.550883 | S4 `0.592123` | id10 `0.624056` | `[242, 249, 252, 256, 268, 276, 278]` |
| `loso_fold_internal` | `id310_like_cap120_lam160` | 0.523274 | 0.549054 | S4 `0.584727` | id10 `0.622761` | `[242, 249, 252, 268, 276, 278]` |
| `loso_fold_internal` | `tighter_cap060_lam120` | 0.526594 | 0.552503 | S4 `0.593189` | id10 `0.622750` | `[256, 268, 276, 278]` |

## Verdict

Fold-internal recomputation preserves a meaningful part of the failed-direction signal. This is still not a promotion gate because source artifacts are pre-existing OOF products, but it justifies a true model-level fold-nested rebuild.

