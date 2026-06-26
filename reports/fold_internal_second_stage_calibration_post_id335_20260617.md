# Fold-internal second-stage calibration — post ID335

Created: `2026-06-17 22:59:29 UTC+09:00`

This is report-only. Calibration is refit inside each outer fold, but the base artifacts are still pre-existing OOF products.

## Results

| split | scenario | base | original full-OOF | fold-internal | gap vs original | improvement vs base | robust | worst target |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `subject_hole_fold_internal` | `id235_like_on_id226` | 0.491771 | 0.482255 | 0.493739 | +0.011484 | -0.001968 | 0.521728 | S4 `0.594261` |
| `subject_hole_fold_internal` | `id239_like_on_id238` | 0.462365 | 0.437967 | 0.509154 | +0.071186 | -0.046789 | 0.538198 | Q3 `0.573987` |
| `subject_hole_fold_internal` | `id242_like_on_id238` | 0.462365 | 0.357651 | 0.585137 | +0.227486 | -0.122772 | 0.618257 | S2 `0.674607` |
| `subject_hole_fold_internal` | `id243_like_on_id238` | 0.462365 | 0.360420 | 0.585541 | +0.225121 | -0.123176 | 0.614320 | S2 `0.674607` |
| `subject_hole_fold_internal` | `id250_like_on_id242` | 0.357651 | 0.261184 | 0.262550 | +0.001366 | +0.095101 | 0.300990 | S3 `0.465115` |
| `subject_hole_fold_internal` | `id251_like_on_id242` | 0.357651 | 0.263237 | 0.287292 | +0.024055 | +0.070358 | 0.321474 | S3 `0.465115` |
| `subject_hole_fold_internal` | `id252_like_on_id249` | 0.374465 | 0.274420 | 0.309219 | +0.034799 | +0.065246 | 0.342825 | S3 `0.465115` |
| `loso_fold_internal` | `id235_like_on_id226` | 0.491771 | 0.482255 | 0.499514 | +0.017259 | -0.007744 | 0.533394 | S4 `0.634688` |
| `loso_fold_internal` | `id239_like_on_id238` | 0.462365 | 0.437967 | 0.506809 | +0.068841 | -0.044444 | 0.539953 | Q1 `0.600169` |
| `loso_fold_internal` | `id242_like_on_id238` | 0.462365 | 0.357651 | 0.551329 | +0.193678 | -0.088964 | 0.577110 | Q2 `0.641555` |
| `loso_fold_internal` | `id243_like_on_id238` | 0.462365 | 0.360420 | 0.559601 | +0.199181 | -0.097236 | 0.584878 | Q2 `0.641555` |
| `loso_fold_internal` | `id250_like_on_id242` | 0.357651 | 0.261184 | 0.261856 | +0.000672 | +0.095794 | 0.300258 | S3 `0.465115` |
| `loso_fold_internal` | `id251_like_on_id242` | 0.357651 | 0.263237 | 0.296376 | +0.033139 | +0.061275 | 0.325677 | S3 `0.465115` |
| `loso_fold_internal` | `id252_like_on_id249` | 0.374465 | 0.274420 | 0.318333 | +0.043913 | +0.056132 | 0.350641 | S3 `0.465115` |

## Best rows by split

| split | best scenario | fold-internal | robust | read |
|---|---|---:|---:|---|
| `subject_hole_fold_internal` | `id250_like_on_id242` | 0.262550 | 0.300990 | calibration survives partially |
| `loso_fold_internal` | `id250_like_on_id242` | 0.261856 | 0.300258 | calibration survives partially |

## Decision

Fold-internal calibration tests whether the second-stage layer survives outside full-OOF fitting, but the base artifacts are still pre-existing OOF products. Use this to choose rebuild priorities; do not promote any CSV or DACON submission from this diagnostic.
