# Fold-internal delta-capped bridge rebuild — post ID338

Created: `2026-06-18 01:39:38 UTC+09:00`

This is report-only. Bridge choices are learned inside each outer fold, but base/extreme source artifacts are existing OOF products.

## Results

| split | scenario | cap | overall | robust | changed | worst target | worst subject |
|---|---|---:|---:|---:|---:|---|---|
| `subject_hole_fold_internal` | `safe_caps_all_extremes` | 0.12 | 0.440194 | 0.458703 | 0.886 | S3 `0.465115` | id10 `0.537198` |
| `subject_hole_fold_internal` | `safe_caps_all_extremes` | 0.15 | 0.430666 | 0.450140 | 0.886 | S3 `0.465115` | id10 `0.526615` |
| `subject_hole_fold_internal` | `safe_caps_all_extremes` | 0.18 | 0.422016 | 0.442369 | 0.886 | S3 `0.465115` | id10 `0.516873` |
| `subject_hole_fold_internal` | `safe_caps_all_extremes` | 0.20 | 0.417514 | 0.438313 | 0.886 | S3 `0.465115` | id10 `0.511878` |
| `subject_hole_fold_internal` | `midrisk_caps_all_extremes` | 0.20 | 0.417514 | 0.438313 | 0.886 | S3 `0.465115` | id10 `0.511878` |
| `subject_hole_fold_internal` | `midrisk_caps_all_extremes` | 0.25 | 0.406092 | 0.428104 | 0.886 | S3 `0.465115` | id10 `0.499291` |
| `subject_hole_fold_internal` | `midrisk_caps_all_extremes` | 0.30 | 0.395423 | 0.418444 | 0.886 | S3 `0.465115` | id10 `0.487013` |
| `subject_hole_fold_internal` | `no_q240_extremes` | 0.15 | 0.443118 | 0.459739 | 0.886 | S3 `0.465115` | id10 `0.531863` |
| `subject_hole_fold_internal` | `no_q240_extremes` | 0.20 | 0.432688 | 0.449893 | 0.886 | S3 `0.465115` | id10 `0.517415` |
| `subject_hole_fold_internal` | `no_q240_extremes` | 0.25 | 0.423449 | 0.441116 | 0.886 | S3 `0.465115` | id10 `0.504665` |
| `subject_hole_fold_internal` | `no_q240_extremes` | 0.30 | 0.414272 | 0.432545 | 0.886 | S3 `0.465115` | id10 `0.491470` |
| `subject_hole_fold_internal` | `q240_only_stress` | 0.12 | 0.440194 | 0.458703 | 0.886 | S3 `0.465115` | id10 `0.537198` |
| `subject_hole_fold_internal` | `q240_only_stress` | 0.15 | 0.430666 | 0.450140 | 0.886 | S3 `0.465115` | id10 `0.526615` |
| `subject_hole_fold_internal` | `q240_only_stress` | 0.20 | 0.417514 | 0.438313 | 0.886 | S3 `0.465115` | id10 `0.511878` |
| `loso_fold_internal` | `safe_caps_all_extremes` | 0.12 | 0.440181 | 0.458699 | 0.900 | S3 `0.465115` | id10 `0.537198` |
| `loso_fold_internal` | `safe_caps_all_extremes` | 0.15 | 0.430666 | 0.450140 | 0.900 | S3 `0.465115` | id10 `0.526615` |
| `loso_fold_internal` | `safe_caps_all_extremes` | 0.18 | 0.422016 | 0.442369 | 0.900 | S3 `0.465115` | id10 `0.516873` |
| `loso_fold_internal` | `safe_caps_all_extremes` | 0.20 | 0.417514 | 0.438313 | 0.900 | S3 `0.465115` | id10 `0.511878` |
| `loso_fold_internal` | `midrisk_caps_all_extremes` | 0.20 | 0.417514 | 0.438313 | 0.900 | S3 `0.465115` | id10 `0.511878` |
| `loso_fold_internal` | `midrisk_caps_all_extremes` | 0.25 | 0.406010 | 0.427996 | 0.900 | S3 `0.465115` | id10 `0.499291` |
| `loso_fold_internal` | `midrisk_caps_all_extremes` | 0.30 | 0.395415 | 0.418415 | 0.900 | S3 `0.465115` | id10 `0.487115` |
| `loso_fold_internal` | `no_q240_extremes` | 0.15 | 0.443097 | 0.459631 | 0.900 | S3 `0.465115` | id10 `0.531348` |
| `loso_fold_internal` | `no_q240_extremes` | 0.20 | 0.433356 | 0.450291 | 0.900 | S3 `0.465115` | id10 `0.517119` |
| `loso_fold_internal` | `no_q240_extremes` | 0.25 | 0.423237 | 0.440976 | 0.900 | S3 `0.465115` | id10 `0.504434` |
| `loso_fold_internal` | `no_q240_extremes` | 0.30 | 0.414440 | 0.432730 | 0.900 | S3 `0.465115` | id10 `0.491769` |
| `loso_fold_internal` | `q240_only_stress` | 0.12 | 0.440181 | 0.458699 | 0.900 | S3 `0.465115` | id10 `0.537198` |
| `loso_fold_internal` | `q240_only_stress` | 0.15 | 0.430666 | 0.450140 | 0.900 | S3 `0.465115` | id10 `0.526615` |
| `loso_fold_internal` | `q240_only_stress` | 0.20 | 0.417514 | 0.438313 | 0.900 | S3 `0.465115` | id10 `0.511878` |

## Best rows

| split | scenario | cap | overall | robust | read |
|---|---|---:|---:|---:|---|
| `subject_hole_fold_internal` | `midrisk_caps_all_extremes` | 0.30 | 0.395423 | 0.418444 | uses all/q240 candidates |
| `loso_fold_internal` | `midrisk_caps_all_extremes` | 0.30 | 0.395415 | 0.418415 | uses all/q240 candidates |

## Decision

Fold-internal target-wise bridge selection preserves a bridge signal, but it still relies on existing ID235/extreme OOF artifacts. Treat this as support for rebuilding the bridge/source layer inside true model-level outer folds, not as submission evidence.

