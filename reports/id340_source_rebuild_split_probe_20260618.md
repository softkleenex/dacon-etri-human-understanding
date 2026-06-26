# ID340 source rebuild split probe

Created: `2026-06-18 03:23:13 UTC+09:00`

This is report-only. It trains lightweight source models inside the frozen ID340 splits, but creates no submission CSV, no base-prediction artifact, no notebook/history artifact, and no DACON submission.

- Numeric feature count: `289`
- Source candidates: `constant_prior, subject_prior_only, raw_logreg_base, raw_logreg_subject_prior, raw_logreg_past, raw_tiny_et_subject_prior, raw_tiny_et_past`

## Best source by split

| split | best source | best loss | ID178 ref loss | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---|
| `subject_hole_5` | `raw_tiny_et_past` | 0.613930 | 0.538638 | +0.075292 | `False` |
| `subject_timeseries_5` | `raw_tiny_et_subject_prior` | 0.632533 | 0.539747 | +0.092786 | `False` |
| `actual_test_band_holdout` | `raw_tiny_et_past` | 0.624550 | 0.541604 | +0.082946 | `False` |
| `test_boundary_neighbor2_holdout` | `subject_prior_only` | 0.632059 | 0.586290 | +0.045769 | `False` |
| `test_boundary_neighbor5_holdout` | `raw_tiny_et_past` | 0.630264 | 0.557986 | +0.072278 | `False` |
| `loso_10` | `raw_tiny_et_subject_prior` | 0.670760 | 0.538638 | +0.132122 | `False` |

## Primary gate

- `subject_hole_5` beats ID178 reference: `False`
- `actual_test_band_holdout` beats ID178 reference: `False`
- `test_boundary_neighbor2_holdout` beats ID178 reference: `False`
- `test_boundary_neighbor5_holdout` beats ID178 reference: `False`
- `loso_10` beats ID178 reference: `False`

Stable best-source counts on primary splits:

- `raw_tiny_et_past`: 3
- `raw_tiny_et_subject_prior`: 1
- `subject_prior_only`: 1

## Top rows per split

### `subject_hole_5`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `reference_id287` | `existing_oof_reference` | 0.378934 | -0.159704 |
| 2 | `reference_id329` | `existing_oof_reference` | 0.516718 | -0.021919 |
| 3 | `reference_id178` | `existing_oof_reference` | 0.538638 | +0.000000 |
| 4 | `raw_tiny_et_past` | `model_source` | 0.613930 | +0.075292 |
| 5 | `raw_tiny_et_subject_prior` | `model_source` | 0.615540 | +0.076902 |
| 6 | `subject_prior_only` | `baseline` | 0.633187 | +0.094549 |
| 7 | `constant_prior` | `baseline` | 0.666613 | +0.127975 |
| 8 | `raw_logreg_base` | `model_source` | 1.070838 | +0.532200 |

### `subject_timeseries_5`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `reference_id287` | `existing_oof_reference` | 0.380239 | -0.159508 |
| 2 | `reference_id329` | `existing_oof_reference` | 0.519671 | -0.020076 |
| 3 | `reference_id178` | `existing_oof_reference` | 0.539747 | +0.000000 |
| 4 | `raw_tiny_et_subject_prior` | `model_source` | 0.632533 | +0.092786 |
| 5 | `raw_tiny_et_past` | `model_source` | 0.632653 | +0.092906 |
| 6 | `constant_prior` | `baseline` | 0.671557 | +0.131809 |
| 7 | `subject_prior_only` | `baseline` | 0.688528 | +0.148781 |
| 8 | `raw_logreg_base` | `model_source` | 1.296972 | +0.757225 |

### `actual_test_band_holdout`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `reference_id287` | `existing_oof_reference` | 0.389428 | -0.152175 |
| 2 | `reference_id329` | `existing_oof_reference` | 0.520499 | -0.021105 |
| 3 | `reference_id178` | `existing_oof_reference` | 0.541604 | +0.000000 |
| 4 | `raw_tiny_et_past` | `model_source` | 0.624550 | +0.082946 |
| 5 | `raw_tiny_et_subject_prior` | `model_source` | 0.624678 | +0.083074 |
| 6 | `subject_prior_only` | `baseline` | 0.627938 | +0.086334 |
| 7 | `constant_prior` | `baseline` | 0.674183 | +0.132580 |
| 8 | `raw_logreg_base` | `model_source` | 1.186199 | +0.644595 |

### `test_boundary_neighbor2_holdout`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `reference_id287` | `existing_oof_reference` | 0.426040 | -0.160250 |
| 2 | `reference_id329` | `existing_oof_reference` | 0.558058 | -0.028232 |
| 3 | `reference_id178` | `existing_oof_reference` | 0.586290 | +0.000000 |
| 4 | `subject_prior_only` | `baseline` | 0.632059 | +0.045769 |
| 5 | `raw_tiny_et_past` | `model_source` | 0.633393 | +0.047103 |
| 6 | `raw_tiny_et_subject_prior` | `model_source` | 0.634196 | +0.047906 |
| 7 | `constant_prior` | `baseline` | 0.681690 | +0.095400 |
| 8 | `raw_logreg_subject_prior` | `model_source` | 1.210674 | +0.624385 |

### `test_boundary_neighbor5_holdout`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `reference_id287` | `existing_oof_reference` | 0.396747 | -0.161239 |
| 2 | `reference_id329` | `existing_oof_reference` | 0.535637 | -0.022349 |
| 3 | `reference_id178` | `existing_oof_reference` | 0.557986 | +0.000000 |
| 4 | `raw_tiny_et_past` | `model_source` | 0.630264 | +0.072278 |
| 5 | `raw_tiny_et_subject_prior` | `model_source` | 0.630926 | +0.072940 |
| 6 | `subject_prior_only` | `baseline` | 0.640515 | +0.082529 |
| 7 | `constant_prior` | `baseline` | 0.679235 | +0.121249 |
| 8 | `raw_logreg_subject_prior` | `model_source` | 1.387913 | +0.829927 |

### `loso_10`

| rank | candidate | kind | mean loss | Δ vs ID178 |
|---:|---|---|---:|---:|
| 1 | `reference_id287` | `existing_oof_reference` | 0.378934 | -0.159704 |
| 2 | `reference_id329` | `existing_oof_reference` | 0.516718 | -0.021919 |
| 3 | `reference_id178` | `existing_oof_reference` | 0.538638 | +0.000000 |
| 4 | `raw_tiny_et_subject_prior` | `model_source` | 0.670760 | +0.132122 |
| 5 | `raw_tiny_et_past` | `model_source` | 0.671085 | +0.132447 |
| 6 | `constant_prior` | `baseline` | 0.679633 | +0.140995 |
| 7 | `subject_prior_only` | `baseline` | 0.679633 | +0.140995 |
| 8 | `raw_logreg_base` | `model_source` | 1.615676 | +1.077038 |

## Decision

- Promote submission candidate: **no**
- Create ID340: **no**
- Any source beats ID178 on all primary splits: `False`
- Reason: This is a source-rebuild probe only. A future ID340 requires a stable source that improves the predeclared split lenses and then passes public-risk geometry; this report creates no artifact.

