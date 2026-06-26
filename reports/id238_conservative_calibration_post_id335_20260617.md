# ID238 conservative calibration diagnostic — post ID335

Created: `2026-06-17 23:08:48 UTC+09:00`

This is report-only. No CSV, notebook, submission-log row, base artifact, or DACON submission was created.

## Results

| split | scenario | base | calibrated | improvement vs base | robust | worst target | read |
|---|---|---:|---:|---:|---:|---|---|
| `subject_hole_fold_internal` | `base_only` | 0.462365 | 0.462365 | +0.000000 | 0.484746 | Q3 `0.509920` | worse than base |
| `subject_hole_fold_internal` | `temp_safe_cap012` | 0.462365 | 0.461268 | +0.001097 | 0.483738 | Q3 `0.512236` | beats base |
| `subject_hole_fold_internal` | `lowq_shrink_cap012` | 0.462365 | 0.468156 | -0.005791 | 0.490759 | Q3 `0.522627` | worse than base |
| `subject_hole_fold_internal` | `iso_shrink_cap012` | 0.462365 | 0.474547 | -0.012182 | 0.501883 | Q3 `0.536579` | worse than base |
| `subject_hole_fold_internal` | `combined_safe_cap012` | 0.462365 | 0.467244 | -0.004879 | 0.489272 | Q3 `0.518634` | worse than base |
| `subject_hole_fold_internal` | `combined_moderate_cap020` | 0.462365 | 0.466845 | -0.004480 | 0.488766 | Q3 `0.521801` | worse than base |
| `loso_fold_internal` | `base_only` | 0.462365 | 0.462365 | +0.000000 | 0.484746 | Q3 `0.509920` | worse than base |
| `loso_fold_internal` | `temp_safe_cap012` | 0.462365 | 0.460105 | +0.002260 | 0.482918 | Q3 `0.511668` | beats base |
| `loso_fold_internal` | `lowq_shrink_cap012` | 0.462365 | 0.467401 | -0.005036 | 0.489523 | Q3 `0.520029` | worse than base |
| `loso_fold_internal` | `iso_shrink_cap012` | 0.462365 | 0.474055 | -0.011690 | 0.501440 | Q3 `0.531729` | worse than base |
| `loso_fold_internal` | `combined_safe_cap012` | 0.462365 | 0.468855 | -0.006490 | 0.491449 | Q3 `0.528153` | worse than base |
| `loso_fold_internal` | `combined_moderate_cap020` | 0.462365 | 0.466278 | -0.003913 | 0.488456 | Q3 `0.518171` | worse than base |

## Best rows

| split | best scenario | calibrated | improvement vs base | robust |
|---|---|---:|---:|---:|
| `subject_hole_fold_internal` | `temp_safe_cap012` | 0.461268 | +0.001097 | 0.483738 |
| `loso_fold_internal` | `temp_safe_cap012` | 0.460105 | +0.002260 | 0.482918 |

## Decision

This diagnostic only tests conservative fold-internal calibration on top of pre-existing ID238 OOF predictions. Any positive result can guide a rebuild, but it is not leaderboard evidence and cannot authorize a submission artifact.
