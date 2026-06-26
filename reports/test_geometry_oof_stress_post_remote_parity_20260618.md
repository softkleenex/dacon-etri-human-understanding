# Test-geometry OOF stress — post remote parity

Created: `2026-06-18 03:18:27 UTC+09:00`

This is report-only. It slices existing OOF artifacts; it does not train a model, create a submission CSV, update the submission log, or submit to DACON.

## Slice sizes

- `all_oof`: 450
- `actual_test_band`: 98
- `test_boundary_neighbor2`: 82
- `test_boundary_neighbor5`: 172
- `fragmented_subjects`: 205
- `simple_subjects`: 245

## Known transfer traps

These rows prove why this diagnostic cannot promote a candidate by itself: they improve OOF geometry slices but did not improve public LB.

| id | public Δ vs 178 | all Δ | test-band Δ | neighbor2 Δ | geometry mean Δ | file |
|---:|---:|---:|---:|---:|---:|---|
| 287 | +0.068467 | -0.159704 | -0.152175 | -0.160250 | -0.157888 | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` |
| 329 | +0.000380 | -0.021919 | -0.021105 | -0.028232 | -0.023895 | `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv` |

## Watch candidates

| id | family | local | public | all Δ | test-band Δ | neighbor2 Δ | neighbor5 Δ | geometry mean Δ | read |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 304 | `failed_direction_combo_family` | 0.532349 |  | -0.006289 | -0.006343 | -0.009075 | -0.006162 | -0.007193 | quarantined family |
| 339 | `sequence_summary_combo_family` | 0.536441 |  | -0.002197 | -0.003109 | -0.007213 | -0.002340 | -0.004221 | OOF geometry improves, but proxy/movement still blocks promotion |
| 307 | `failed_direction_combo_family` | 0.537792 |  | -0.000845 | -0.001430 | -0.002900 | -0.000884 | -0.001738 | quarantined family |
| 337 | `sequence_temporal_family` | 0.538066 |  | -0.000571 | -0.000080 | -0.000200 | -0.000208 | -0.000163 | controlled probe only |
| 336 | `sequence_temporal_family` | 0.537663 |  | -0.000974 | +0.000098 | -0.000213 | -0.000181 | -0.000099 | watch only |
| 338 | `sequence_temporal_family` | 0.537861 |  | -0.000777 | +0.000847 | +0.001565 | +0.000766 | +0.001059 | watch only |
| 179 | `sequence_temporal_family` | 0.538290 |  | -0.000348 | +0.001078 | +0.001921 | +0.001009 | +0.001336 | watch only |
| 334 | `sequence_temporal_family` | 0.540347 |  | +0.001710 | +0.000894 | +0.004196 | +0.001634 | +0.002242 | watch only |
| 335 | `sequence_temporal_family` | 0.539690 |  | +0.001052 | +0.001556 | +0.003940 | +0.001770 | +0.002422 | watch only |
| 295 | `failed_direction_combo_family` | 0.548583 |  | +0.009945 | +0.009469 | +0.010103 | +0.010083 | +0.009885 | quarantined family |
| 297 | `failed_direction_combo_family` | 0.550861 |  | +0.012223 | +0.011754 | +0.012122 | +0.012250 | +0.012042 | quarantined family |

## Decision

- Promote submission candidate: **no**
- Create ID340: **no**
- Reason: OOF geometry stress is useful as a warning lens, but known public failures ID287/ID329 also look strong on these slices. Therefore no candidate can be promoted from this full-OOF stress alone.
- Next action: If pursuing ID340, rebuild the source under a true nested/test-geometry-aware split rather than recombining existing OOF artifacts.

