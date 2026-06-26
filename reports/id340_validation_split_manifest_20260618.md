# ID340 validation split manifest

Created: `2026-06-18 03:18:26 UTC+09:00`

This freezes the validation lenses for any future ID340 source rebuild. No model was trained, no predictions were created, and no DACON submission was made.

## Split summary

| split | folds | valid min | valid mean | valid max | role |
|---|---:|---:|---:|---:|---|
| `subject_hole_5` | 5 | 85 | 90.0 | 95 | current strict local baseline lens |
| `subject_timeseries_5` | 5 | 59 | 59.0 | 59 | chronological same-subject stress |
| `actual_test_band_holdout` | 1 | 98 | 98.0 | 98 | known sample-row relative-position band stress |
| `test_boundary_neighbor2_holdout` | 1 | 82 | 82.0 | 82 | near hidden-test block boundary stress |
| `test_boundary_neighbor5_holdout` | 1 | 172 | 172.0 | 172 | wider hidden-test block boundary stress |
| `loso_10` | 10 | 33 | 45.0 | 57 | subject robustness sanity check |

## Future ID340 gate

- True nested/source rebuild required: `True`
- Existing full-OOF recombination alone forbidden: `True`
- Public-risk proxy ceiling before artifact promotion: `0.59`
- Required lenses: subject-hole, test-geometry stress, and LOSO sanity.
- A unique full artifact contract is required before any submission discussion.

## Operational read

The next useful work is not ID340-as-a-CSV. It is a source rebuild that is selected inside these splits and only then compared to ID178/ID329/ID287 geometry. This is the guardrail that prevents another ID287/ID329-style local-CV mirage.

