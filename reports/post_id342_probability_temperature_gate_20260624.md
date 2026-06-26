# Post-ID342 probability and temperature gate

Created: `2026-06-24 08:34:21 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Submission Contract

- Submission path: `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv`
- Base-test path: `artifacts/base_predictions/base_test_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv`
- Rows: `250`
- Max absolute delta vs base-test: `0.000000000000`
- All targets binary: `False`

## Temperature Rows

| target | wins | mean delta | max delta | min delta | eligible |
|---|---:|---:|---:|---:|---|
| `S3` | 0 | 0.001856684 | 0.004197615 | 0.000468000 | `False` |
| `Q3` | 2 | 0.003552168 | 0.015964080 | -0.002221903 | `False` |

## Decision

- Eligible rows: `0`
- Selected targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: ID342 submission is a probability file matching base_test; thresholding is not a direct local lever. Fold-internal temperature/shift calibration does not open Q3/S3, so do not create artifacts or submit.
