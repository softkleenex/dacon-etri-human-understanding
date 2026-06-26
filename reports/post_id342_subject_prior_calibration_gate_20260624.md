# Post-ID342 subject-prior calibration gate

Created: `2026-06-24 08:28:32 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Subject Distribution

| target | train label mean | OOF prediction mean | test prediction mean |
|---|---:|---:|---:|
| `Q3` | 0.600000 | 0.595947 | 0.632949 |
| `S3` | 0.662222 | 0.660489 | 0.638923 |

## Top Rows

| target | config | wins | mean delta | max delta | eligible |
|---|---|---:|---:|---:|---|
| `Q3` | `Q3_logit_subject_mean_a10_w0.1` | 0 | 0.000918056 | 0.001608883 | `False` |
| `Q3` | `Q3_logit_subject_mean_a5_w0.1` | 0 | 0.000993866 | 0.001840688 | `False` |
| `Q3` | `Q3_prob_subject_mean_a10_w0.1` | 0 | 0.001007180 | 0.001809470 | `False` |
| `Q3` | `Q3_logit_subject_mean_a2_w0.1` | 0 | 0.001059636 | 0.002019700 | `False` |
| `Q3` | `Q3_prob_subject_mean_a5_w0.1` | 0 | 0.001139214 | 0.001972783 | `False` |
| `Q3` | `Q3_prob_subject_mean_a2_w0.1` | 0 | 0.001242218 | 0.002143697 | `False` |
| `S3` | `S3_prob_subject_mean_a2_w0.1` | 0 | 0.001384175 | 0.002330596 | `False` |
| `S3` | `S3_logit_subject_mean_a2_w0.1` | 0 | 0.001384573 | 0.002328823 | `False` |
| `S3` | `S3_logit_subject_mean_a5_w0.1` | 0 | 0.001554763 | 0.002360231 | `False` |
| `S3` | `S3_prob_subject_mean_a5_w0.1` | 0 | 0.001564800 | 0.002421293 | `False` |
| `S3` | `S3_logit_subject_mean_a10_w0.1` | 0 | 0.001799998 | 0.003073199 | `False` |
| `S3` | `S3_prob_subject_mean_a10_w0.1` | 0 | 0.001842576 | 0.003063839 | `False` |

## Decision

- Eligible rows: `0`
- Selected targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Subject-prior calibration does not improve Q3/S3 against ID342 under primary lenses. Do not force participant-level priors into test predictions; use only as validation context.
