# Q2 lead geometry gate — post ID341 Q2/Q3/S3 lane

Created: `2026-06-23 18:10:10 UTC+09:00`

This is report-only. Full-test predictions are computed in memory only for geometry inspection and are not written to disk.

## Selected Q2 config

- Config available: `True`
- Candidate: `blend_id341_q2q3s3_lr_c0_08_w0.05`
- Weight: `0.05`
- Mean Δ341: `-0.002758`
- Max Δ341: `-0.000201`

## Scale geometry

| scale | eff weight | primary wins | all primary | local | proxy | max Δ341 | failed proj | reviewable |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| 0.25 | 0.01250 | 5 | `True` | 0.538556 | 0.601911 | 0.009422 | -0.002789 | `True` |
| 0.50 | 0.02500 | 5 | `True` | 0.538542 | 0.602362 | 0.018845 | -0.005034 | `True` |
| 0.75 | 0.03750 | 5 | `True` | 0.538542 | 0.602814 | 0.028267 | -0.007279 | `True` |
| 1.00 | 0.05000 | 5 | `True` | 0.538555 | 0.603267 | 0.037690 | -0.009524 | `True` |
| 1.25 | 0.06250 | 4 | `False` | 0.538581 | 0.603719 | 0.047112 | -0.011769 | `False` |

## Decision

- Reviewable low-movement candidate exists: `True`
- Best proxy: `0.6019107525231998` at scale `0.25`
- Public `0.58` plausible: `False`
- Q3/S3 still uncovered: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Keep the Q2 lead as context only unless a later independent Q3/S3 source appears or the explicit-approved upstream rebuild confirms broader missing-target coverage.

