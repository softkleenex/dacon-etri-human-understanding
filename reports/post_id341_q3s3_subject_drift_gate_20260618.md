# Post-ID341 Q3/S3 subject-drift gate

Created: `2026-06-23 18:10:37 UTC+09:00`

This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Target evidence

| target | wins vs ID341 | mean Δ341 | min Δ341 | max Δ341 |
|---|---:|---:|---:|---:|
| `Q3` | 1/5 | -0.000317 | -0.001587 | +0.000000 |
| `S3` | 2/5 | -0.000076 | -0.000374 | +0.000000 |

## Decision

- Gate passed: `True`
- Q3 wins vs ID341: `1`
- S3 wins vs ID341: `2`
- Context signal: `False`
- Artifact allowed: `False`
- Closed for artifact: `True`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Keep subject-drift Q3/S3 evidence as negative/context-only evidence. It does not cover both Q3 and S3 strongly enough, does not beat the ID178 public-anchor proxy, and cannot create an artifact.

