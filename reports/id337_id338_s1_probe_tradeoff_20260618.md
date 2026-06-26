# ID337 vs ID338 S1 probe tradeoff

Created: `2026-06-18 01:16:36 UTC+09:00`

Both candidates are generated-unsubmitted. No DACON submission was made.

| id | base | local CV | gain vs base | max Δ178 | failed proj | proxy | tier | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 337 | 178 | 0.538066 | 0.000572 | 0.037585 | -0.000554 | 0.605022 | `reviewable_low_movement` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 338 | 179 | 0.537861 | 0.000429 | 0.073989 | +0.000549 | 0.606833 | `reviewable_low_movement` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |

## Interpretation

- ID337 is the safer proxy choice: lowest movement from ID178 and best conservative proxy among the pair.
- ID338 is the stronger local choice: it improves over ID337 by `0.000205` local CV, but proxy worsens by `0.001811`.
- Neither candidate is Public `0.58` evidence. Treat both as controlled probes only, requiring explicit approval before submission.

Post-ID338 rerank top reviewable IDs: `[337, 307, 295, 304, 297, 179, 338, 305, 296, 336]`.
