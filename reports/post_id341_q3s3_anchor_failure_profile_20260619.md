# Post-ID341 Q3/S3 anchor failure profile

Created: `2026-06-23 18:10:18 UTC+09:00`

This is report-only. It trains no model and creates no new predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Known public traps

| candidate | public Δ178 | mean Q3/S3 OOF Δ178 | mean Q3/S3 move | trap |
|---:|---:|---:|---:|---|
| 287 | +0.068467 | -0.121549 | 0.091209 | `True` |
| 329 | +0.000380 | -0.014175 | 0.015911 | `True` |

## Q3/S3 anchor profile

| candidate | target | OOF Δ178 | primary wins | mean move | max move | pred mean |
|---:|---|---:|---:|---:|---:|---:|
| 178 | `Q3` | +0.000000 | 0 | 0.000000 | 0.000000 | 0.598015 |
| 178 | `S3` | +0.000000 | 0 | 0.000000 | 0.000000 | 0.660085 |
| 221 | `Q3` | -0.004859 | 2 | 0.038971 | 0.168335 | 0.593880 |
| 221 | `S3` | -0.001404 | 5 | 0.015157 | 0.062446 | 0.660894 |
| 226 | `Q3` | -0.051075 | 5 | 0.117158 | 0.407957 | 0.591909 |
| 226 | `S3` | -0.025778 | 5 | 0.046038 | 0.234480 | 0.662222 |
| 287 | `Q3` | -0.220600 | 5 | 0.146441 | 0.298395 | 0.623006 |
| 287 | `S3` | -0.022497 | 5 | 0.035977 | 0.218055 | 0.661767 |
| 329 | `Q3` | -0.021426 | 5 | 0.022813 | 0.097558 | 0.587779 |
| 329 | `S3` | -0.006923 | 5 | 0.009009 | 0.072426 | 0.660468 |
| 341 | `Q3` | +0.000000 | 0 | 0.000000 | 0.000000 | 0.598015 |
| 341 | `S3` | +0.000000 | 0 | 0.000000 | 0.000000 | 0.660085 |

## Persistence slices

### `Q3`

| slice | rows | positive rate |
|---|---:|---:|
| `first_subject_row` | 10 | 0.500000 |
| `offset_1_to_0` | 75 | 0.000000 |
| `onset_0_to_1` | 78 | 1.000000 |
| `persistent` | 287 | 0.651568 |

### `S3`

| slice | rows | positive rate |
|---|---:|---:|
| `first_subject_row` | 10 | 0.600000 |
| `offset_1_to_0` | 75 | 0.000000 |
| `onset_0_to_1` | 76 | 1.000000 |
| `persistent` | 289 | 0.747405 |

## Decision

- Anchor failure profile ready: `True`
- Q3/S3 known OOF-public trap IDs: `[287, 329]`
- Q3/S3 OOF gain alone is submission evidence: `False`
- Unsubmitted upstream has Q3/S3 OOF signal: `True`
- Unsubmitted upstream requires rebuild, not direct submit: `True`
- Public `0.58` evidence: `False`
- Manual probe if spending a DACON submission: `None`
- Manual probe family status: `consumed_after_id341_id340_id337_public_micro_probe_slate`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Treat Q3/S3 OOF gains as rebuild-prior evidence only. Direct local artifact promotion remains blocked; the score-relevant route is still the explicit-approved clean ID221→ID226 no-submit upstream rebuild.

