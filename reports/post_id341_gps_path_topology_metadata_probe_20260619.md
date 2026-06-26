# GPS path-topology metadata probe — post ID341

Created: `2026-06-19 06:49:29 UTC+09:00`

This is report-only. It creates no predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model training job, or DACON submission.

## Decision

- Metadata probe passed: `True`
- Source family: `gps_path_topology_sequence_v1`
- Target focus: `Q3, S3`
- Strong topology train/test rates: `0.9333` / `0.9360`
- Next local allowed step: `design_feature_spec_only`
- Train model now: `False`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: GPS topology metadata was feasible and has already been consumed by the feature-matrix/model-smoke lane, which selected no Q3/S3 target. Do not replay it as a new local artifact route; continue only with the explicit-approved upstream rebuild or another report-only source family.

## Source coverage

| source | rows | subjects | subject-days | train coverage | test coverage | median train rows/day |
|---|---:|---:|---:|---:|---:|---:|
| `mGps` | 800611 | 10 | 660 | 0.9444 | 0.9400 | 1387.0 |
| `mWifi` | 76336 | 10 | 685 | 0.9756 | 0.9840 | 119.0 |
| `mBle` | 21830 | 10 | 651 | 0.9244 | 0.9400 | 31.0 |
| `mActivity` | 961062 | 10 | 700 | 1.0000 | 1.0000 | 1440.0 |

## GPS topology feasibility

- GPS rows with coordinate payload: `800611`
- Mean/median points per GPS row: `10.676` / `11.000`
- Nonempty payload rate: `1.0000`

| split | expected days | strong topology days | weak topology days | strong rate | median GPS rows | median GPS points |
|---|---:|---:|---:|---:|---:|---:|
| `test` | 250 | 234 | 234 | 0.9360 | 1376.0 | 13957.0 |
| `train` | 450 | 420 | 421 | 0.9333 | 1371.0 | 12568.5 |

## Boundary

Passing this probe only means the raw logs are dense enough to design topology features. It does not open a submission artifact, does not count as Public 0.58 evidence, and does not supersede the approved clean upstream ID221→ID226 route.
