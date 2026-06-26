# Subject residual target-restricted gate — post ID341

Created: `2026-06-18 07:37:17 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Feature and scan setup

- Focus targets: `Q1, S2, S4`
- Base numeric columns: `56`
- Residual feature columns: `399`
- Train/test missing mean: `0.0763` / `0.0627`
- Selection rule: wins >= `3`, mean Δ <= `-5e-05`, max Δ <= `0.0005`

## Selected configs

| target | config | wins | mean Δ | max Δ |
|---|---|---:|---:|---:|
| `Q1` | `Q1_id178_subject_residual_et_d4_l25_w0.01` | 3 | -0.000216 | +0.000215 |
| `S4` | `S4_id178_subject_residual_et_d4_l25_w0.01` | 3 | -0.000106 | +0.000239 |

## Assembled candidate by primary split

| split | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |
|---|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | 0.538649 | 0.538584 | +0.000065 | 0.538638 | +0.000011 | `False` |
| `actual_test_band_holdout` | 0.541367 | 0.541465 | -0.000097 | 0.541604 | -0.000236 | `True` |
| `test_boundary_neighbor2_holdout` | 0.585939 | 0.586137 | -0.000198 | 0.586290 | -0.000351 | `True` |
| `test_boundary_neighbor5_holdout` | 0.557872 | 0.557920 | -0.000049 | 0.557986 | -0.000114 | `True` |
| `loso_10` | 0.538633 | 0.538584 | +0.000050 | 0.538638 | -0.000004 | `False` |

## Best scanned rows by target

### `Q1`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `Q1_id178_subject_residual_et_d4_l25_w0.025` | 3 | -0.000349 | +0.000634 |
| `Q1_id341_subject_residual_et_d4_l25_w0.025` | 3 | -0.000349 | +0.000634 |
| `Q1_id178_subject_residual_hgb_l4_l2_5_w0.025` | 3 | -0.000227 | +0.000758 |
| `Q1_id341_subject_residual_hgb_l4_l2_5_w0.025` | 3 | -0.000227 | +0.000758 |
| `Q1_id178_subject_residual_et_d4_l25_w0.01` | 3 | -0.000216 | +0.000215 |

### `S2`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `S2_id178_subject_residual_et_d4_l25_w0.005` | 2 | +0.000027 | +0.000158 |
| `S2_id341_subject_residual_et_d4_l25_w0.005` | 2 | +0.000027 | +0.000158 |
| `S2_id178_subject_residual_hgb_l4_l2_5_w0.005` | 2 | +0.000063 | +0.000171 |
| `S2_id341_subject_residual_hgb_l4_l2_5_w0.005` | 2 | +0.000063 | +0.000171 |
| `S2_id178_subject_residual_et_d4_l25_w0.01` | 2 | +0.000065 | +0.000326 |

### `S4`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `S4_id178_subject_residual_et_d4_l25_w0.05` | 3 | -0.000315 | +0.001379 |
| `S4_id341_subject_residual_et_d4_l25_w0.05` | 3 | -0.000315 | +0.001379 |
| `S4_id178_subject_residual_et_d4_l25_w0.025` | 3 | -0.000223 | +0.000626 |
| `S4_id341_subject_residual_et_d4_l25_w0.025` | 3 | -0.000223 | +0.000626 |
| `S4_id178_subject_residual_et_d4_l25_w0.01` | 3 | -0.000106 | +0.000239 |

## In-memory test geometry

- Mean absolute delta vs ID341 base test: `0.000425`
- Mean absolute delta vs ID178 base test: `0.000549`

## Decision

- Selected targets: `Q1, S4`
- All primary splits beat ID341: `False`
- Primary wins vs ID341: `3`
- Mean primary Δ vs ID341: `-0.000046`
- Max primary Δ vs ID341: `+0.000065`
- Projected public proxy from ID341: `0.603784`
- Beats ID178 public anchor proxy: `False`
- Public 0.58 reviewable: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Do not create ID342 from this gate unless all primary splits beat ID341 and the projected proxy falls below the actual ID178 public anchor. If the gate fails, treat Q1/S2/S4 residuals as design hints for the approved Kaggle phase-2 raw multimodal runner rather than local submission material.

