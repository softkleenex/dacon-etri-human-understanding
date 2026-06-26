# Phase-3 usage temporal microstructure smoke — post ID341

Created: `2026-06-18 08:29:14 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Feature coverage

- Raw rows / event rows: `45197` / `158426`
- Unique app names observed but not used directly: `585`
- Daily usage feature rows/columns: `690` / `302`
- Train/test rows with any usage: `450` / `250`
- Train/test missing mean: `0.0288` / `0.0167`
- Category event counts: `{'other': 58078, 'system_home': 34548, 'communication': 31601, 'media_search': 17585, 'routine_health': 10539, 'finance_shop': 6075}`

## Selected configs

No usage-temporal config passed the fixed eligibility rule.

## Assembled candidate by primary split

| split | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |
|---|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | 0.538584 | 0.538584 | +0.000000 | 0.538638 | -0.000054 | `False` |
| `actual_test_band_holdout` | 0.541465 | 0.541465 | +0.000000 | 0.541604 | -0.000139 | `False` |
| `test_boundary_neighbor2_holdout` | 0.586137 | 0.586137 | +0.000000 | 0.586290 | -0.000153 | `False` |
| `test_boundary_neighbor5_holdout` | 0.557920 | 0.557920 | +0.000000 | 0.557986 | -0.000066 | `False` |
| `loso_10` | 0.538584 | 0.538584 | +0.000000 | 0.538638 | -0.000054 | `False` |

## Best scanned rows by target

### `Q2`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `Q2_id341_usage_temporal_hgb_l4_l2_5_w0.005` | 1 | +0.000007 | +0.000179 |
| `Q2_id341_usage_temporal_hgb_l4_l2_5_w0.01` | 1 | +0.000023 | +0.000366 |
| `Q2_id341_usage_temporal_hgb_l4_l2_5_w0.025` | 1 | +0.000114 | +0.000972 |
| `Q2_id341_usage_temporal_hgb_l4_l2_5_w0.05` | 1 | +0.000414 | +0.002126 |
| `Q2_id178_usage_temporal_hgb_l4_l2_5_w0.025` | 1 | +0.000731 | +0.001328 |
| `Q2_id178_usage_temporal_hgb_l4_l2_5_w0.05` | 1 | +0.000998 | +0.002461 |

### `Q3`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `Q3_id178_usage_temporal_hgb_l4_l2_5_w0.005` | 1 | +0.000110 | +0.000357 |
| `Q3_id341_usage_temporal_hgb_l4_l2_5_w0.005` | 1 | +0.000110 | +0.000357 |
| `Q3_id178_usage_temporal_hgb_l4_l2_5_w0.01` | 1 | +0.000226 | +0.000718 |
| `Q3_id341_usage_temporal_hgb_l4_l2_5_w0.01` | 1 | +0.000226 | +0.000718 |
| `Q3_id178_usage_temporal_lr_c0_12_w0.005` | 1 | +0.000241 | +0.000429 |
| `Q3_id341_usage_temporal_lr_c0_12_w0.005` | 1 | +0.000241 | +0.000429 |

### `S1`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `S1_id178_usage_temporal_lr_c0_12_w0.005` | 0 | +0.000195 | +0.000384 |
| `S1_id341_usage_temporal_lr_c0_12_w0.005` | 0 | +0.000195 | +0.000384 |
| `S1_id178_usage_temporal_hgb_l4_l2_5_w0.005` | 0 | +0.000235 | +0.000462 |
| `S1_id341_usage_temporal_hgb_l4_l2_5_w0.005` | 0 | +0.000235 | +0.000462 |
| `S1_id178_usage_temporal_lr_c0_03_w0.005` | 0 | +0.000251 | +0.000558 |
| `S1_id341_usage_temporal_lr_c0_03_w0.005` | 0 | +0.000251 | +0.000558 |

### `S2`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `S2_id178_usage_temporal_lr_c0_03_w0.005` | 1 | +0.000222 | +0.000394 |
| `S2_id341_usage_temporal_lr_c0_03_w0.005` | 1 | +0.000222 | +0.000394 |
| `S2_id178_usage_temporal_lr_c0_03_w0.01` | 1 | +0.000448 | +0.000793 |
| `S2_id341_usage_temporal_lr_c0_03_w0.01` | 1 | +0.000448 | +0.000793 |
| `S2_id178_usage_temporal_lr_c0_03_w0.025` | 1 | +0.001157 | +0.002018 |
| `S2_id341_usage_temporal_lr_c0_03_w0.025` | 1 | +0.001157 | +0.002018 |

### `S4`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `S4_id178_usage_temporal_hgb_l4_l2_5_w0.005` | 2 | +0.000023 | +0.000120 |
| `S4_id341_usage_temporal_hgb_l4_l2_5_w0.005` | 2 | +0.000023 | +0.000120 |
| `S4_id178_usage_temporal_hgb_l4_l2_5_w0.01` | 2 | +0.000051 | +0.000243 |
| `S4_id341_usage_temporal_hgb_l4_l2_5_w0.01` | 2 | +0.000051 | +0.000243 |
| `S4_id178_usage_temporal_hgb_l4_l2_5_w0.025` | 2 | +0.000158 | +0.000629 |
| `S4_id341_usage_temporal_hgb_l4_l2_5_w0.025` | 2 | +0.000158 | +0.000629 |

## Decision

- Selected targets: `none`
- Usage temporal has config signal: `False`
- All primary splits beat ID341: `False`
- Primary wins vs ID341: `0`
- Mean primary Δ vs ID341: `+0.000000`
- Max primary Δ vs ID341: `+0.000000`
- Projected public proxy from ID341: `0.603830`
- Beats ID178 public anchor proxy: `False`
- Final usage-temporal gate pass: `False`
- Public 0.58 reviewable: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If the usage-temporal gate fails, do not create ID342 from mUsageStats microstructure. If it finds a small target-specific signal, keep it report-only until it can combine with an independent source and beat ID178 proxy under the transfer gate.

