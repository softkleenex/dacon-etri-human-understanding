# Phase-2 block A Q1/S4 sleep-residual bridge — post ID341

Created: `2026-06-18 07:57:00 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Feature coverage

- Sleep feature columns: `294`
- Residual feature columns: `399`
- Bridge feature columns: `693`
- Train/test missing mean: `0.0731` / `0.0647`

## Selected configs

| target | config | wins | mean Δ | max Δ |
|---|---|---:|---:|---:|
| `Q1` | `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.05` | 5 | -0.001489 | -0.000058 |

## Assembled candidate by primary split

| split | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |
|---|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | 0.538575 | 0.538584 | -0.000008 | 0.538638 | -0.000062 | `True` |
| `actual_test_band_holdout` | 0.541362 | 0.541465 | -0.000103 | 0.541604 | -0.000242 | `True` |
| `test_boundary_neighbor2_holdout` | 0.585454 | 0.586137 | -0.000684 | 0.586290 | -0.000836 | `True` |
| `test_boundary_neighbor5_holdout` | 0.557720 | 0.557920 | -0.000200 | 0.557986 | -0.000266 | `True` |
| `loso_10` | 0.538515 | 0.538584 | -0.000069 | 0.538638 | -0.000123 | `True` |

## Best scanned rows by target

### `Q1`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.05` | 5 | -0.001489 | -0.000058 |
| `Q1_id341_sleep_residual_bridge_hgb_l4_l2_5_w0.05` | 5 | -0.001489 | -0.000058 |
| `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.025` | 5 | -0.001009 | -0.000157 |
| `Q1_id341_sleep_residual_bridge_hgb_l4_l2_5_w0.025` | 5 | -0.001009 | -0.000157 |
| `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.01` | 5 | -0.000479 | -0.000096 |
| `Q1_id341_sleep_residual_bridge_hgb_l4_l2_5_w0.01` | 5 | -0.000479 | -0.000096 |

### `S4`

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `S4_id178_sleep_residual_bridge_et_d4_l25_w0.05` | 3 | -0.000603 | +0.001349 |
| `S4_id341_sleep_residual_bridge_et_d4_l25_w0.05` | 3 | -0.000603 | +0.001349 |
| `S4_id178_sleep_residual_bridge_et_d4_l25_w0.025` | 3 | -0.000366 | +0.000611 |
| `S4_id341_sleep_residual_bridge_et_d4_l25_w0.025` | 3 | -0.000366 | +0.000611 |
| `S4_id178_sleep_residual_bridge_et_d4_l25_w0.01` | 3 | -0.000162 | +0.000228 |
| `S4_id341_sleep_residual_bridge_et_d4_l25_w0.01` | 3 | -0.000162 | +0.000228 |

## In-memory test geometry

- Mean absolute delta vs ID341 base test: `0.001350`
- Mean absolute delta vs ID178 base test: `0.001474`

## Decision

- Selected targets: `Q1`
- All primary splits beat ID341: `True`
- Primary wins vs ID341: `5`
- Mean primary Δ vs ID341: `-0.000213`
- Max primary Δ vs ID341: `-0.000008`
- Projected public proxy from ID341: `0.603617`
- Beats ID178 public anchor proxy: `False`
- Final block-A gate pass: `False`
- Public 0.58 reviewable: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If final_block_a_gate_pass is false, do not widen local blending; keep block A as negative/partial evidence and run the approved no-submit Kaggle block only if explicit remote execution is desired.

