# Phase-2 block B S2 sleep/recovery retry — post ID341

Created: `2026-06-18 08:13:32 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Feature coverage

- Sleep-boundary feature columns: `294`
- Subject-residual feature columns: `399`
- Evening/recovery feature columns: `148`
- Combined S2 recovery feature columns: `841`
- Train/test missing mean: `0.0808` / `0.0721`

## Selected config

No S2 sleep/recovery config passed the fixed eligibility rule.

## Assembled candidate by primary split

| split | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |
|---|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | 0.538584 | 0.538584 | +0.000000 | 0.538638 | -0.000054 | `False` |
| `actual_test_band_holdout` | 0.541465 | 0.541465 | +0.000000 | 0.541604 | -0.000139 | `False` |
| `test_boundary_neighbor2_holdout` | 0.586137 | 0.586137 | +0.000000 | 0.586290 | -0.000153 | `False` |
| `test_boundary_neighbor5_holdout` | 0.557920 | 0.557920 | +0.000000 | 0.557986 | -0.000066 | `False` |
| `loso_10` | 0.538584 | 0.538584 | +0.000000 | 0.538638 | -0.000054 | `False` |

## Top scanned S2 rows

| config | wins | mean Δ | max Δ |
|---|---:|---:|---:|
| `S2_id178_sleep_recovery_retry_lr_c0_03_w0.005` | 3 | +0.000036 | +0.000254 |
| `S2_id341_sleep_recovery_retry_lr_c0_03_w0.005` | 3 | +0.000036 | +0.000254 |
| `S2_id178_sleep_recovery_retry_lr_c0_03_w0.01` | 3 | +0.000090 | +0.000518 |
| `S2_id341_sleep_recovery_retry_lr_c0_03_w0.01` | 3 | +0.000090 | +0.000518 |
| `S2_id178_sleep_recovery_retry_lr_c0_12_w0.005` | 2 | +0.000035 | +0.000233 |
| `S2_id341_sleep_recovery_retry_lr_c0_12_w0.005` | 2 | +0.000035 | +0.000233 |
| `S2_id178_sleep_recovery_retry_hgb_l4_l2_5_w0.005` | 2 | +0.000037 | +0.000158 |
| `S2_id341_sleep_recovery_retry_hgb_l4_l2_5_w0.005` | 2 | +0.000037 | +0.000158 |
| `S2_id178_sleep_recovery_retry_hgb_l4_l2_5_w0.01` | 2 | +0.000085 | +0.000329 |
| `S2_id341_sleep_recovery_retry_hgb_l4_l2_5_w0.01` | 2 | +0.000085 | +0.000329 |

## Decision

- Selected targets: `none`
- S2 has config signal: `False`
- All primary splits beat ID341: `False`
- Primary wins vs ID341: `0`
- Mean primary Δ vs ID341: `+0.000000`
- Max primary Δ vs ID341: `+0.000000`
- Projected public proxy from ID341: `0.603830`
- Beats ID178 public anchor proxy: `False`
- Final block-B gate pass: `False`
- Public 0.58 reviewable: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If final_block_b_gate_pass is false, close the local S2 sleep/recovery retry as a standalone artifact path and require a materially different raw source before creating a submission candidate.

