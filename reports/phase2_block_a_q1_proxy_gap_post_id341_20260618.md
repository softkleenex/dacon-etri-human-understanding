# Phase-2 block A Q1 proxy gap — post ID341

Created: `2026-06-18 08:08:14 UTC+09:00`

This is report-only. It reads the existing block-A smoke and performs gap math only; no model is trained and no predictions or submission artifacts are created.

## Gap math

- Current projected proxy: `0.603617`
- Gap vs actual ID178 Public `0.6014113009`: `+0.002206`
- Required additional mean delta to beat ID178: `-0.002206`
- Current Q1 target delta: `-0.001489`
- Q1-only signal factor needed to beat ID178: `11.37x`
- Q1-only signal factor needed for Public 0.58: `65.02x`

## Q1 frontier rows

| candidate | wins | target mean Δ | max Δ | proxy if Q1-only | gap vs ID178 |
|---|---:|---:|---:|---:|---:|
| `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.05` | 5 | -0.001489 | -0.000058 | 0.603617 | +0.002206 |
| `Q1_id341_sleep_residual_bridge_hgb_l4_l2_5_w0.05` | 5 | -0.001489 | -0.000058 | 0.603617 | +0.002206 |
| `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.025` | 5 | -0.001009 | -0.000157 | 0.603686 | +0.002274 |
| `Q1_id341_sleep_residual_bridge_hgb_l4_l2_5_w0.025` | 5 | -0.001009 | -0.000157 | 0.603686 | +0.002274 |
| `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.01` | 5 | -0.000479 | -0.000096 | 0.603761 | +0.002350 |
| `Q1_id341_sleep_residual_bridge_hgb_l4_l2_5_w0.01` | 5 | -0.000479 | -0.000096 | 0.603761 | +0.002350 |
| `Q1_id178_sleep_residual_bridge_lr_c0_03_w0.01` | 5 | -0.000363 | -0.000010 | 0.603778 | +0.002367 |
| `Q1_id341_sleep_residual_bridge_lr_c0_03_w0.01` | 5 | -0.000363 | -0.000010 | 0.603778 | +0.002367 |
| `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.005` | 5 | -0.000253 | -0.000054 | 0.603794 | +0.002382 |
| `Q1_id341_sleep_residual_bridge_hgb_l4_l2_5_w0.005` | 5 | -0.000253 | -0.000054 | 0.603794 | +0.002382 |

## S4 frontier check

| candidate | wins | target mean Δ | max Δ | eligible |
|---|---:|---:|---:|---|
| `S4_id178_sleep_residual_bridge_et_d4_l25_w0.05` | 3 | -0.000603 | +0.001349 | `False` |
| `S4_id341_sleep_residual_bridge_et_d4_l25_w0.05` | 3 | -0.000603 | +0.001349 | `False` |
| `S4_id178_sleep_residual_bridge_et_d4_l25_w0.025` | 3 | -0.000366 | +0.000611 | `False` |
| `S4_id341_sleep_residual_bridge_et_d4_l25_w0.025` | 3 | -0.000366 | +0.000611 | `False` |
| `S4_id178_sleep_residual_bridge_et_d4_l25_w0.01` | 3 | -0.000162 | +0.000228 | `False` |
| `S4_id341_sleep_residual_bridge_et_d4_l25_w0.01` | 3 | -0.000162 | +0.000228 | `False` |

## Decision

- Q1 all-primary signal confirmed: `True`
- Q1 sufficient for ID178 proxy: `False`
- Q1 sufficient for Public 0.58: `False`
- Additional independent target/source signal required: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Do not promote Q1-only block A. Use it as the current strongest parity target, but reaching ID178/Public requires an additional independent target/source signal rather than merely enlarging this local Q1 overlay.

