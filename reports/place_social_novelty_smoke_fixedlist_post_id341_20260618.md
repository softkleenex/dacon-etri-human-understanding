# Place/social novelty smoke — post ID341

Created: `2026-06-18 05:56:25 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push, or DACON submission.

## Feature coverage

- Rows: `700` total / `450` train / `250` test
- Feature columns: `88`
- Mean missing rate: train `0.0198`, test `0.0123`
- Raw sources: `['wifi_bssid', 'ble_address', 'usage_app_name']`

## Selected target configs

| target | candidate | wins | mean Δ | max Δ |
|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |

## Top scan rows by target

### `Q1`

| rank | candidate | wins | mean Δ | max Δ |
|---:|---|---:|---:|---:|
| 1 | `Q1_id178_lr_c0_05_w0.01` | 3 | -0.000081 | +0.000326 |
| 2 | `Q1_id341_lr_c0_05_w0.01` | 3 | -0.000081 | +0.000326 |
| 3 | `Q1_id178_hgb_l5_l2_2_w0.01` | 3 | -0.000076 | +0.000303 |
| 4 | `Q1_id341_hgb_l5_l2_2_w0.01` | 3 | -0.000076 | +0.000303 |
| 5 | `Q1_id178_hgb_l5_l2_2_w0.03` | 3 | -0.000056 | +0.000998 |

### `Q2`

| rank | candidate | wins | mean Δ | max Δ |
|---:|---|---:|---:|---:|
| 1 | `Q2_id341_lr_c0_2_w0.03` | 3 | -0.000092 | +0.001140 |
| 2 | `Q2_id341_lr_c0_2_w0.01` | 3 | -0.000076 | +0.000355 |
| 3 | `Q2_id341_lr_c0_05_w0.01` | 1 | -0.000012 | +0.000323 |
| 4 | `Q2_id341_lr_c0_2_w0.05` | 1 | +0.000064 | +0.002023 |
| 5 | `Q2_id341_lr_c0_05_w0.03` | 1 | +0.000085 | +0.001040 |

### `Q3`

| rank | candidate | wins | mean Δ | max Δ |
|---:|---|---:|---:|---:|
| 1 | `Q3_id178_lr_c0_2_w0.01` | 1 | +0.000102 | +0.000448 |
| 2 | `Q3_id341_lr_c0_2_w0.01` | 1 | +0.000102 | +0.000448 |
| 3 | `Q3_id178_lr_c0_05_w0.01` | 1 | +0.000146 | +0.000634 |
| 4 | `Q3_id341_lr_c0_05_w0.01` | 1 | +0.000146 | +0.000634 |
| 5 | `Q3_id178_lr_c0_2_w0.03` | 1 | +0.000404 | +0.001436 |

### `S3`

| rank | candidate | wins | mean Δ | max Δ |
|---:|---|---:|---:|---:|
| 1 | `S3_id178_hgb_l5_l2_2_w0.01` | 0 | +0.000394 | +0.000503 |
| 2 | `S3_id341_hgb_l5_l2_2_w0.01` | 0 | +0.000394 | +0.000503 |
| 3 | `S3_id178_lr_c0_2_w0.01` | 0 | +0.000470 | +0.000696 |
| 4 | `S3_id341_lr_c0_2_w0.01` | 0 | +0.000470 | +0.000696 |
| 5 | `S3_id178_lr_c0_05_w0.01` | 0 | +0.000513 | +0.000710 |

## Decision

- Reviewable low-movement candidate: `False`
- Public `0.58` plausible: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: If no reviewable place/social novelty row appears, do not widen this source locally; prefer the report-only upstream rebuild or a materially different raw source.

