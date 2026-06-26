# Q2/Q3/S3 paper-family fold smoke — post ID341

Created: `2026-06-23 18:10:17 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Paper-family source groups

| target | paper family | source groups | feature columns | missing rate |
|---|---|---|---:|---:|
| `Q2` | `q2_mUsageStats_DW` | `mUsageStats, DW` | 31 | 0.0194 |
| `Q3` | `q3_wPedo_mUsageStats_DW` | `wPedo, mUsageStats, DW` | 58 | 0.0677 |
| `S3` | `s3_wPedo_mLight_wLight_DW` | `wPedo, mLight, wLight, DW` | 64 | 0.0890 |

## Selected target configs

| target | candidate | wins | mean Δ341 | max Δ341 |
|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |

## Top rows by target

### `Q2`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id341_q2_mUsageStats_DW_lr_c0_02_w0.005` | 0 | +0.000186 | +0.000294 |
| 2 | `blend_id341_q2_mUsageStats_DW_lr_c0_08_w0.005` | 0 | +0.000245 | +0.000329 |
| 3 | `blend_id341_q2_mUsageStats_DW_lr_c0_20_w0.005` | 0 | +0.000274 | +0.000352 |
| 4 | `blend_id341_q2_mUsageStats_DW_lr_c0_02_w0.01` | 0 | +0.000378 | +0.000598 |
| 5 | `blend_id341_q2_mUsageStats_DW_lr_c0_08_w0.01` | 0 | +0.000496 | +0.000664 |

### `Q3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_q3_wPedo_mUsageStats_DW_lr_c0_02_w0.005` | 1 | +0.000159 | +0.000364 |
| 2 | `blend_id341_q3_wPedo_mUsageStats_DW_lr_c0_02_w0.005` | 1 | +0.000159 | +0.000364 |
| 3 | `blend_id178_q3_wPedo_mUsageStats_DW_lr_c0_20_w0.005` | 1 | +0.000175 | +0.000246 |
| 4 | `blend_id341_q3_wPedo_mUsageStats_DW_lr_c0_20_w0.005` | 1 | +0.000175 | +0.000246 |
| 5 | `blend_id178_q3_wPedo_mUsageStats_DW_lr_c0_02_w0.01` | 1 | +0.000323 | +0.000731 |

### `S3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_s3_wPedo_mLight_wLight_DW_lr_c0_20_w0.005` | 1 | +0.000145 | +0.000261 |
| 2 | `blend_id341_s3_wPedo_mLight_wLight_DW_lr_c0_20_w0.005` | 1 | +0.000145 | +0.000261 |
| 3 | `blend_id178_s3_wPedo_mLight_wLight_DW_lr_c0_20_w0.01` | 1 | +0.000303 | +0.000527 |
| 4 | `blend_id341_s3_wPedo_mLight_wLight_DW_lr_c0_20_w0.01` | 1 | +0.000303 | +0.000527 |
| 5 | `blend_id178_s3_wPedo_mLight_wLight_DW_lr_c0_08_w0.005` | 0 | +0.000214 | +0.000357 |

## Decision

- Paper-family smoke completed: `True`
- Selected targets: `[]`
- Selected Q3/S3 targets: `[]`
- Q3/S3 covered by paper family: `False`
- All Q2/Q3/S3 targets covered: `False`
- Public `0.58` evidence: `False`
- Next local allowed step: `close_paper_family_local_lane`
- Manual probe if spending a DACON submission: `None`
- Manual probe family status: `consumed_after_id341_id340_id337_public_micro_probe_slate`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If Q3/S3 is selected, run a no-artifact geometry gate; otherwise keep the score-relevant route at the explicit-approved clean ID221→ID226 upstream rebuild.

