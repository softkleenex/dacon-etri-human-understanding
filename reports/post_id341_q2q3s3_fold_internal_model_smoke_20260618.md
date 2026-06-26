# Q2/Q3/S3 fold-internal model smoke — post ID341

Created: `2026-06-23 18:09:37 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Feature input

- Aligned train rows: `450`
- Feature columns used: `494`
- Mean missing rate: `0.0221`

## Selected target configs

| target | candidate | wins | mean Δ341 | max Δ341 |
|---|---|---:|---:|---:|
| `Q2` | `blend_id341_q2q3s3_lr_c0_08_w0.05` | 5 | -0.002758 | -0.000201 |

## Top rows by target

### `Q2`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id341_q2q3s3_lr_c0_08_w0.05` | 5 | -0.002758 | -0.000201 |
| 2 | `blend_id341_q2q3s3_lr_c0_20_w0.025` | 5 | -0.001698 | -0.000287 |
| 3 | `blend_id341_q2q3s3_lr_c0_08_w0.025` | 5 | -0.001639 | -0.000289 |
| 4 | `blend_id341_q2q3s3_lr_c0_20_w0.01` | 5 | -0.000755 | -0.000190 |
| 5 | `blend_id341_q2q3s3_lr_c0_08_w0.01` | 5 | -0.000722 | -0.000162 |

### `Q3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_q2q3s3_lr_c0_02_w0.005` | 0 | +0.000216 | +0.000339 |
| 2 | `blend_id341_q2q3s3_lr_c0_02_w0.005` | 0 | +0.000216 | +0.000339 |
| 3 | `blend_id178_q2q3s3_lr_c0_08_w0.005` | 0 | +0.000277 | +0.000502 |
| 4 | `blend_id341_q2q3s3_lr_c0_08_w0.005` | 0 | +0.000277 | +0.000502 |
| 5 | `blend_id178_q2q3s3_lr_c0_20_w0.005` | 0 | +0.000304 | +0.000524 |

### `S3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_q2q3s3_lr_c0_02_w0.005` | 0 | +0.000486 | +0.000818 |
| 2 | `blend_id341_q2q3s3_lr_c0_02_w0.005` | 0 | +0.000486 | +0.000818 |
| 3 | `blend_id178_q2q3s3_lr_c0_20_w0.005` | 0 | +0.000583 | +0.001032 |
| 4 | `blend_id341_q2q3s3_lr_c0_20_w0.005` | 0 | +0.000583 | +0.001032 |
| 5 | `blend_id178_q2q3s3_lr_c0_08_w0.005` | 0 | +0.000585 | +0.001001 |

## Decision

- Model smoke completed: `True`
- Selected targets: `['Q2']`
- Any Q2/Q3/S3 signal found: `True`
- All Q2/Q3/S3 targets covered: `False`
- Public `0.58` evidence: `False`
- Next local allowed step: `combined_no_artifact_geometry_gate`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If selected targets exist, run a combined no-artifact geometry gate; otherwise close this local fallback and return to the explicit-approved upstream rebuild path.

