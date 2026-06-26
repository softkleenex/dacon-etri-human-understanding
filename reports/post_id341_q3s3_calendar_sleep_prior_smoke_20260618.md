# Q3/S3 calendar/sleep-prior smoke — post ID341

Created: `2026-06-23 18:10:36 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Feature coverage

- Rows: train `450`, test `250`
- Raw feature columns: `227`
- Missing indicators: `57`
- Mean missing rate: `0.0739`
- Train/test any-non-null row share: `1.0` / `1.0`
- Direct target columns exported: `[]`

## Selected target configs

| target | candidate | wins | mean Δ341 | max Δ341 |
|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |

## Top rows by target

### `Q3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_q3s3_calendar_sleep_prior_lr_c0_02_w0.005` | 0 | +0.000589 | +0.001018 |
| 2 | `blend_id341_q3s3_calendar_sleep_prior_lr_c0_02_w0.005` | 0 | +0.000589 | +0.001018 |
| 3 | `blend_id178_q3s3_calendar_sleep_prior_lr_c0_08_w0.005` | 0 | +0.000738 | +0.001363 |
| 4 | `blend_id341_q3s3_calendar_sleep_prior_lr_c0_08_w0.005` | 0 | +0.000738 | +0.001363 |
| 5 | `blend_id178_q3s3_calendar_sleep_prior_lr_c0_20_w0.005` | 0 | +0.000800 | +0.001479 |

### `S3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_q3s3_calendar_sleep_prior_lr_c0_02_w0.005` | 0 | +0.000215 | +0.000323 |
| 2 | `blend_id341_q3s3_calendar_sleep_prior_lr_c0_02_w0.005` | 0 | +0.000215 | +0.000323 |
| 3 | `blend_id178_q3s3_calendar_sleep_prior_lr_c0_08_w0.005` | 0 | +0.000278 | +0.000375 |
| 4 | `blend_id341_q3s3_calendar_sleep_prior_lr_c0_08_w0.005` | 0 | +0.000278 | +0.000375 |
| 5 | `blend_id178_q3s3_calendar_sleep_prior_lr_c0_20_w0.005` | 0 | +0.000331 | +0.000470 |

## Decision

- Model smoke completed: `True`
- Selected targets: `[]`
- Q3 signal found: `False`
- S3 signal found: `False`
- All Q3/S3 targets covered: `False`
- Public `0.58` evidence: `False`
- Next local allowed step: `close_calendar_sleep_prior_lane`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Keep any selected Q3/S3 calendar/sleep-prior row as context only; do not create a submission artifact unless a later independent gate covers Q3/S3 and clears the Public-0.58 threshold. If no target is selected, keep the explicit-approved upstream rebuild as the score-relevant route.

