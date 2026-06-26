# Q3/S3 physio-light-onset smoke — post ID341

Created: `2026-06-23 18:07:39 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Feature coverage

- Rows: `700` total / `450` train / `250` test
- Raw feature columns: `121`; total with missing flags: `242`
- Train/test non-null row share: `1.000` / `1.000`
- Train/test missing mean: `0.220` / `0.188`
- All-null raw features: `[]`

## Selected target configs

| target | candidate | wins | mean Δ341 | max Δ341 |
|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |

## Top rows by target

### `Q3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_plo_lr_c0_20_w0.005` | 3 | +0.000095 | +0.000395 |
| 2 | `blend_id341_plo_lr_c0_20_w0.005` | 3 | +0.000095 | +0.000395 |
| 3 | `blend_id178_plo_lr_c0_20_w0.01` | 3 | +0.000207 | +0.000803 |
| 4 | `blend_id341_plo_lr_c0_20_w0.01` | 3 | +0.000207 | +0.000803 |
| 5 | `blend_id178_plo_lr_c0_20_w0.025` | 3 | +0.000637 | +0.002100 |

### `S3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_plo_lr_c0_02_w0.005` | 0 | +0.000203 | +0.000245 |
| 2 | `blend_id341_plo_lr_c0_02_w0.005` | 0 | +0.000203 | +0.000245 |
| 3 | `blend_id178_plo_lr_c0_08_w0.005` | 0 | +0.000220 | +0.000318 |
| 4 | `blend_id341_plo_lr_c0_08_w0.005` | 0 | +0.000220 | +0.000318 |
| 5 | `blend_id178_plo_lr_c0_20_w0.005` | 0 | +0.000248 | +0.000378 |

## Decision

- Selected targets: `[]`
- Q3 signal found: `False`
- S3 signal found: `False`
- All Q3/S3 targets covered: `False`
- Public `0.58` evidence: `False`
- Next local allowed step: `close_physio_light_onset_lane`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If a selected target appears, run only a no-artifact geometry gate. Otherwise close this local lane and return to the explicit-approved upstream rebuild path for missing Q3/S3 coverage.

