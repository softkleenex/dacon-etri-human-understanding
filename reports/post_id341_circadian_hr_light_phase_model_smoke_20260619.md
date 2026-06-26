# Circadian HR/light model smoke — post ID341

Created: `2026-06-19 09:36:51 UTC+09:00`

This is a no-submit, in-memory model smoke. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, persisted model, or DACON submission.

## Feature input

- Matrix: `data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet`
- Aligned train rows: `450`
- Feature columns used: `48`
- Missing rate: `0.000000`
- Test rows present but unused for model smoke: `250`

## Selected target configs

| target | candidate | wins | mean Δ341 | max Δ341 |
|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |

## Top rows by target

### `Q3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_circadian_phase_lr_c0_20_w0.005` | 2 | +0.000007 | +0.000133 |
| 2 | `blend_id341_circadian_phase_lr_c0_20_w0.005` | 2 | +0.000007 | +0.000133 |
| 3 | `blend_id178_circadian_phase_lr_c0_08_w0.005` | 2 | +0.000014 | +0.000132 |
| 4 | `blend_id341_circadian_phase_lr_c0_08_w0.005` | 2 | +0.000014 | +0.000132 |
| 5 | `blend_id178_circadian_phase_lr_c0_20_w0.01` | 2 | +0.000021 | +0.000274 |

### `S3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_circadian_phase_lr_c0_20_w0.005` | 2 | +0.000063 | +0.000205 |
| 2 | `blend_id341_circadian_phase_lr_c0_20_w0.005` | 2 | +0.000063 | +0.000205 |
| 3 | `blend_id178_circadian_phase_lr_c0_20_w0.01` | 2 | +0.000134 | +0.000416 |
| 4 | `blend_id341_circadian_phase_lr_c0_20_w0.01` | 2 | +0.000134 | +0.000416 |
| 5 | `blend_id178_circadian_phase_lr_c0_20_w0.025` | 2 | +0.000403 | +0.001077 |

### `S2`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_circadian_phase_lr_c0_02_w0.005` | 2 | +0.000058 | +0.000159 |
| 2 | `blend_id341_circadian_phase_lr_c0_02_w0.005` | 2 | +0.000058 | +0.000159 |
| 3 | `blend_id178_circadian_phase_lr_c0_02_w0.01` | 2 | +0.000126 | +0.000329 |
| 4 | `blend_id341_circadian_phase_lr_c0_02_w0.01` | 2 | +0.000126 | +0.000329 |
| 5 | `blend_id178_circadian_phase_lr_c0_02_w0.025` | 2 | +0.000392 | +0.000907 |

## Decision

- Any circadian HR/light signal found: `False`
- Selected targets: `(none)`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If selected targets exist, run a no-artifact geometry gate before any artifact contract. Otherwise close the circadian HR/light local model lane and return to the explicit-approved upstream route.

## Boundary

This report can justify only another no-artifact geometry check. It is not a Public 0.58 claim and does not create ID342 or authorize DACON/Kaggle actions.

