# GPS path-topology model smoke — post ID341

Created: `2026-06-19 06:16:59 UTC+09:00`

This is a no-submit, in-memory model smoke. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, persisted model, or DACON submission.

## Feature input

- Matrix: `data/processed/post_id341_gps_path_topology_features_v1.parquet`
- Aligned train rows: `450`
- Feature columns used: `32`
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
| 1 | `blend_id178_gps_topology_lr_c0_02_w0.005` | 0 | +0.000294 | +0.000594 |
| 2 | `blend_id341_gps_topology_lr_c0_02_w0.005` | 0 | +0.000294 | +0.000594 |
| 3 | `blend_id178_gps_topology_lr_c0_08_w0.005` | 0 | +0.000338 | +0.000576 |
| 4 | `blend_id341_gps_topology_lr_c0_08_w0.005` | 0 | +0.000338 | +0.000576 |
| 5 | `blend_id178_gps_topology_lr_c0_20_w0.005` | 0 | +0.000359 | +0.000567 |

### `S3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_gps_topology_lr_c0_20_w0.01` | 3 | -0.000076 | +0.000019 |
| 2 | `blend_id341_gps_topology_lr_c0_20_w0.01` | 3 | -0.000076 | +0.000019 |
| 3 | `blend_id178_gps_topology_lr_c0_20_w0.005` | 3 | -0.000043 | +0.000008 |
| 4 | `blend_id341_gps_topology_lr_c0_20_w0.005` | 3 | -0.000043 | +0.000008 |
| 5 | `blend_id178_gps_topology_lr_c0_08_w0.005` | 3 | -0.000002 | +0.000094 |

## Decision

- Any GPS topology signal found: `False`
- Selected targets: `(none)`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If selected targets exist, run a no-artifact geometry gate before any artifact contract. Otherwise close the GPS topology local model lane and return to the explicit-approved upstream route.

## Boundary

This report can justify only another no-artifact geometry check. It is not a Public 0.58 claim and does not create ID342 or authorize DACON/Kaggle actions.

