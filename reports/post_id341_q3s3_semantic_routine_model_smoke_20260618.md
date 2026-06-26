# Q3/S3 semantic-routine model smoke — post ID341

Created: `2026-06-23 18:07:53 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Feature coverage

- Rows: `700` total / `450` train / `250` test
- Raw feature columns: `63`; total feature columns: `56`
- Train/test non-null row share: `1.000` / `1.000`
- Train/test missing mean: `0.271` / `0.257`
- All-null raw features: `['sem_usage_late_total_share', 'sem_usage_app_entropy', 'sem_wifi_late_item_share', 'sem_wifi_mean_rssi', 'sem_ble_late_item_share', 'sem_ble_mean_rssi', 'sem_gps_speed_mean', 'sem_gps_moving_share', 'sem_amb_label_entropy', 'sem_amb_top_score_mean', 'sem_amb_stress_audio_score_mean', 'sem_hr_late_mean', 'sem_hr_onset_mean']`
- Raw identifier columns exported: `[]`
- Exact lat/lon columns exported: `[]`

## Selected target configs

| target | candidate | wins | mean Δ341 | max Δ341 |
|---|---|---:|---:|---:|
| _(none)_ |  |  |  |  |

## Top rows by target

### `Q3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_sem_lr_c0_02_w0.005` | 0 | +0.000147 | +0.000392 |
| 2 | `blend_id341_sem_lr_c0_02_w0.005` | 0 | +0.000147 | +0.000392 |
| 3 | `blend_id178_sem_lr_c0_08_w0.005` | 0 | +0.000171 | +0.000386 |
| 4 | `blend_id341_sem_lr_c0_08_w0.005` | 0 | +0.000171 | +0.000386 |
| 5 | `blend_id178_sem_lr_c0_02_w0.01` | 0 | +0.000299 | +0.000787 |

### `S3`

| rank | candidate | wins | mean Δ341 | max Δ341 |
|---:|---|---:|---:|---:|
| 1 | `blend_id178_sem_lr_c0_08_w0.005` | 0 | +0.000287 | +0.000505 |
| 2 | `blend_id341_sem_lr_c0_08_w0.005` | 0 | +0.000287 | +0.000505 |
| 3 | `blend_id178_sem_lr_c0_02_w0.005` | 0 | +0.000332 | +0.000606 |
| 4 | `blend_id341_sem_lr_c0_02_w0.005` | 0 | +0.000332 | +0.000606 |
| 5 | `blend_id178_sem_lr_c0_08_w0.01` | 0 | +0.000583 | +0.001014 |

## Decision

- Model smoke completed: `True`
- Resource gate ready: `True`
- Selected targets: `[]`
- Q3 signal found: `False`
- S3 signal found: `False`
- All Q3/S3 targets covered: `False`
- Public `0.58` evidence: `False`
- Next local allowed step: `close_semantic_routine_model_smoke`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If selected targets exist, run only a no-artifact geometry gate. Otherwise close this local lane and return to the explicit-approved upstream rebuild path for Q3/S3.

