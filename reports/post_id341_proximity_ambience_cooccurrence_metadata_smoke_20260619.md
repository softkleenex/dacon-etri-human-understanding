# Post-ID341 proximity/ambience co-occurrence metadata smoke

Created: `2026-06-23 18:16:11 UTC+09:00`

This is report-only. It creates no feature matrix, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, model, or DACON submission.

## Decision

- Metadata smoke passed: `True`
- Source family: `proximity_ambience_cooccurrence_v1`
- Target focus: `Q3,S3`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_proximity_ambience_cooccurrence_feature_spec_no_matrix`
- Next action: Metadata smoke passed. Next may only be a report-only feature spec; do not write a feature matrix, train, create ID342, run Kaggle, or submit.

## Bin profile

- bin_count: `96580`
- subjects: `10`
- wifi_bins: `76321`
- ble_bins: `21826`
- proximity_bins: `76794`
- ambience_bins: `95448`
- evening_bins: `24125`
- late_bins: `24004`
- proximity_social_cooccur_bins: `17073`
- proximity_place_cooccur_bins: `19388`
- screen_proximity_cooccur_bins: `33835`
- evening_proximity_social_cooccur_bins: `5571`
- late_proximity_social_cooccur_bins: `3458`
- unique_wifi_tokens_count_only: `125239`
- unique_ble_tokens_count_only: `350456`
- median_wifi_devices_per_wifi_bin: `13.0`
- median_ble_devices_per_ble_bin: `8.0`
- speech_bins_ge_010: `18879`
- crowd_bins_ge_010: `67`
- inside_outside_vehicle_bins_ge_010: `22118`

## Coverage by split

| source | split | expected days | covered days | coverage |
|---|---|---:|---:|---:|
| `wifi` | `test` | 250 | 246 | 0.9840 |
| `wifi` | `train` | 450 | 439 | 0.9756 |
| `ble` | `test` | 250 | 235 | 0.9400 |
| `ble` | `train` | 450 | 416 | 0.9244 |
| `ambience` | `test` | 250 | 250 | 1.0000 |
| `ambience` | `train` | 450 | 450 | 1.0000 |
| `screen` | `test` | 250 | 250 | 1.0000 |
| `screen` | `train` | 450 | 450 | 1.0000 |
| `proximity_any_bins` | `test` | 250 | 246 | 0.9840 |
| `proximity_any_bins` | `train` | 450 | 439 | 0.9756 |
| `proximity_social_cooccur_bins` | `test` | 250 | 120 | 0.4800 |
| `proximity_social_cooccur_bins` | `train` | 450 | 221 | 0.4911 |

## Privacy note

Raw WiFi/BLE identifiers were counted transiently only; no token values are written to this report.
