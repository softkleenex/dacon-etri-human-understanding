# Post-ID341 proximity/ambience co-occurrence feature spec

Created: `2026-06-23 18:16:11 UTC+09:00`

This is report-only. It creates no feature matrix, model, predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Feature spec ready: `True`
- Source family: `proximity_ambience_cooccurrence_v1`
- Target focus: `Q3,S3`
- Feature count: `80`
- Public `0.58` evidence: `False`
- Score route opened: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next local allowed step: `report_only_proximity_ambience_cooccurrence_feature_generation_dryrun_no_matrix`
- Next action: Feature spec is ready. Next may only be an in-memory dry run that checks 700-row alignment and finite bounded columns; do not write a feature matrix, train, create ID342, run Kaggle, or submit.

## Row contract

- key_columns: `['subject_id', 'sleep_date', 'lifelog_date']`
- expected_train_rows: `450`
- expected_test_rows: `250`
- expected_total_rows: `700`
- label_columns_allowed: `False`
- prediction_columns_allowed: `False`
- raw_proximity_token_columns_allowed: `False`

## Feature groups

- `wifi_proximity_richness`: `wifi_bins, wifi_devices_sum, wifi_strong_sum`
- `ble_proximity_richness`: `ble_bins, ble_devices_sum, ble_strong_sum`
- `ambience_social_place`: `speech_max, speech_mean, crowd_max, place_max, vehicle_music_max`
- `cross_sensor_cooccurrence`: `proximity_social_bins, proximity_place_bins, screen_proximity_bins, social_screen_proximity_bins`

## Windows

- `all_day`: full lifelog day / hours `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]`
- `morning`: 06:00-11:59 recovery/start context / hours `[6, 7, 8, 9, 10, 11]`
- `afternoon`: 12:00-17:59 daytime social/place context / hours `[12, 13, 14, 15, 16, 17]`
- `evening`: 18:00-23:59 late-day stimulation / hours `[18, 19, 20, 21, 22, 23]`
- `late`: 22:00-03:59 late reactivation band / hours `[22, 23, 0, 1, 2, 3]`

## Schema bounds

- max_feature_columns: `80`
- actual_feature_columns: `80`
- all_features_prefixed: `True`
- no_raw_identifier_features: `True`

## Leakage guards

- Build rows only from lifelog_date source timestamps and sample/train key contracts.
- Do not persist raw WiFi/BLE identifier values; use aggregate counts/ratios only.
- Use fixed ambience groups and fixed RSSI thresholds, not target-selected tokens.
- Any later preprocessing/model smoke must be fold-internal; this spec alone is not score evidence.
