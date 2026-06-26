# ID338 material Q3/S3 no-submit probe

Created: `2026-06-23 23:49:13 UTC+09:00`

This is report-only. It creates no full-test predictions, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Feature Inputs

- Total feature columns: `322`
- All sources train-aligned: `True`

| source | features | mean missing | path |
|---|---:|---:|---|
| `phone_wearable_discordance` | 85 | 0.0000 | `data/processed/post_id341_phone_wearable_discordance_features_v1.parquet` |
| `proximity_ambience` | 80 | 0.0000 | `data/processed/post_id341_proximity_ambience_cooccurrence_features_v1.parquet` |
| `gps_path_topology` | 32 | 0.0000 | `data/processed/post_id341_gps_path_topology_features_v1.parquet` |
| `circadian_hr_light` | 48 | 0.0000 | `data/processed/post_id341_circadian_hr_light_phase_features_v1.parquet` |
| `usage_semantic` | 77 | 0.0000 | `data/processed/post_id341_usage_semantic_burst_markov_features_v1.parquet` |

## Selected Target Configs

| target | candidate | source | wins | mean Δ338 | max Δ338 | mean movement | failed proj |
|---|---|---|---:|---:|---:|---:|---:|
| `S3` | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.005` | `phone_wearable_discordance` | 4 | -0.000062 | 0.000058 | 0.000573 | 0.0005 |

## Top Rows By Target

### `Q3`

| rank | candidate | source | wins | mean Δ338 | max Δ338 | mean movement | failed proj |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `blend_id338_circadian_hr_light_lr_c0_20_w0.0025` | `circadian_hr_light` | 2 | 0.000005 | 0.000074 | 0.000288 | 0.0004 |
| 2 | `blend_id338_circadian_hr_light_lr_c0_08_w0.0025` | `circadian_hr_light` | 2 | 0.000008 | 0.000073 | 0.000270 | 0.0004 |
| 3 | `blend_id338_circadian_hr_light_lr_c0_20_w0.005` | `circadian_hr_light` | 2 | 0.000012 | 0.000150 | 0.000576 | 0.0003 |
| 4 | `blend_id338_circadian_hr_light_lr_c0_08_w0.005` | `circadian_hr_light` | 2 | 0.000017 | 0.000148 | 0.000539 | 0.0003 |
| 5 | `blend_id338_circadian_hr_light_lr_c0_20_w0.01` | `circadian_hr_light` | 2 | 0.000031 | 0.000306 | 0.001153 | 0.0002 |
| 6 | `blend_id338_circadian_hr_light_lr_c0_20_w0.02` | `circadian_hr_light` | 2 | 0.000092 | 0.000639 | 0.002306 | -0.0001 |
| 7 | `blend_id338_circadian_hr_light_lr_c0_20_w0.035` | `circadian_hr_light` | 2 | 0.000236 | 0.001189 | 0.004035 | -0.0005 |
| 8 | `blend_id338_circadian_hr_light_lr_c0_20_w0.05` | `circadian_hr_light` | 2 | 0.000444 | 0.001797 | 0.005764 | -0.0010 |
| 9 | `blend_id338_circadian_hr_light_lr_c0_03_w0.0025` | `circadian_hr_light` | 1 | 0.000017 | 0.000074 | 0.000250 | 0.0004 |
| 10 | `blend_id338_circadian_hr_light_lr_c0_01_w0.0025` | `circadian_hr_light` | 1 | 0.000035 | 0.000114 | 0.000242 | 0.0004 |

### `S3`

| rank | candidate | source | wins | mean Δ338 | max Δ338 | mean movement | failed proj |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.02` | `phone_wearable_discordance` | 4 | -0.000192 | 0.000267 | 0.002292 | 0.0006 |
| 2 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.01` | `phone_wearable_discordance` | 4 | -0.000115 | 0.000123 | 0.001146 | 0.0005 |
| 3 | `blend_id338_phone_wearable_discordance_lr_c0_08_w0.01` | `phone_wearable_discordance` | 4 | -0.000088 | 0.000164 | 0.001074 | 0.0005 |
| 4 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.005` | `phone_wearable_discordance` | 4 | -0.000062 | 0.000058 | 0.000573 | 0.0005 |
| 5 | `blend_id338_phone_wearable_discordance_lr_c0_08_w0.005` | `phone_wearable_discordance` | 4 | -0.000049 | 0.000080 | 0.000537 | 0.0005 |
| 6 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.0025` | `phone_wearable_discordance` | 4 | -0.000032 | 0.000029 | 0.000286 | 0.0005 |
| 7 | `blend_id338_phone_wearable_discordance_lr_c0_08_w0.0025` | `phone_wearable_discordance` | 4 | -0.000025 | 0.000039 | 0.000269 | 0.0005 |
| 8 | `blend_id338_phone_wearable_discordance_lr_c0_20_w0.035` | `phone_wearable_discordance` | 3 | -0.000240 | 0.000525 | 0.004010 | 0.0007 |
| 9 | `blend_id338_phone_wearable_discordance_lr_c0_08_w0.035` | `phone_wearable_discordance` | 3 | -0.000159 | 0.000655 | 0.003760 | 0.0006 |
| 10 | `blend_id338_phone_wearable_discordance_lr_c0_08_w0.02` | `phone_wearable_discordance` | 3 | -0.000141 | 0.000347 | 0.002149 | 0.0006 |

## Decision

- Probe completed: `True`
- Selected targets: `['S3']`
- Any Q3/S3 signal found: `True`
- All Q3/S3 targets covered: `False`
- Public `0.58` evidence: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Next local allowed step: `id338_material_q3s3_no_artifact_geometry_gate`
- Next action: If selected targets exist, run only a no-artifact geometry gate against ID338/test-shape risk. Do not create a CSV/base/notebook/log artifact without a separate promotion contract.

