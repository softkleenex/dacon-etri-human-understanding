# Post-ID342 registered material-source rescan

Created: `2026-06-24 04:11:10 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Research Priors

- [ETRI Lifelog Dataset 2024 paper](https://arxiv.org/abs/2508.03698): Q1/S3 were strongest with 3-hour step-count features, Q2/Q3 with daily mobile app usage, and demographics/weekday helped across models.
- [DACON 4th ETRI data page](https://dacon.io/en/competitions/official/236468/data): The challenge exposes multimodal smartphone, wearable, proximity, GPS, light, app, and sleep-related lifelog streams for the seven Q/S targets.

## Decision

- Reviewed candidates: `450`
- Local gate pass count: `0`
- Best candidate: `proximity_ambience_cooccurrence_s2_lr_c0_20_w0.05`
- Best source/target: `proximity_ambience_cooccurrence` / `S2`
- Best mean all-target delta vs ID342: `-0.000171524`
- Best projected public proxy: `0.595911728`
- Best gap to Public 0.55: `+0.045911728`
- Artifact review allowed: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Submit now: `False`
- Reason: No registered material-source row cleared the strict ID342/Public-0.55 local gate.

## Top Rows

| rank | pass | candidate | source | target | weight | wins | mean all Δ342 | max all Δ342 | target mean Δ342 | proxy | gap 0.55 |
|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_20_w0.05` | `proximity_ambience_cooccurrence` | `S2` | 0.05 | 4 | -0.000172 | +0.000252 | -0.001201 | 0.595912 | +0.045912 |
| 2 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_08_w0.05` | `proximity_ambience_cooccurrence` | `S2` | 0.05 | 4 | -0.000149 | +0.000211 | -0.001045 | 0.595934 | +0.045934 |
| 3 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_20_w0.025` | `proximity_ambience_cooccurrence` | `S2` | 0.025 | 4 | -0.000101 | +0.000086 | -0.000710 | 0.595982 | +0.045982 |
| 4 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_08_w0.025` | `proximity_ambience_cooccurrence` | `S2` | 0.025 | 4 | -0.000090 | +0.000070 | -0.000627 | 0.595994 | +0.045994 |
| 5 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_02_w0.05` | `proximity_ambience_cooccurrence` | `S2` | 0.05 | 3 | -0.000056 | +0.000211 | -0.000389 | 0.596028 | +0.046028 |
| 6 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_20_w0.01` | `proximity_ambience_cooccurrence` | `S2` | 0.01 | 4 | -0.000045 | +0.000023 | -0.000314 | 0.596038 | +0.046038 |
| 7 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_02_w0.025` | `proximity_ambience_cooccurrence` | `S2` | 0.025 | 3 | -0.000043 | +0.000078 | -0.000298 | 0.596041 | +0.046041 |
| 8 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_08_w0.01` | `proximity_ambience_cooccurrence` | `S2` | 0.01 | 4 | -0.000040 | +0.000018 | -0.000279 | 0.596043 | +0.046043 |
| 9 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_20_w0.005` | `proximity_ambience_cooccurrence` | `S2` | 0.005 | 4 | -0.000023 | +0.000009 | -0.000162 | 0.596060 | +0.046060 |
| 10 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_02_w0.01` | `proximity_ambience_cooccurrence` | `S2` | 0.01 | 3 | -0.000021 | +0.000023 | -0.000147 | 0.596062 | +0.046062 |
| 11 | `False` | `phone_wearable_discordance_s3_lr_c0_20_w0.025` | `phone_wearable_discordance` | `S3` | 0.025 | 3 | -0.000021 | +0.000055 | -0.000146 | 0.596062 | +0.046062 |
| 12 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_08_w0.005` | `proximity_ambience_cooccurrence` | `S2` | 0.005 | 4 | -0.000021 | +0.000007 | -0.000145 | 0.596063 | +0.046063 |
| 13 | `False` | `usage_semantic_burst_markov_s3_lr_c0_20_w0.025` | `usage_semantic_burst_markov` | `S3` | 0.025 | 3 | -0.000014 | +0.000119 | -0.000097 | 0.596069 | +0.046069 |
| 14 | `False` | `phone_wearable_discordance_s3_lr_c0_20_w0.01` | `phone_wearable_discordance` | `S3` | 0.01 | 4 | -0.000012 | +0.000020 | -0.000086 | 0.596071 | +0.046071 |
| 15 | `False` | `phone_wearable_discordance_s3_lr_c0_08_w0.025` | `phone_wearable_discordance` | `S3` | 0.025 | 3 | -0.000012 | +0.000067 | -0.000083 | 0.596071 | +0.046071 |
| 16 | `False` | `circadian_hr_light_phase_s4_lr_c0_02_w0.025` | `circadian_hr_light_phase` | `S4` | 0.025 | 3 | -0.000012 | +0.000151 | -0.000082 | 0.596072 | +0.046072 |
| 17 | `False` | `proximity_ambience_cooccurrence_s2_lr_c0_02_w0.005` | `proximity_ambience_cooccurrence` | `S2` | 0.005 | 3 | -0.000011 | +0.000010 | -0.000079 | 0.596072 | +0.046072 |
| 18 | `False` | `phone_wearable_discordance_s3_lr_c0_20_w0.05` | `phone_wearable_discordance` | `S3` | 0.05 | 2 | -0.000011 | +0.000129 | -0.000076 | 0.596072 | +0.046072 |
| 19 | `False` | `gps_path_topology_s3_lr_c0_20_w0.025` | `gps_path_topology` | `S3` | 0.025 | 2 | -0.000008 | +0.000022 | -0.000059 | 0.596075 | +0.046075 |
| 20 | `False` | `circadian_hr_light_phase_s4_lr_c0_02_w0.01` | `circadian_hr_light_phase` | `S4` | 0.01 | 3 | -0.000008 | +0.000057 | -0.000058 | 0.596075 | +0.046075 |
| 21 | `False` | `usage_semantic_burst_markov_s3_lr_c0_20_w0.01` | `usage_semantic_burst_markov` | `S3` | 0.01 | 3 | -0.000008 | +0.000046 | -0.000058 | 0.596075 | +0.046075 |
| 22 | `False` | `phone_wearable_discordance_s3_lr_c0_08_w0.01` | `phone_wearable_discordance` | `S3` | 0.01 | 3 | -0.000008 | +0.000025 | -0.000058 | 0.596075 | +0.046075 |
| 23 | `False` | `circadian_hr_light_phase_s4_lr_c0_08_w0.025` | `circadian_hr_light_phase` | `S4` | 0.025 | 3 | -0.000008 | +0.000165 | -0.000054 | 0.596076 | +0.046076 |
| 24 | `False` | `gps_path_topology_s3_lr_c0_20_w0.01` | `gps_path_topology` | `S3` | 0.01 | 3 | -0.000007 | +0.000005 | -0.000052 | 0.596076 | +0.046076 |
| 25 | `False` | `circadian_hr_light_phase_s4_lr_c0_08_w0.01` | `circadian_hr_light_phase` | `S4` | 0.01 | 3 | -0.000007 | +0.000062 | -0.000048 | 0.596076 | +0.046076 |
