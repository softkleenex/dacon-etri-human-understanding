# Post-ID342 cross-source split stability audit

Created: `2026-06-25 06:56:51 UTC+09:00`

This is report-only. It reads existing JSON reports only and creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- Extracted rows: `128`
- Representative rows: `13`
- Review signal targets: `[]`
- Stable-alignment targets: `['S3']`
- Test-neighbor-only targets: `[]`
- Gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Keep this as negative meta-evidence and design guidance only. S3 has a tiny stable phone/GPS micro-alignment, but its best proxy remains around the ID342 public band; S4 aligns mostly on test-neighbor holdouts and conflicts on LOSO. No target reaches a Public-0.58 review proxy, so do not create artifacts, Kaggle runs, submission-log rows, or DACON submissions.

## Target Summary

| target | status | families | stable both-win families | stable aligned splits | neighbor aligned splits | best public proxy |
|---|---|---:|---|---:|---:|---:|
| `Q3` | `no_cross_source_alignment` | 1 | `[]` | 0 | 0 | 0.596090028 |
| `S3` | `stable_alignment_but_proxy_too_weak` | 6 | `['gps_path_topology', 'phone_wearable_discordance']` | 2 | 3 | 0.596049063 |
| `S4` | `mixed_unstable_alignment` | 5 | `['raw_hr_light_event_sequence']` | 1 | 3 | 0.595763465 |
| `S2` | `no_cross_source_alignment` | 1 | `[]` | 0 | 0 | 0.595911728 |

## Split Alignment

| target | split | winning families | aligned win | conflicting signs | mean delta | best family | worst family |
|---|---|---:|---|---|---:|---|---|
| `Q3` | `subject_hole_5` | 0/1 | `False` | `False` | +0.000124007 | mis_lstm_block_sequence_proxy (+0.000124007) | mis_lstm_block_sequence_proxy (+0.000124007) |
| `Q3` | `actual_test_band_holdout` | 0/1 | `False` | `False` | +0.000308601 | mis_lstm_block_sequence_proxy (+0.000308601) | mis_lstm_block_sequence_proxy (+0.000308601) |
| `Q3` | `test_boundary_neighbor2_holdout` | 1/1 | `False` | `False` | -0.000204050 | mis_lstm_block_sequence_proxy (-0.000204050) | mis_lstm_block_sequence_proxy (-0.000204050) |
| `Q3` | `test_boundary_neighbor5_holdout` | 1/1 | `False` | `False` | -0.000073124 | mis_lstm_block_sequence_proxy (-0.000073124) | mis_lstm_block_sequence_proxy (-0.000073124) |
| `Q3` | `loso_10` | 0/1 | `False` | `False` | +0.000081735 | mis_lstm_block_sequence_proxy (+0.000081735) | mis_lstm_block_sequence_proxy (+0.000081735) |
| `S3` | `subject_hole_5` | 3/6 | `True` | `True` | +0.000155046 | mis_lstm_block_sequence_proxy (-0.000185784) | usage_semantic_burst_markov (+0.000833418) |
| `S3` | `actual_test_band_holdout` | 2/6 | `True` | `True` | -0.000078806 | usage_semantic_burst_markov (-0.001250210) | mis_lstm_block_sequence_proxy (+0.000535092) |
| `S3` | `test_boundary_neighbor2_holdout` | 5/6 | `True` | `True` | -0.000150854 | mis_lstm_block_sequence_proxy (-0.000481283) | gps_path_topology (+0.000036993) |
| `S3` | `test_boundary_neighbor5_holdout` | 6/6 | `True` | `False` | -0.000476450 | mis_lstm_block_sequence_proxy (-0.001518572) | usage_semantic_burst_markov (-0.000130147) |
| `S3` | `loso_10` | 2/6 | `True` | `True` | +0.000177738 | gps_path_topology (-0.000007081) | mis_lstm_block_sequence_proxy (+0.000453926) |
| `S4` | `subject_hole_5` | 2/5 | `True` | `True` | -0.000223149 | raw_hr_light_event_sequence (-0.003663323) | processed_sleep_fragmentation (+0.002046073) |
| `S4` | `actual_test_band_holdout` | 5/5 | `True` | `False` | -0.003525169 | external_solar_holiday_calendar (-0.006761661) | circadian_hr_light_phase (-0.001290222) |
| `S4` | `test_boundary_neighbor2_holdout` | 5/5 | `True` | `False` | -0.002673811 | external_solar_holiday_calendar (-0.005343554) | circadian_hr_light_phase (-0.000652027) |
| `S4` | `test_boundary_neighbor5_holdout` | 4/5 | `True` | `True` | +0.000201647 | mis_lstm_block_sequence_proxy (-0.001637097) | raw_hr_light_event_sequence (+0.004483614) |
| `S4` | `loso_10` | 1/5 | `False` | `True` | +0.000969270 | raw_hr_light_event_sequence (-0.000309351) | processed_sleep_fragmentation (+0.002148118) |
| `S2` | `subject_hole_5` | 1/1 | `False` | `False` | -0.000151105 | proximity_ambience_cooccurrence (-0.000151105) | proximity_ambience_cooccurrence (-0.000151105) |
| `S2` | `actual_test_band_holdout` | 1/1 | `False` | `False` | -0.001140203 | proximity_ambience_cooccurrence (-0.001140203) | proximity_ambience_cooccurrence (-0.001140203) |
| `S2` | `test_boundary_neighbor2_holdout` | 1/1 | `False` | `False` | -0.004365650 | proximity_ambience_cooccurrence (-0.004365650) | proximity_ambience_cooccurrence (-0.004365650) |
| `S2` | `test_boundary_neighbor5_holdout` | 1/1 | `False` | `False` | -0.002113679 | proximity_ambience_cooccurrence (-0.002113679) | proximity_ambience_cooccurrence (-0.002113679) |
| `S2` | `loso_10` | 0/1 | `False` | `False` | +0.001767300 | proximity_ambience_cooccurrence (+0.001767300) | proximity_ambience_cooccurrence (+0.001767300) |

## Representative Rows

| target | family | source detail | candidate | wins | stable wins | neighbor wins | mean delta | max delta |
|---|---|---|---|---:|---:|---:|---:|---:|
| `Q3` | `mis_lstm_block_sequence_proxy` | `s3_sleep_onset_blocks` | `q3_id342_mis_blockseq_s3_sleep_onset_blocks_hgb_l7_w0.005` | 2 | 0 | 2 | +0.000047434 | +0.000308601 |
| `S2` | `proximity_ambience_cooccurrence` | `proximity_ambience_cooccurrence` | `proximity_ambience_cooccurrence_s2_lr_c0_20_w0.05` | 4 | 1 | 3 | -0.001200667 | +0.001767300 |
| `S3` | `phone_wearable_discordance` | `phone_wearable_discordance` | `phone_wearable_discordance_s3_lr_c0_20_w0.01` | 4 | 2 | 2 | -0.000085749 | +0.000137458 |
| `S3` | `gps_path_topology` | `gps_path_topology` | `gps_path_topology_s3_lr_c0_20_w0.01` | 3 | 2 | 1 | -0.000052011 | +0.000036993 |
| `S3` | `mis_lstm_block_sequence_proxy` | `all_block_sequence` | `s3_id342_mis_blockseq_all_block_sequence_hgb_l7_w0.03` | 3 | 1 | 2 | -0.000239324 | +0.000535092 |
| `S3` | `usage_semantic_burst_markov` | `usage_semantic_burst_markov` | `usage_semantic_burst_markov_s3_lr_c0_20_w0.025` | 3 | 0 | 3 | -0.000097217 | +0.000833418 |
| `S3` | `raw_hr_light_event_sequence` | `s4_waso_fragmentation` | `s3_id342_sequence_hr_light_event_s4_waso_fragmentation_lr_c0_30_w0.01` | 3 | 0 | 3 | -0.000049330 | +0.000294551 |
| `S3` | `processed_sleep_fragmentation` | `s4_waso_fragmentation_core` | `s4_waso_fragmentation_core_s3_hgb_l7_w0.005` | 2 | 0 | 2 | +0.000075639 | +0.000378615 |
| `S4` | `raw_hr_light_event_sequence` | `s3_onset_settling` | `s4_id342_sequence_hr_light_event_s3_onset_settling_lr_c0_30_w0.05` | 4 | 2 | 2 | -0.001189470 | +0.004483614 |
| `S4` | `mis_lstm_block_sequence_proxy` | `sparse_event_channels` | `s4_id342_mis_blockseq_sparse_event_channels_lr_c0_10_w0.03` | 4 | 1 | 3 | -0.001468349 | +0.000715146 |
| `S4` | `external_solar_holiday_calendar` | `solar_holiday_interaction` | `solar_holiday_interaction_S4_lr_c0_08_w0.05` | 3 | 0 | 3 | -0.002238508 | +0.001238751 |
| `S4` | `processed_sleep_fragmentation` | `sleep_hr_awake_light` | `sleep_hr_awake_light_s4_hgb_l7_w0.05` | 3 | 0 | 3 | -0.000272711 | +0.002148118 |
| `S4` | `circadian_hr_light_phase` | `circadian_hr_light_phase` | `circadian_hr_light_phase_s4_lr_c0_02_w0.025` | 3 | 0 | 3 | -0.000082173 | +0.001053687 |
