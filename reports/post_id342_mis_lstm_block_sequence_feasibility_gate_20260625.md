# Post-ID342 MIS-LSTM block-sequence feasibility gate

Created: `2026-06-25 05:40:08 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Domain Read

- Q3 uses day/evening load and transition blocks. S3 uses pre-sleep/onset blocks with no-screen, dark, charging, stillness, and rest channels. S4 uses post-onset/morning-tail interruption blocks. This is a low-capacity feasibility proxy for MIS-LSTM, not a neural runner.

## Feature Contract

- Window: `12.0` to `36.0` relative hours
- Rows: `700` (`450` train, `250` test)
- Numeric feature columns before rank: `1257`
- Numeric feature columns used: `720`
- Mean missing rate: `0.2717738095238096`

| source | raw rows | window rows | feature count |
|---|---:|---:|---:|
| `wHr` | `382918` | `382918` | `264` |
| `wlight` | `633741` | `633741` | `84` |
| `mlight` | `96258` | `96258` | `84` |
| `mScreenStatus` | `939653` | `939653` | `84` |
| `mACStatus` | `939896` | `939896` | `36` |
| `mActivity` | `961062` | `961062` | `108` |
| `wPedo` | `748100` | `748100` | `300` |

## Families

| family | feature count | examples |
|---|---:|---|
| `all_block_sequence` | 720 | `mis_block4_day_sleep_pedo_b16_20_step_sum`, `mis_block4_day_sleep_pedo_b12_16_step_sum`, `mis_block3_evening_sleep_pedo_b18_21_step_sum`, `mis_block4_day_sleep_pedo_b16_20_distance_sum`, `mis_block4_day_sleep_pedo_b12_16_distance_sum` |
| `q3_day_evening_load` | 635 | `mis_block4_day_sleep_pedo_b16_20_step_sum`, `mis_block4_day_sleep_pedo_b12_16_step_sum`, `mis_block3_evening_sleep_pedo_b18_21_step_sum`, `mis_block4_day_sleep_pedo_b16_20_distance_sum`, `mis_block4_day_sleep_pedo_b12_16_distance_sum` |
| `s3_sleep_onset_blocks` | 458 | `mis_block3_evening_sleep_pedo_b18_21_step_sum`, `mis_block3_evening_sleep_pedo_b18_21_distance_sum`, `mis_block4_day_sleep_screen_b20_24_screen_off_sum`, `mis_block4_day_sleep_screen_b16_20_screen_off_sum`, `mis_block4_day_sleep_screen_b12_16_screen_off_sum` |
| `s4_sleep_fragment_blocks` | 465 | `mis_block3_evening_sleep_pedo_b18_21_step_sum`, `mis_block3_evening_sleep_pedo_b18_21_distance_sum`, `mis_block4_day_sleep_screen_b16_20_screen_on_sum`, `mis_block4_day_sleep_screen_b20_24_screen_on_sum`, `mis_block4_day_sleep_screen_b12_16_screen_on_sum` |
| `continuous_channels` | 498 | `mis_block4_day_sleep_pedo_b16_20_step_sum`, `mis_block4_day_sleep_pedo_b12_16_step_sum`, `mis_block3_evening_sleep_pedo_b18_21_step_sum`, `mis_block4_day_sleep_pedo_b16_20_distance_sum`, `mis_block4_day_sleep_pedo_b12_16_distance_sum` |
| `sparse_event_channels` | 230 | `mis_block4_day_sleep_screen_b20_24_screen_off_sum`, `mis_block4_day_sleep_screen_b16_20_screen_off_sum`, `mis_block4_day_sleep_screen_b12_16_screen_off_sum`, `mis_block4_day_sleep_screen_b16_20_screen_on_sum`, `mis_block4_day_sleep_screen_b20_24_screen_on_sum` |

## Target Scan

| target | selected | eligible | best config | wins | mean delta | max delta |
|---|---|---:|---|---:|---:|---:|
| `Q3` | `False` | 0 | `q3_id342_mis_blockseq_s3_sleep_onset_blocks_hgb_l7_w0.005` | 2 | 0.000047434 | 0.000308601 |
| `S3` | `False` | 0 | `s3_id342_mis_blockseq_all_block_sequence_hgb_l7_w0.03` | 3 | -0.000239324 | 0.000535092 |
| `S4` | `True` | 11 | `s4_id342_mis_blockseq_sparse_event_channels_lr_c0_10_w0.03` | 4 | -0.001468349 | 0.000715146 |

## Decision

- Selected targets: `['S4']`
- Estimated all-target delta vs ID342: `-0.000209764`
- Projected Public proxy: `0.595873488`
- Gap to Public 0.55: `0.045873488`
- Final gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: If this gate fails, do not launch a MIS-LSTM Kaggle runner. If it passes, run a separate no-submit artifact/runner contract review before any remote execution.
