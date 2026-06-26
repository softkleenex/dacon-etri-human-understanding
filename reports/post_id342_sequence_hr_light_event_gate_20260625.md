# Post-ID342 sequence HR/light event gate

Created: `2026-06-25 02:04:57 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Domain Read

- S3 approximates sleep-onset latency via settling into no-screen, darkness, charging, stillness, and HR drop. S4 approximates WASO via post-onset HR spikes, light/screen/movement interruptions, and fragmented low-HR/dark/still runs. All features are label-free raw-sequence summaries.

## Feature Contract

- Window: `18.0` to `36.0` relative hours
- Rows: `700` (`450` train, `250` test)
- Numeric feature columns before rank: `807`
- Numeric feature columns used: `640`
- Mean missing rate: `0.22245535714285714`

| source | raw rows | window rows | feature count |
|---|---:|---:|---:|
| `wHr` | `382918` | `207154` | `149` |
| `wlight` | `633741` | `439112` | `64` |
| `mlight` | `96258` | `71697` | `64` |
| `mScreenStatus` | `939653` | `700586` | `53` |
| `mACStatus` | `939896` | `700957` | `22` |
| `mActivity` | `961062` | `716397` | `59` |
| `wPedo` | `748100` | `533547` | `109` |

## Families

| family | feature count | examples |
|---|---:|---|
| `all_sequence_events` | 640 | `est_seq_screen_screen_on_window_rows_delta_prev1`, `est_seq_screen_screen_off_window_rows_delta_prev1`, `est_seq_screen_screen_off_longest_run_minutes_delta_prev1`, `est_seq_screen_screen_on_window_rows_dev_past3`, `est_seq_screen_screen_on_window_rows` |
| `s4_waso_fragmentation` | 276 | `est_seq_screen_screen_on_window_rows_delta_prev1`, `est_seq_screen_screen_on_window_rows_dev_past3`, `est_seq_screen_screen_on_window_rows`, `est_seq_screen_screen_on_window_rows_prev1`, `est_seq_hr_hr_rel_low_window_rows_delta_prev1` |
| `s3_onset_settling` | 295 | `est_seq_screen_screen_off_window_rows_delta_prev1`, `est_seq_screen_screen_off_longest_run_minutes_delta_prev1`, `est_seq_screen_screen_off_window_rows_dev_past3`, `est_seq_screen_screen_off_window_rows`, `est_seq_screen_screen_off_window_rows_prev1` |
| `physio_light_events` | 271 | `est_seq_hr_hr_rel_low_window_rows_delta_prev1`, `est_seq_hr_hr_spike_window_rows_delta_prev1`, `est_seq_hr_hr_rel_low_window_rows`, `est_seq_hr_hr_rel_low_window_rows_prev1`, `est_seq_hr_hr_rel_low_window_rows_dev_past3` |
| `screen_charge_inactivity` | 350 | `est_seq_screen_screen_on_window_rows_delta_prev1`, `est_seq_screen_screen_off_window_rows_delta_prev1`, `est_seq_screen_screen_off_longest_run_minutes_delta_prev1`, `est_seq_screen_screen_on_window_rows_dev_past3`, `est_seq_screen_screen_on_window_rows` |

## Target Scan

| target | selected | eligible | best config | wins | mean delta | max delta |
|---|---|---:|---|---:|---:|---:|
| `S4` | `True` | 5 | `s4_id342_sequence_hr_light_event_s3_onset_settling_lr_c0_10_w0.01` | 4 | -0.000254566 | 0.000695438 |
| `S3` | `False` | 0 | `s3_id342_sequence_hr_light_event_s4_waso_fragmentation_lr_c0_30_w0.01` | 3 | -0.000049330 | 0.000294551 |

## Decision

- Selected targets: `['S4']`
- Estimated all-target delta vs ID342: `-0.000036367`
- Projected Public proxy: `0.596046885`
- Gap to Public 0.55: `0.046046885`
- Final gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: If this gate fails, close local raw HR/light/screen event reconstruction as an immediate ID342 overlay route. If it passes but remains far above Public 0.55, keep it report-only and pivot to validation/public-gap redesign before any artifact.
