# Post-ID342 tiny temporal-CNN Q3/S3 smoke

Created: `2026-06-24 03:25:23 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Feature Contract

- Source: 44 half-hour multi-channel raw-sensor tensor plus label-free calendar/progress features
- Research read: This is the smallest local proxy for multi-channel image/sequence lifelog models: it keeps the time axis intact and lets a tiny Conv1d encoder learn local temporal motifs.
- Train rows: `450`
- Channels/bins: `68` x `44`
- Calendar/progress features: `8`

## Target Results

| target | selected | best config | wins | mean delta | max delta |
|---|---:|---|---:|---:|---:|
| Q3 | `False` | `Q3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.005` | 0 | +0.000275 | +0.000577 |
| S3 | `False` | `S3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.005` | 2 | +0.000012 | +0.000295 |

## Top Rows

### `Q3`

| rank | config | wins | mean delta | max delta |
|---:|---|---:|---:|---:|
| 1 | `Q3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.005` | 0 | +0.000275 | +0.000577 |
| 2 | `Q3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.01` | 0 | +0.000561 | +0.001163 |
| 3 | `Q3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.02` | 0 | +0.001164 | +0.002367 |
| 4 | `Q3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.03` | 0 | +0.001809 | +0.003610 |
| 5 | `Q3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.05` | 0 | +0.003219 | +0.006217 |

### `S3`

| rank | config | wins | mean delta | max delta |
|---:|---|---:|---:|---:|
| 1 | `S3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.005` | 2 | +0.000012 | +0.000295 |
| 2 | `S3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.01` | 2 | +0.000037 | +0.000599 |
| 3 | `S3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.02` | 2 | +0.000125 | +0.001234 |
| 4 | `S3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.03` | 2 | +0.000264 | +0.001904 |
| 5 | `S3_id342_tiny_temporal_cnn_tinycnn_c18_e55_seed34201_w0.05` | 2 | +0.000681 | +0.003345 |

## Decision

- Selected targets: `[]`
- Tiny temporal CNN has Q3/S3 signal: `False`
- Partial signal targets: `[]`
- Final gate pass: `False`
- Projected public proxy from ID342: `0.596083252`
- Gap to Public 0.55 after proxy: `+0.046083252`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If this local neural proxy fails, do not launch a Kaggle no-submit run for the same architecture. If it passes, use Kaggle only for parity/review evidence.
