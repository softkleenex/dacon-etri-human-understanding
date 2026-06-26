# Post-ID342 raw temporal Q3/S3 sequence-image smoke

Created: `2026-06-24 03:11:50 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Time Window

- Anchor rule: Timestamps 00:00-11:59 are assigned to the previous lifelog date.
- Relative hours: `12` to `33`
- Focus targets: `Q3,S3`

## Coverage

- Raw total rows in source files: `6122179`
- Rows inside relative window: `5307398`
- Feature columns before alignment: `2278`
- Feature columns used: `420`
- Train/test any-feature coverage: `1.0000` / `1.0000`
- Train/test missing mean: `0.2416` / `0.2262`

## Target Results

| target | selected | best config | wins | mean delta | max delta |
|---|---:|---|---:|---:|---:|
| Q3 | `False` | `Q3_id342_raw_temporal_ssknn_k1_w0.05` | 3 | -0.001735 | +0.002487 |
| S3 | `False` | `S3_id342_raw_temporal_ssknn_k7_w0.005` | 2 | +0.000092 | +0.000222 |

## Decision

- Selected targets: `[]`
- Raw temporal has Q3/S3 signal: `False`
- Partial signal targets: `['Q3']`
- Final gate pass: `False`
- Projected public proxy from ID342: `0.596083252`
- Gap to Public 0.55 after proxy: `+0.046083252`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Do not create a DACON artifact from this smoke. If the final gate passes locally, use Kaggle no-submit for parity and artifact-review evidence only. If it fails, keep the raw temporal representation as negative evidence and move to a stronger sequence model.
