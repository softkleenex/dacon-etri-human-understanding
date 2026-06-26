# Post-ID341 target gap budget

Created: `2026-06-23 18:07:16 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Gap math

- ID341 proxy: `0.603830`
- ID178 public anchor: `0.601411`
- Public-0.58 review cutoff used by gates: `0.590000`
- Required sum target delta to beat ID178: `-0.016929`
- Required sum target delta for Public-0.58 review: `-0.096808`

## Strict all-primary budget

- Selected targets: `Q1, S4`
- Sum target delta: `-0.006457`
- Projected proxy: `0.602907`
- Gap vs ID178: `+0.001496`
- Additional sum delta needed to beat ID178: `-0.010472`

| target | source | candidate | wins | mean Δ341 | max Δ341 | anchor | weight |
|---|---|---|---:|---:|---:|---|---:|
| Q1 | `block_a_sleep_residual` | `Q1_id178_sleep_residual_bridge_hgb_l4_l2_5_w0.05` | 5 | -0.001489 | -0.000058 | `id178` | 0.05 |
| Q2 |  |  |  |  |  |  |  |
| Q3 |  |  |  |  |  |  |  |
| S1 |  |  |  |  |  |  |  |
| S2 |  |  |  |  |  |  |  |
| S3 |  |  |  |  |  |  |  |
| S4 | `sleep_evening_combined_sources` | `S4_id178_lr_c0_05_w0.1` | 5 | -0.004968 | -0.000621 | `id178` | 0.1 |

## Watch 4/5 budget

- Selected targets: `Q1, S1, S2, S4`
- Sum target delta: `-0.011018`
- Projected proxy: `0.602256`
- Gap vs ID178: `+0.000844`
- Additional sum delta needed to beat ID178: `-0.005911`

| target | source | candidate | wins | mean Δ341 | max Δ341 | anchor | weight |
|---|---|---|---:|---:|---:|---|---:|
| Q1 | `sleep_evening_combined_sources` | `Q1_id178_hgb_l5_l2_2_w0.05` | 4 | -0.002109 | 0.000086 | `id178` | 0.05 |
| Q2 |  |  |  |  |  |  |  |
| Q3 |  |  |  |  |  |  |  |
| S1 | `sleep_evening_combined_sources` | `S1_id178_lr_c0_2_w0.03` | 4 | -0.001482 | 0.000565 | `id178` | 0.03 |
| S2 | `sleep_evening_combined_sources` | `S2_id178_lr_c0_05_w0.1` | 4 | -0.002460 | 0.000299 | `id178` | 0.1 |
| S3 |  |  |  |  |  |  |  |
| S4 | `sleep_evening_combined_sources` | `S4_id178_lr_c0_05_w0.1` | 5 | -0.004968 | -0.000621 | `id178` | 0.1 |

## Decision

- Strict budget beats ID178: `False`
- Watch budget beats ID178: `False`
- Strict budget Public-0.58 reviewable: `False`
- Watch budget Public-0.58 reviewable: `False`
- Missing strict targets: `Q2, Q3, S1, S2, S3`
- Recommended next focus: `Q2/Q3/S3 independent raw or upstream source signal before more Q1/S4 tuning`

Do not create ID342 from current local source leads. The strict all-primary budget remains above ID178; future work should seek independent Q2/Q3/S3 signal or run the explicit-approved no-submit Kaggle upstream rebuild.

