# Sleep-boundary transition v2 gate — post ID341

Created: `2026-06-18 04:42:05 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Selected target configs

| target | candidate | wins | mean Δ | max Δ |
|---|---|---:|---:|---:|
| `Q1` | `Q1_id178_hgb_l5_l2_2_w0.05` | 4 | -0.002109 | +0.000086 |
| `S1` | `S1_id178_lr_c0_2_w0.03` | 4 | -0.001482 | +0.000565 |
| `S2` | `S2_id178_lr_c0_05_w0.1` | 4 | -0.002460 | +0.000299 |
| `S4` | `S4_id178_lr_c0_05_w0.1` | 5 | -0.004968 | -0.000621 |

## Assembled candidate by split

| split | candidate loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |
|---|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | 0.537585 | 0.538584 | -0.000998 | 0.538638 | -0.001053 | `True` |
| `subject_timeseries_5` | 0.538929 | 0.539688 | -0.000758 | 0.539747 | -0.000818 | `True` |
| `actual_test_band_holdout` | 0.539343 | 0.541465 | -0.002122 | 0.541604 | -0.002261 | `True` |
| `test_boundary_neighbor2_holdout` | 0.583755 | 0.586137 | -0.002382 | 0.586290 | -0.002535 | `True` |
| `test_boundary_neighbor5_holdout` | 0.555665 | 0.557920 | -0.002256 | 0.557986 | -0.002322 | `True` |
| `loso_10` | 0.538472 | 0.538584 | -0.000112 | 0.538638 | -0.000166 | `True` |

## Test geometry / public-risk proxy

- Local CV proxy (subject_hole_5): `0.537585`
- Mean/max Δ vs ID178: `0.007552` / `0.095181`
- Mean/max Δ vs ID341: `0.007429` / `0.095181`
- Failed-direction projection: `-0.001901`
- Geometry proxy: `0.606708`
- Local+recent-gap proxy: `0.600358`
- Conservative proxy: `0.606708`
- Beats ID341 proxy: `False`
- Public `0.58` plausible: `False`

## Decision

- Selected targets: `Q1, S1, S2, S4`
- All primary splits beat ID341: `True`
- Primary wins vs ID341: `5`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: If proxy remains above ID341, keep v2 as a feature-source lead but do not promote. If it beats ID341 proxy in a future v3, require full artifact contract with unique ID > 341.

