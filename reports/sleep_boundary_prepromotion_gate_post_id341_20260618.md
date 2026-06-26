# Sleep-boundary prepromotion gate — post ID341

Created: `2026-06-18 04:46:51 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

Candidate slug: `id341_sleep_boundary_v3_scale005_q1_s1_s2_s4_micro`
Scale: `0.05`

## Effective target weights

| target | effective weight | base config |
|---|---:|---|
| `Q1` | 0.002500 | `Q1_id178_hgb_l5_l2_2_w0.05` |
| `S1` | 0.001500 | `S1_id178_lr_c0_2_w0.03` |
| `S2` | 0.005000 | `S2_id178_lr_c0_05_w0.1` |
| `S4` | 0.005000 | `S4_id178_lr_c0_05_w0.1` |

## Split gate

| split | loss | ID341 loss | Δ vs ID341 | pass |
|---|---:|---:|---:|---|
| `subject_hole_5` | 0.538511 | 0.538584 | -0.000072 | `True` |
| `subject_timeseries_5` | 0.539620 | 0.539688 | -0.000068 | `True` |
| `actual_test_band_holdout` | 0.541317 | 0.541465 | -0.000147 | `True` |
| `test_boundary_neighbor2_holdout` | 0.585981 | 0.586137 | -0.000157 | `True` |
| `test_boundary_neighbor5_holdout` | 0.557766 | 0.557920 | -0.000154 | `True` |
| `loso_10` | 0.538536 | 0.538584 | -0.000047 | `True` |

## Geometry and public context

- Proxy: `0.601697`
- Mean/max Δ vs ID341: `0.000371` / `0.004759`
- Failed-direction projection: `-0.000176`
- Proxy minus Public ID178: `+0.000286`
- Proxy minus Public ID329: `-0.000094`

## Gate checks

- `all_splits_beat_id341`: `True`
- `max_delta_vs_341_within_ceiling`: `True`
- `mean_delta_vs_341_within_ceiling`: `True`
- `failed_projection_non_positive`: `True`
- `proxy_below_current_id341_proxy`: `True`
- `proxy_below_best_actual_public_anchor_id178`: `False`
- `public_058_plausible`: `False`

## Decision

- Reviewable feature lead: `True`
- Promote submission artifact: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Why: The scale-0.05 row is clean as a low-movement feature lead, but its conservative proxy remains above the best actual public anchor ID178 and far above Public 0.58. Keep it as a source lead; do not create ID342 yet.
- Next action: Use sleep-boundary features inside the upstream/new-source lane, or search for an additional independent source that lowers the proxy below the ID178 public anchor before promotion.

