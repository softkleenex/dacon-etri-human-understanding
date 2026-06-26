# Subject residual sequence-drift smoke — post ID341

Created: `2026-06-18 07:28:55 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle run, or DACON submission.

## Feature coverage

- Base numeric columns selected: `56`
- Rows: `450` train / `250` test
- Sequence/residual feature columns: `399`
- Mean missing rate: train `0.0763`, test `0.0627`

Top selected base columns:

- `mlight_all_q25` — score `3.4142`, coverage `1.000`, variance `252.963`
- `ch2025_mUsageStats_app_game_sum` — score `2.7985`, coverage `0.978`, variance `585680766.069`
- `wlight_all_q25` — score `2.6143`, coverage `0.920`, variance `931.722`
- `slp_pedo_mid_steps` — score `2.4598`, coverage `0.784`, variance `14606.844`
- `mlight_morn_mean` — score `2.1951`, coverage `0.973`, variance `157663.378`
- `ch2025_mUsageStats_app_music_sum` — score `2.1731`, coverage `0.978`, variance `253368499288.166`
- `ch2025_mLight_light_mean_max` — score `2.1669`, coverage `1.000`, variance `113147073.401`
- `ch2025_mLight_light_max_max` — score `2.1669`, coverage `1.000`, variance `113147073.401`
- `mlight_all_max` — score `2.1669`, coverage `1.000`, variance `113147073.401`
- `ch2025_mUsageStats_app_tiktok_sum` — score `2.1664`, coverage `0.978`, variance `481411255947.088`
- `mlight_all_std` — score `2.1661`, coverage `1.000`, variance `914061.795`
- `wHr_slp_hr_spike_count` — score `2.1477`, coverage `0.700`, variance `267.576`

## Best by primary frozen split

| split | best candidate | loss | ID341 loss | Δ vs ID341 | ID178 loss | Δ vs ID178 | pass |
|---|---|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | `blend_id341_subject_residual_et_d4_l25_w0.005` | 0.538726 | 0.538584 | +0.000142 | 0.538638 | +0.000088 | `False` |
| `actual_test_band_holdout` | `blend_id341_subject_residual_et_d4_l25_w0.005` | 0.541589 | 0.541465 | +0.000124 | 0.541604 | -0.000015 | `False` |
| `test_boundary_neighbor2_holdout` | `blend_id341_subject_residual_et_d4_l25_w0.05` | 0.585628 | 0.586137 | -0.000509 | 0.586290 | -0.000662 | `True` |
| `test_boundary_neighbor5_holdout` | `blend_id341_subject_residual_hgb_l4_l2_5_w0.005` | 0.557963 | 0.557920 | +0.000042 | 0.557986 | -0.000024 | `False` |
| `loso_10` | `blend_id341_subject_residual_hgb_l4_l2_5_w0.005` | 0.538716 | 0.538584 | +0.000132 | 0.538638 | +0.000078 | `False` |

## Target-level smoke signal vs ID341

| target | primary wins | mean Δ | min Δ | max Δ |
|---|---:|---:|---:|---:|
| `Q1` | 5/5 | -0.001715 | -0.006466 | -0.000005 |
| `Q2` | 1/5 | -0.000552 | -0.002762 | +0.000000 |
| `Q3` | 1/5 | -0.000317 | -0.001587 | +0.000000 |
| `S1` | 1/5 | -0.000003 | -0.000014 | +0.000000 |
| `S2` | 4/5 | -0.000388 | -0.000664 | +0.000000 |
| `S3` | 2/5 | -0.000076 | -0.000374 | +0.000000 |
| `S4` | 3/5 | -0.002066 | -0.003984 | +0.000000 |

## In-memory test geometry

- Source model: `hgb_l4_l2_5`
- Mean absolute delta vs ID341 base test: `0.148293`
- Mean absolute delta vs ID178 base test: `0.148333`

## Decision

- All primary splits beat ID341: `False`
- Primary split wins vs ID341: `1`
- Mean primary Δ vs ID341: `-0.000014`
- Any target has three primary wins: `True`
- Source family has smoke signal: `True`
- Projected public proxy from ID341: `0.603816`
- Beats ID178 public anchor proxy: `False`
- Public 0.58 reviewable: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: If smoke signal is true, combine only the winning target/split configs with transfer-v2 and joint-risk gates before any artifact promotion. If false, do not promote this source; keep it as negative evidence and prioritize the Kaggle phase-2 raw multimodal runner.

