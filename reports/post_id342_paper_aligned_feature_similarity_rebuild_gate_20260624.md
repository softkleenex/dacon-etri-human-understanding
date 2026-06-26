# Post-ID342 paper-aligned feature-similarity rebuild gate

Created: `2026-06-24 03:52:30 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Research Priors

- [DACON 4th ETRI Human Understanding AI Paper Challenge data page](https://dacon.io/en/competitions/official/236468/data): The competition data items include smartphone charge/activity/ambience/BLE/GPS/light/screen/app/Wi-Fi plus smartwatch HR/light/steps; Q3 is pre-sleep stress and S3 is SOL guideline adherence.
- [ETRI Lifelog Dataset 2024 paper](https://arxiv.org/abs/2508.03698): The paper reports that Q1 and S3 baseline performance was strongest with 3-hour step-count features, while Q2 and Q3 were strongest with daily mobile app usage; demographics/weekday helped all targets.

## OOF Summary

- Candidate loss: `0.534649`
- ID342 loss: `0.534000`
- Delta vs ID342: `+0.000649`
- Delta vs ID178: `-0.003989`
- Delta vs ID221: `+0.001658`
- Primary wins vs ID342: `2`
- Primary mean/max delta vs ID342: `+0.000527` / `+0.001757`
- Projected public proxy: `0.596731761`
- Gap to Public 0.55: `+0.046731761`

## Split Gate

| split | loss | delta vs ID342 | delta vs ID178 | delta vs ID221 |
|---|---:|---:|---:|---:|
| subject_hole_5 | 0.534657 | +0.000808 | -0.003835 | +0.001831 |
| actual_test_band_holdout | 0.537714 | -0.000041 | -0.003890 | +0.000055 |
| test_boundary_neighbor2_holdout | 0.579748 | +0.001757 | -0.006542 | +0.005541 |
| test_boundary_neighbor5_holdout | 0.553997 | -0.000519 | -0.003989 | -0.000785 |
| loso_10 | 0.538372 | +0.000631 | -0.004093 | +0.001701 |

## Selection Summary

| target | top selections | full-train/test selected |
|---|---|---|
| `Q1` | `base:15, move_gps_k1_dw0.03_w0.03:2, move_gps_k1_dw0_w0.03:1` | `base` |
| `Q2` | `date_time_k1_dw0_w0.03:8, screen_app_k1_dw0.03_w0.01:2, screen_app_k1_dw0.03_w0.03:2, base:1, date_time_k1_dw0.03_w0.03:1` | `screen_app_k1_dw0.03_w0.01` |
| `Q3` | `date_time_k1_dw0_w0.03:13, date_time_k1_dw0_w0.01:2, screen_app_k1_dw0.03_w0.03:1, date_time_k1_dw0.03_w0.03:1, base:1` | `date_time_k1_dw0_w0.03` |
| `S1` | `base:18` | `base` |
| `S2` | `base:18` | `base` |
| `S3` | `move_gps_k1_dw0_w0.03:7, base:4, move_gps_k1_dw0_w0.01:3, light_hr_k1_dw0_w0.03:2, move_gps_k1_dw0.03_w0.03:1` | `move_gps_k1_dw0_w0.01` |
| `S4` | `date_time_k1_dw0_w0.03:8, date_time_k1_dw0.03_w0.03:7, base:2, light_hr_k1_dw0.03_w0.03:1` | `date_time_k1_dw0.03_w0.03` |

## Decision

- Gate pass: `False`
- Failed checks: `primary_wins_vs_id342,primary_mean_delta_material,primary_worst_delta_not_positive,public_proxy_low_enough,public_055_gap_low_enough`
- Artifact review allowed: `False`
- Create submission artifact: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Submit now: `False`
- Reason: The paper-aligned feature-similarity rebuild does not clear the strict ID342/Public-0.55 gate.
