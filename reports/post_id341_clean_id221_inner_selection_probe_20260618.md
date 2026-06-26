# Post-ID341 clean ID221 inner-selection probe

Created: `2026-06-18 05:30:02 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## OOF summary

- Candidate loss: `0.534923`
- Δ vs ID178: `-0.003715`
- Δ vs ID218: `+0.000820`
- Δ vs historical ID221: `+0.001933`
- Δ vs ID341: `-0.003661`

## Split gate

| split | loss | Δ178 | Δ218 | Δ221 | Δ341 |
|---|---:|---:|---:|---:|---:|
| `subject_hole_5` | 0.534234 | -0.004258 | +0.000272 | +0.001408 | -0.004203 |
| `subject_timeseries_5` | 0.536251 | -0.003496 | +0.000681 | +0.001733 | -0.003436 |
| `actual_test_band_holdout` | 0.539700 | -0.001904 | +0.000716 | +0.002042 | -0.001764 |
| `test_boundary_neighbor2_holdout` | 0.578640 | -0.007650 | +0.001123 | +0.004432 | -0.007497 |
| `test_boundary_neighbor5_holdout` | 0.555976 | -0.002010 | -0.000048 | +0.001194 | -0.001944 |
| `loso_10` | 0.538423 | -0.004042 | +0.000796 | +0.001752 | -0.003987 |

## Selection summary

- Active selections: `78` / `161`

Top selected configs:

```json
[
  [
    "base",
    83
  ],
  [
    "date_time_k1_dw0_w0.03",
    22
  ],
  [
    "screen_app_k1_dw0_w0.05",
    8
  ],
  [
    "date_time_k1_dw0.1_w0.03",
    6
  ],
  [
    "screen_app_k1_dw0.03_w0.05",
    6
  ],
  [
    "date_time_k1_dw0.03_w0.02",
    5
  ],
  [
    "screen_app_k1_dw0_w0.03",
    4
  ],
  [
    "light_hr_k3_dw0.03_w0.07",
    3
  ],
  [
    "date_time_k1_dw0_w0.01",
    3
  ],
  [
    "date_time_k1_dw0.03_w0.01",
    3
  ],
  [
    "move_gps_k7_dw0.03_w0.03",
    2
  ],
  [
    "light_hr_k3_dw0_w0.07",
    2
  ],
  [
    "light_hr_k3_dw0_w0.01",
    2
  ],
  [
    "date_time_k1_dw0.1_w0.01",
    2
  ],
  [
    "light_hr_k3_dw0.03_w0.005",
    1
  ],
  [
    "date_time_k1_dw0_w0.02",
    1
  ],
  [
    "screen_app_k1_dw0.03_w0.01",
    1
  ],
  [
    "date_time_k1_dw0_w0.005",
    1
  ],
  [
    "screen_app_k1_dw0.03_w0.005",
    1
  ],
  [
    "move_gps_k7_dw0.03_w0.07",
    1
  ]
]
```

## Test geometry preview

- Mean/max Δ vs ID178: `0.036173` / `0.226519`
- Mean/max Δ vs ID341: `0.036053` / `0.226519`
- Projection onto ID287 failed direction: `-0.0154`
- Projection onto ID329 gap direction: `+0.3484`

## Decision

- Inner selection improves ID218: `False`
- Beats ID178 primary: `True`
- Beats ID341 primary: `True`
- Beats historical ID221 primary: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: If this probe improves ID218 and keeps low failed-direction exposure, implement the same inner-selection contract in the Kaggle clean-backbone runner. Otherwise treat clean ID221 as an anchor/control only.

