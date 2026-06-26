# Post-ID341 clean ID226 calibration smoke

Created: `2026-06-18 05:36:44 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## OOF summary

- Candidate loss: `0.534441`
- Δ vs ID178: `-0.004197`
- Δ vs ID218: `+0.000338`
- Δ vs historical ID221: `+0.001451`
- Δ vs historical ID226: `+0.042670`
- Δ vs ID341: `-0.004143`

## Split gate

| split | loss | Δ178 | Δ218 | Δ221 | Δ226 | Δ341 |
|---|---:|---:|---:|---:|---:|---:|
| `subject_hole_5` | 0.535667 | -0.002824 | +0.001706 | +0.002842 | +0.044100 | -0.002770 |
| `subject_timeseries_5` | 0.538870 | -0.000877 | +0.003299 | +0.004351 | +0.046380 | -0.000818 |
| `actual_test_band_holdout` | 0.539855 | -0.001749 | +0.000871 | +0.002197 | +0.040672 | -0.001609 |
| `test_boundary_neighbor2_holdout` | 0.584075 | -0.002214 | +0.006558 | +0.009868 | +0.064950 | -0.002062 |
| `test_boundary_neighbor5_holdout` | 0.559275 | +0.001289 | +0.003250 | +0.004493 | +0.052393 | +0.001354 |
| `loso_10` | 0.538068 | -0.004397 | +0.000442 | +0.001397 | +0.043109 | -0.004341 |

## Selection summary

- Active selections: `153` / `161`

Top selected configs:

```json
[
  [
    "temp_t0.85_s+0.00",
    63
  ],
  [
    "temp_t0.85_s-0.12",
    42
  ],
  [
    "iso_logit_a0.1",
    11
  ],
  [
    "iso_logit_a0.05",
    11
  ],
  [
    "q16_logit_a0.2",
    9
  ],
  [
    "base",
    8
  ],
  [
    "temp_t0.85_s+0.12",
    7
  ],
  [
    "iso_prob_a0.1",
    3
  ],
  [
    "temp_t1_s+0.12",
    3
  ],
  [
    "temp_t1_s-0.12",
    1
  ],
  [
    "q8_logit_a0.2",
    1
  ],
  [
    "q8_logit_a0.1",
    1
  ],
  [
    "q8_logit_a0.05",
    1
  ]
]
```

## Test geometry preview

- Mean/max Δ vs ID178: `0.040719` / `0.251618`
- Mean/max Δ vs ID341: `0.040603` / `0.251618`
- Mean/max Δ vs historical ID226: `0.104233` / `0.605588`
- Projection onto ID287 failed direction: `+0.0395`
- Projection onto ID329 gap direction: `-0.0558`

## Decision

- Clean ID226 calibration signal: `False`
- Beats ID178 primary: `False`
- Beats ID341 primary: `False`
- Beats historical ID226 primary: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: Use this as a clean ID226 layer read only. If calibration does not beat ID218, do not replay ID227/228/237; move to a stronger raw/source model rather than reviving ID172-style full-OOF calibration.

