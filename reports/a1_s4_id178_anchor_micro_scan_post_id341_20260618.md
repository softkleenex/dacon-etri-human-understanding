# A1 S4 ID178-anchor micro scan — post ID341

Created: `2026-06-18 04:02:31 UTC+09:00`

This is report-only. It creates no CSV, notebook/history artifact, base prediction artifact, submission-log row, or DACON submission.

- Base: `ID178`
- Source: `ID239`
- Target: `S4`
- Method: `prob`
- ID341 proxy reference: `0.603830`

## Top variants

| rank | variant | artifact review | primary pass | target primary pass | local | S4 Δ local | proxy | Δ proxy vs ID341 | max Δ178 | failed proj |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `id178_s4_id239_prob_w0.0025` | `False` | `True` | `True` | 0.538492 | -0.001018 | 0.603926 | +0.000096 | 0.001834 | +0.000632 |
| 2 | `id178_s4_id239_prob_w0.005` | `False` | `True` | `True` | 0.538347 | -0.002034 | 0.604099 | +0.000270 | 0.003668 | +0.001263 |
| 3 | `id178_s4_id239_prob_w0.0075` | `False` | `True` | `True` | 0.538203 | -0.003046 | 0.604273 | +0.000443 | 0.005503 | +0.001895 |
| 4 | `id178_s4_id239_prob_w0.01` | `False` | `True` | `True` | 0.538058 | -0.004056 | 0.604446 | +0.000616 | 0.007337 | +0.002526 |
| 5 | `id178_s4_id239_prob_w0.015` | `False` | `True` | `True` | 0.537771 | -0.006066 | 0.604793 | +0.000963 | 0.011005 | +0.003790 |
| 6 | `id178_s4_id239_prob_w0.02` | `False` | `True` | `True` | 0.537486 | -0.008065 | 0.605141 | +0.001311 | 0.014673 | +0.005053 |
| 7 | `id178_s4_id239_prob_w0.03` | `False` | `True` | `True` | 0.536919 | -0.012029 | 0.605840 | +0.002010 | 0.022010 | +0.007579 |
| 8 | `id178_s4_id239_prob_w0.05` | `False` | `True` | `True` | 0.535806 | -0.019825 | 0.607267 | +0.003438 | 0.036684 | +0.012632 |
| 9 | `id178_s4_id239_prob_w0.075` | `False` | `True` | `True` | 0.534447 | -0.029337 | 0.609085 | +0.005256 | 0.055025 | +0.018948 |
| 10 | `id178_s4_id239_prob_w0.1` | `False` | `True` | `True` | 0.533123 | -0.038605 | 0.610907 | +0.007077 | 0.073367 | +0.025264 |

## Best variant split rows

Best variant: `id178_s4_id239_prob_w0.0025`

| split | primary | loss | ID178 | Δ | S4 loss | ID178 S4 | S4 Δ | pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `subject_hole_5` | `True` | 0.538492 | 0.538638 | -0.000145 | 0.601681 | 0.602700 | -0.001018 | `True` |
| `subject_timeseries_5` | `False` | 0.539598 | 0.539747 | -0.000149 | 0.605230 | 0.606273 | -0.001043 | `True` |
| `actual_test_band_holdout` | `True` | 0.541431 | 0.541604 | -0.000173 | 0.676128 | 0.677336 | -0.001208 | `True` |
| `test_boundary_neighbor2_holdout` | `True` | 0.586120 | 0.586290 | -0.000170 | 0.653477 | 0.654668 | -0.001191 | `True` |
| `test_boundary_neighbor5_holdout` | `True` | 0.557834 | 0.557986 | -0.000152 | 0.618328 | 0.619394 | -0.001066 | `True` |
| `loso_10` | `True` | 0.538492 | 0.538638 | -0.000145 | 0.601681 | 0.602700 | -0.001018 | `True` |

## Decision

- Has artifact-review variant: `False`
- Best artifact-review pass: `False`
- Best Public 0.58 plausible: `False`
- Create artifact now: **no**
- Submit now: **no**

This scan is still full-OOF-source assisted. A variant can only move to artifact review if it beats ID341's conservative proxy while passing frozen split and geometry gates.

