# Sequence source-generation execution review — post ID335

Created: `2026-06-17 23:34:42 UTC+09:00`

This is report-only. It summarizes diagnostics already executed without `--promote`; no submission artifact or DACON submission was created.

## Lane summary

| lane | base id | base mean | best mean | improvement | changed targets |
|---|---:|---:|---:|---:|---|
| `Q2_sequence_summary_embedding` | 191 | 0.538009 | 0.536429 | +0.001581 | `['Q1', 'Q2', 'Q3']` |
| `S3_smartphone_light_compact` | 193 | 0.536429 | 0.536380 | +0.000049 | `['S3']` |
| `Q3_S2_S4_wavelet_spectral` | 203 | 0.535980 | 0.535764 | +0.000216 | `['Q3', 'S2', 'S4']` |

## Target-level details

### Q2_sequence_summary_embedding

| target | best | base loss | best loss | improvement |
|---|---|---:|---:|---:|
| Q1 | `blend_lr_c0_3_0.1` | 0.551890 | 0.549258 | +0.002632 |
| Q2 | `blend_lr_c0_3_0.15` | 0.556679 | 0.550689 | +0.005991 |
| Q3 | `blend_lr_c0_3_0.07` | 0.566647 | 0.564204 | +0.002443 |
| S1 | `base` | 0.467751 | 0.467751 | +0.000000 |
| S2 | `base` | 0.529522 | 0.529522 | +0.000000 |
| S3 | `base` | 0.490893 | 0.490893 | +0.000000 |
| S4 | `base` | 0.602685 | 0.602685 | +0.000000 |

### S3_smartphone_light_compact

| target | best | base loss | best loss | improvement |
|---|---|---:|---:|---:|
| Q1 | `base` | 0.549258 | 0.549258 | +0.000000 |
| Q2 | `base` | 0.550689 | 0.550689 | +0.000000 |
| Q3 | `base` | 0.564204 | 0.564204 | +0.000000 |
| S1 | `base` | 0.467751 | 0.467751 | +0.000000 |
| S2 | `base` | 0.529522 | 0.529522 | +0.000000 |
| S3 | `blend__phone_light__hgb_l7__w0.05` | 0.490893 | 0.490553 | +0.000340 |
| S4 | `base` | 0.602685 | 0.602685 | +0.000000 |

### Q3_S2_S4_wavelet_spectral

| target | best | base loss | best loss | improvement |
|---|---|---:|---:|---:|
| Q1 | `base` | 0.548258 | 0.548258 | +0.000000 |
| Q2 | `base` | 0.550689 | 0.550689 | +0.000000 |
| Q3 | `blend__wavelet_spectral__lr_c0_1__w0.05` | 0.563914 | 0.562877 | +0.001037 |
| S1 | `base` | 0.467737 | 0.467737 | +0.000000 |
| S2 | `blend__wavelet_spectral__lr_c0_3__w0.005` | 0.529522 | 0.529516 | +0.000005 |
| S3 | `base` | 0.490553 | 0.490553 | +0.000000 |
| S4 | `blend__wavelet_spectral__lr_c0_3__w0.05` | 0.601188 | 0.600718 | +0.000471 |

## Decision

Executed report-only source diagnostics reproduce the existing sequence-source gains, but the gains are small and remain selected under public-structure diagnostics rather than true model-level nested validation. Use these results to guide future Kaggle source generation, not artifact promotion.
