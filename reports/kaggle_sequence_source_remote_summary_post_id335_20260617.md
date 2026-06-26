# Kaggle sequence-source remote summary post-ID335

Input dir: `artifacts/kaggle_outputs/sequence_source_post_id335`
Diagnostics found: `6`
Manifest files found: `1`

## Best overall diagnostic

- File: `artifacts/kaggle_outputs/sequence_source_post_id335/kaggle_outputs/id173_random_conv_sequence_post_id335_s1_nk512_seed45502_20260617.json`
- Base ID: `173`
- Kernels/seed: `512` / `45502`
- Base mean: `0.545253`
- Targetwise best mean: `0.542888`
- Improvement vs base: `0.002365`

## Ranked diagnostics

| rank | kernels | seed | targetwise best | improvement | file |
|---:|---:|---:|---:|---:|---|
| 1 | 512 | 45502 | 0.542888 | 0.002365 | `id173_random_conv_sequence_post_id335_s1_nk512_seed45502_20260617.json` |
| 2 | 1024 | 45701 | 0.543946 | 0.001307 | `id173_random_conv_sequence_post_id335_s1_nk1024_seed45701_20260617.json` |
| 3 | 768 | 45601 | 0.544097 | 0.001156 | `id173_random_conv_sequence_post_id335_s1_nk768_seed45601_20260617.json` |
| 4 | 768 | 45602 | 0.544170 | 0.001083 | `id173_random_conv_sequence_post_id335_s1_nk768_seed45602_20260617.json` |
| 5 | 512 | 45501 | 0.544183 | 0.001070 | `id173_random_conv_sequence_post_id335_s1_nk512_seed45501_20260617.json` |
| 6 | 1024 | 45702 | 0.544227 | 0.001027 | `id173_random_conv_sequence_post_id335_s1_nk1024_seed45702_20260617.json` |

## Target-level best improvements

| target | improvement | model | kernels | seed |
|---|---:|---|---:|---:|
| Q1 | 0.005038 | `blend_hgb_l7_0.15` | 512 | 45501 |
| Q2 | 0.001387 | `blend_lr_c0_03_0.07` | 1024 | 45702 |
| Q3 | 0.000509 | `blend_lr_c0_1_0.05` | 1024 | 45701 |
| S1 | 0.010855 | `blend_lr_c0_1_0.2` | 512 | 45502 |
| S2 | 0.000019 | `blend_lr_c0_1_0.01` | 768 | 45602 |
| S3 | 0.000000 | `base` | 512 | 45502 |
| S4 | 0.002520 | `blend_lr_c0_03_0.15` | 512 | 45502 |

## Guardrail

This is a report-only summary. It creates no DACON submission and does not promote any artifact. Any promotion must be a separate, uniquely named, fully matched CSV/notebook/log/base-prediction decision.
