# Kaggle sequence-anchor probe remote summary post-ID335

Input dir: `artifacts/kaggle_outputs/sequence_anchor_probe_post_id335`
Diagnostics found: `4`
Manifest files found: `1`

## Best overall diagnostic

- File: `artifacts/kaggle_outputs/sequence_anchor_probe_post_id335/kaggle_outputs/id179_random_conv_sequence_post_id335_anchor_s1_nk512_seed45502_20260618.json`
- Base ID: `179`
- Kernels/seed: `512` / `45502`
- Base mean: `0.538290`
- Targetwise best mean: `0.537542`
- Improvement vs base: `0.000748`

## Ranked diagnostics

| rank | kernels | seed | targetwise best | improvement | file |
|---:|---:|---:|---:|---:|---|
| 1 | 512 | 45502 | 0.537542 | 0.000748 | `id179_random_conv_sequence_post_id335_anchor_s1_nk512_seed45502_20260618.json` |
| 2 | 512 | 45502 | 0.537628 | 0.001009 | `id178_random_conv_sequence_post_id335_anchor_s1_nk512_seed45502_20260618.json` |
| 3 | 512 | 45502 | 0.537654 | 0.000356 | `id191_random_conv_sequence_post_id335_anchor_s1_nk512_seed45502_20260618.json` |
| 4 | 512 | 45502 | 0.537723 | 0.000233 | `id190_random_conv_sequence_post_id335_anchor_s1_nk512_seed45502_20260618.json` |

## Target-level best improvements

| target | improvement | model | kernels | seed |
|---|---:|---|---:|---:|
| Q1 | 0.000044 | `blend_hgb_l7_0.01` | 512 | 45502 |
| Q2 | 0.000000 | `base` | 512 | 45502 |
| Q3 | 0.000000 | `base` | 512 | 45502 |
| S1 | 0.006820 | `blend_lr_c0_1_0.15` | 512 | 45502 |
| S2 | 0.000000 | `base` | 512 | 45502 |
| S3 | 0.000000 | `base` | 512 | 45502 |
| S4 | 0.000202 | `blend_lr_c0_1_0.03` | 512 | 45502 |

## Guardrail

This is a report-only summary. It creates no DACON submission and does not promote any artifact. Any promotion must be a separate, uniquely named, fully matched CSV/notebook/log/base-prediction decision.
