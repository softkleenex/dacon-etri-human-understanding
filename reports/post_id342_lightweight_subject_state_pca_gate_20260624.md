# Post-ID342 lightweight subject-state PCA gate

Created: `2026-06-24 05:27:40 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Top Rows

| source | target | config | wins | mean delta | max delta | eligible |
|---|---|---|---:|---:|---:|---|
| `registered_material` | `S3` | `registered_material_S3_pca016_lr_c0_10_w0.005` | 2 | 0.000053036 | 0.000167495 | `False` |
| `registered_material` | `S3` | `registered_material_S3_pca064_lr_c0_10_w0.005` | 2 | 0.000059297 | 0.000245970 | `False` |
| `env_plus_material` | `S3` | `env_plus_material_S3_pca032_lr_c0_10_w0.005` | 1 | 0.000079920 | 0.000345154 | `False` |
| `registered_material` | `S3` | `registered_material_S3_pca016_lr_c0_03_w0.005` | 1 | 0.000082139 | 0.000225539 | `False` |
| `env_social_routine` | `Q3` | `env_social_routine_Q3_pca008_lr_c0_10_w0.005` | 1 | 0.000090192 | 0.000273502 | `False` |
| `env_social_routine` | `Q3` | `env_social_routine_Q3_pca008_lr_c0_03_w0.005` | 1 | 0.000093058 | 0.000283524 | `False` |
| `registered_material` | `Q3` | `registered_material_Q3_pca004_lr_c0_10_w0.005` | 2 | 0.000099881 | 0.000361385 | `False` |
| `registered_material` | `S3` | `registered_material_S3_pca064_lr_c0_03_w0.005` | 2 | 0.000100061 | 0.000300559 | `False` |
| `env_social_routine` | `S3` | `env_social_routine_S3_pca016_lr_c0_10_w0.005` | 1 | 0.000102432 | 0.000284120 | `False` |
| `registered_material` | `Q3` | `registered_material_Q3_pca004_lr_c0_03_w0.005` | 1 | 0.000103917 | 0.000369154 | `False` |
| `env_plus_material` | `Q3` | `env_plus_material_Q3_pca004_lr_c0_10_w0.005` | 1 | 0.000106940 | 0.000396993 | `False` |
| `env_plus_material` | `Q3` | `env_plus_material_Q3_pca004_lr_c0_03_w0.005` | 1 | 0.000110841 | 0.000405027 | `False` |

## Decision

- Eligible rows: `0`
- Selected targets: `[]`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Lightweight PCA over existing label-free matrices does not open Q3/S3. Do not escalate to Kaggle or artifacts; only a materially different subject-state model remains.
