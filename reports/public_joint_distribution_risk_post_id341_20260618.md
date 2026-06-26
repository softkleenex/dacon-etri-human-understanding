# Public joint-distribution risk — post ID341

Created: `2026-06-18 06:29:20 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Input coverage

- Plausible public rows loaded: `23` / `23`
- Skipped rows: `0`
- Train temporal volatility: `0.369248`

## Metric correlations

| metric | n | Pearson public | Spearman public | Pearson gap | Spearman gap |
|---|---:|---:|---:|---:|---:|
| `extreme_rate_05_95` | 23 | 0.042440 | -0.155470 | 0.738198 | 0.431862 |
| `extreme_rate_01_99` | 23 | 0.104061 | 0.392016 | 0.718100 | 0.444554 |
| `prediction_entropy` | 23 | 0.019825 | 0.222572 | -0.697044 | -0.252791 |
| `local_cv` | 23 | 0.156420 | 0.621542 | -0.686977 | -0.671937 |
| `prediction_sharpness` | 23 | -0.069594 | -0.235357 | 0.654920 | 0.242331 |
| `target_mean_l1_to_train` | 23 | 0.514183 | 0.496865 | 0.643716 | 0.158648 |
| `target_mean_max_abs_to_train` | 23 | 0.489845 | 0.284172 | 0.586195 | 0.332987 |
| `temporal_volatility_ratio_to_train` | 23 | -0.160565 | -0.420156 | 0.575491 | 0.055207 |
| `pair11_l1_to_train` | 23 | 0.556958 | 0.471296 | 0.411833 | 0.177244 |
| `mean_abs_delta_vs_id178` | 22 | 0.539810 | 0.534201 | 0.272069 | 0.331997 |
| `max_abs_delta_vs_id178` | 22 | 0.452954 | 0.415338 | -0.056322 | -0.006831 |
| `corr_l2_to_train` | 23 | 0.339806 | 0.420156 | -0.319730 | -0.055207 |
| `corr_l1_to_train` | 23 | 0.306411 | 0.406209 | -0.361184 | -0.066830 |

## Recent public anchors

| id | local | public | mean L1 | corr L2 | pair11 L1 | vol ratio | entropy | sharp | Δ178 mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 178 | 0.538638 | 0.601411 | 0.014594 | 0.189826 | 0.027899 | 0.260328 | 0.562111 | 0.205802 |  |
| 287 | 0.378934 | 0.669879 | 0.030329 | 0.153261 | 0.031407 | 0.488425 | 0.446538 | 0.285777 | 0.115353 |
| 329 | 0.516718 | 0.601791 | 0.014643 | 0.192020 | 0.028556 | 0.253852 | 0.572204 | 0.195938 | 0.019372 |

## Best public rows by actual score

| id | public | local | mean L1 | corr L2 | pair11 L1 | vol ratio | sharp | file |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 178 | 0.601411 | 0.538638 | 0.014594 | 0.189826 | 0.027899 | 0.260328 | 0.205802 | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` |
| 329 | 0.601791 | 0.516718 | 0.014643 | 0.192020 | 0.028556 | 0.253852 | 0.195938 | `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv` |
| 6 | 0.611675 | 0.592500 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_006_score0.5925.csv` |
| 8 | 0.615136 | 0.591400 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_008_score0.5914.csv` |
| 93 | 0.616656 | 0.591466 | 0.021235 | 0.272868 | 0.030479 | 0.144315 | 0.168295 | `submission_id_085_score0.591466.csv` |
| 7 | 0.616659 | 0.592100 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_007_score0.5921.csv` |
| 16 | 0.618483 | 0.587500 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_016_score0.5875.csv` |
| 11 | 0.620848 | 0.590900 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_011_score0.5909.csv` |
| 17 | 0.623883 | 0.611700 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_017_score0.6117.csv` |
| 76 | 0.624152 | 0.599611 | 0.014119 | 0.377833 | 0.034702 | 0.048093 | 0.138604 | `submission_id_076_score0.5996.csv` |

## Decision

- Strong warning metrics: `extreme_rate_05_95`, `extreme_rate_01_99`, `prediction_entropy`, `local_cv`, `prediction_sharpness`, `target_mean_l1_to_train`, `target_mean_max_abs_to_train`, `temporal_volatility_ratio_to_train`, `pair11_l1_to_train`, `mean_abs_delta_vs_id178`, `max_abs_delta_vs_id178`
- Joint-distribution gate useful: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Use strong joint-distribution risk metrics as report-level candidate-selection warnings only; do not create a submission artifact from this diagnostic. If no robust signal appears, keep the Kaggle upstream rebuild as the main next action.

