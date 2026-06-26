# Public joint-distribution risk — post ID341

Created: `2026-06-23 16:41:33 UTC+09:00`

This is report-only. It creates no submission CSV, base prediction artifact, notebook/history artifact, submission-log row, or DACON submission.

## Input coverage

- Plausible public rows loaded: `25` / `25`
- Skipped rows: `0`
- Train temporal volatility: `0.369248`

## Metric correlations

| metric | n | Pearson public | Spearman public | Pearson gap | Spearman gap |
|---|---:|---:|---:|---:|---:|
| `extreme_rate_05_95` | 25 | -0.010552 | -0.341001 | 0.729897 | 0.515683 |
| `extreme_rate_01_99` | 25 | 0.116001 | 0.385695 | 0.713505 | 0.431455 |
| `local_cv` | 25 | 0.212010 | 0.671538 | -0.671846 | -0.676923 |
| `prediction_entropy` | 25 | 0.094836 | 0.379157 | -0.674019 | -0.363522 |
| `mean_abs_delta_vs_id178` | 24 | 0.554954 | 0.654525 | 0.166770 | 0.131505 |
| `target_mean_l1_to_train` | 25 | 0.528256 | 0.602395 | 0.630796 | 0.028230 |
| `prediction_sharpness` | 25 | -0.156141 | -0.388712 | 0.619617 | 0.353967 |
| `pair11_l1_to_train` | 25 | 0.566774 | 0.598920 | 0.402913 | 0.021281 |
| `target_mean_max_abs_to_train` | 25 | 0.484556 | 0.123901 | 0.583561 | 0.385616 |
| `max_abs_delta_vs_id178` | 24 | 0.488648 | 0.567521 | -0.069900 | -0.146505 |
| `temporal_volatility_ratio_to_train` | 25 | -0.242094 | -0.530298 | 0.541002 | 0.186321 |
| `corr_l2_to_train` | 25 | 0.407361 | 0.529430 | -0.296629 | -0.185452 |
| `corr_l1_to_train` | 25 | 0.379629 | 0.517269 | -0.331805 | -0.197613 |

## Recent public anchors

| id | local | public | mean L1 | corr L2 | pair11 L1 | vol ratio | entropy | sharp | Δ178 mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 178 | 0.538638 | 0.601411 | 0.014594 | 0.189826 | 0.027899 | 0.260328 | 0.562111 | 0.205802 |  |
| 287 | 0.378934 | 0.669879 | 0.030329 | 0.153261 | 0.031407 | 0.488425 | 0.446538 | 0.285777 | 0.115353 |
| 329 | 0.516718 | 0.601791 | 0.014643 | 0.192020 | 0.028556 | 0.253852 | 0.572204 | 0.195938 | 0.019372 |

## Best public rows by actual score

| id | public | local | mean L1 | corr L2 | pair11 L1 | vol ratio | sharp | file |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 337 | 0.600621 | 0.538066 | 0.014337 | 0.189507 | 0.028107 | 0.261011 | 0.205506 | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 340 | 0.601379 | 0.538426 | 0.014623 | 0.189883 | 0.027931 | 0.260420 | 0.205675 | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 178 | 0.601411 | 0.538638 | 0.014594 | 0.189826 | 0.027899 | 0.260328 | 0.205802 | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` |
| 329 | 0.601791 | 0.516718 | 0.014643 | 0.192020 | 0.028556 | 0.253852 | 0.195938 | `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv` |
| 6 | 0.611675 | 0.592500 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_006_score0.5925.csv` |
| 8 | 0.615136 | 0.591400 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_008_score0.5914.csv` |
| 93 | 0.616656 | 0.591466 | 0.021235 | 0.272868 | 0.030479 | 0.144315 | 0.168295 | `submission_id_085_score0.591466.csv` |
| 7 | 0.616659 | 0.592100 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_007_score0.5921.csv` |
| 16 | 0.618483 | 0.587500 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_016_score0.5875.csv` |
| 11 | 0.620848 | 0.590900 | 0.014851 | 0.346904 | 0.029026 | 0.049208 | 0.146282 | `submission_id_011_score0.5909.csv` |

## Decision

- Strong warning metrics: `extreme_rate_05_95`, `extreme_rate_01_99`, `local_cv`, `prediction_entropy`, `mean_abs_delta_vs_id178`, `target_mean_l1_to_train`, `prediction_sharpness`, `pair11_l1_to_train`, `target_mean_max_abs_to_train`, `max_abs_delta_vs_id178`, `temporal_volatility_ratio_to_train`, `corr_l2_to_train`, `corr_l1_to_train`
- Joint-distribution gate useful: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Use strong joint-distribution risk metrics as report-level candidate-selection warnings only; do not create a submission artifact from this diagnostic. If no robust signal appears, keep the Kaggle upstream rebuild as the main next action.

