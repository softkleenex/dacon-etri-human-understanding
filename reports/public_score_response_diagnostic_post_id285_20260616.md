# Public score response diagnostic post-ID285

Usable public-scored records with local files: `21`. This is too small and heterogeneous to reconstruct labels; it is only a risk diagnostic.

## Best usable public records

| id | local CV | Public LB | gap | file |
|---:|---:|---:|---:|---|
| 178 | 0.538638 | 0.601411 | 0.062773 | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` |
| 6 | 0.592500 | 0.611675 | 0.019175 | `submission_id_006_score0.5925.csv` |
| 8 | 0.591400 | 0.615136 | 0.023736 | `submission_id_008_score0.5914.csv` |
| 93 | 0.591466 | 0.616656 | 0.025190 | `submission_id_085_score0.591466.csv` |
| 7 | 0.592100 | 0.616659 | 0.024559 | `submission_id_007_score0.5921.csv` |
| 16 | 0.587500 | 0.618483 | 0.030983 | `submission_id_016_score0.5875.csv` |
| 11 | 0.590900 | 0.620848 | 0.029948 | `submission_id_011_score0.5909.csv` |
| 17 | 0.611700 | 0.623883 | 0.012183 | `submission_id_017_score0.6117.csv` |
| 76 | 0.599611 | 0.624152 | 0.024541 | `submission_id_076_score0.5996.csv` |
| 15 | 0.615400 | 0.626045 | 0.010645 | `submission_id_015_score0.6154.csv` |

## Strongest observed scalar-shape correlations

| feature | Spearman vs Public LB | Pearson vs Public LB | Spearman vs gap |
|---|---:|---:|---:|
| S3_std | -0.550 | -0.571 | -0.173 |
| max_abs_delta_vs_id178 | 0.550 | 0.488 | 0.173 |
| Q3_mean | -0.498 | -0.547 | -0.230 |
| mean_abs_delta_vs_id178 | 0.496 | 0.526 | 0.232 |
| extreme_frac_010 | -0.478 | -0.287 | -0.221 |
| S4_std | -0.315 | -0.082 | 0.128 |
| S2_std | -0.301 | -0.369 | -0.164 |
| overall_entropy | 0.245 | 0.166 | -0.020 |
| overall_mean | -0.214 | -0.553 | -0.476 |
| Q2_mean | -0.203 | -0.215 | -0.402 |
| Q1_mean | -0.195 | -0.277 | -0.441 |
| extreme_frac_005 | -0.166 | -0.042 | 0.236 |

Interpretation: treat this as a veto layer. With only a small number of true public scores, the robust path is to prefer candidates that lower local while keeping prediction movement and probability sharpness under control.

No DACON submission was made.
