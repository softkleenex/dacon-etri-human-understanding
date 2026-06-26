# Candidate tier manifest post-ID322 constrained combo

A Pareto/dynamic-program constrained-combo scan directly optimized target combinations under fixed aggregate projection budgets. This improves the practical local frontier from ID314 to ID318 and creates bolder budget-0.05 cards ID319/ID320.

| rank | role | id | local CV | ID178-gap proxy | max Δ ID178 | mean Δ ID178 | failed projection | risk | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | constrained_combo_practical_budget050 | 318 | 0.516844 | 0.579617 | 0.079086 | 0.018678 | +0.0497 | medium | `submission_id_318_post317_combo_cap080_budget050_practical_cv0.516844_unsubmitted.csv` |
| 2 | constrained_combo_balanced_budget050 | 319 | 0.514544 | 0.577317 | 0.098294 | 0.027384 | +0.0496 | medium | `submission_id_319_post317_combo_cap100_budget050_balanced_cv0.514544_unsubmitted.csv` |
| 3 | constrained_combo_scoreguard_budget050 | 320 | 0.511972 | 0.574745 | 0.113916 | 0.028794 | +0.0499 | high | `submission_id_320_post317_combo_cap120_budget050_scoreguard_cv0.511972_unsubmitted.csv` |
| 4 | secondpass_practical_previous | 314 | 0.520647 | 0.583420 | 0.078300 | 0.018700 | +0.0266 | medium | `submission_id_314_post313_secondpass_cap080_lam120_practical_cv0.520647_unsubmitted.csv` |
| 5 | firstpass_scoreguard | 308 | 0.521679 | 0.584452 | 0.059530 | 0.016407 | +0.0340 | medium | `submission_id_308_post307_aggbudget_cap060_lam120_scoreguard_cv0.521679_unsubmitted.csv` |
| 6 | firstpass_balanced | 309 | 0.523249 | 0.586022 | 0.049147 | 0.013501 | +0.0328 | medium | `submission_id_309_post307_aggbudget_cap050_lam120_balanced_cv0.523249_unsubmitted.csv` |
| 7 | constrained_combo_zero_projection_wide | 321 | 0.515882 | 0.578655 | 0.195733 | 0.042075 | -0.0002 | frontier_stress | `submission_id_321_post317_combo_cap200_budget000_zerowide_cv0.515882_unsubmitted.csv` |
| 8 | secondpass_high_movement_scoreguard | 315 | 0.509564 | 0.572337 | 0.175338 | 0.035835 | +0.0416 | frontier_stress | `submission_id_315_post313_secondpass_cap180_lam120_scoreguard_cv0.509564_unsubmitted.csv` |
| 9 | constrained_combo_frontier_budget160 | 322 | 0.489676 | 0.552449 | 0.197509 | 0.041722 | +0.1597 | frontier_stress | `submission_id_322_post317_combo_cap200_budget160_frontier_cv0.489676_unsubmitted.csv` |
| 10 | secondpass_frontier_previous | 317 | 0.481968 | 0.544741 | 0.196587 | 0.048606 | +0.2495 | frontier_stress | `submission_id_317_post313_secondpass_cap200_lam000_frontierstress_cv0.481968_unsubmitted.csv` |
| 11 | confirmed_public_anchor | 178 | 0.538638 | 0.601411 | 0.000000 | 0.000000 | +0.0000 | low | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 12 | failed_public_probe | 287 | 0.378934 | 0.441707 | 0.298460 | 0.115353 | +1.0000 | frontier_stress | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

## Recommendation

- ID318 local 0.516844, projection +0.0497, max delta 0.0791. It is the best practical constrained-combo upgrade over ID314.
- ID319 local 0.514544 and ID320 local 0.511972 keep aggregate projection near +0.05 but have larger max movement.
- ID321 local 0.515882 with projection -0.0002, but max delta 0.1957 makes it wide stress.
- ID322 local 0.489676, projection +0.1597; lower-risk than ID317 but still stress.

No DACON submission was made. If submitting next, ID318 is the current practical candidate; ID319/320 are bolder; ID321/322 are diagnostic/stress.

No DACON submission was made.
