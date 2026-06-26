# Candidate tier manifest post-ID328 third-pass constrained combo

Third-pass constrained-combo search included IDs 318-322 as sources. It nudged the practical candidate lower and found a much lower local frontier, while keeping ID287 failed-direction projection explicit.

| rank | role | id | local CV | ID178-gap proxy | max Δ ID178 | mean Δ ID178 | failed projection | risk | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | thirdpass_practical_budget050 | 323 | 0.516767 | 0.579540 | 0.077587 | 0.019746 | +0.0498 | medium | `submission_id_323_post322_combo3_cap080_budget050_practical_cv0.516767_unsubmitted.csv` |
| 2 | thirdpass_balanced_budget050 | 324 | 0.514188 | 0.576961 | 0.099643 | 0.025798 | +0.0498 | medium | `submission_id_324_post322_combo3_cap100_budget050_balanced_cv0.514188_unsubmitted.csv` |
| 3 | thirdpass_scoreguard_budget050 | 325 | 0.510506 | 0.573279 | 0.149716 | 0.034963 | +0.0498 | high | `submission_id_325_post322_combo3_cap150_budget050_scoreguard_cv0.510506_unsubmitted.csv` |
| 4 | previous_practical_budget050 | 318 | 0.516844 | 0.579617 | 0.079086 | 0.018678 | +0.0497 | medium | `submission_id_318_post317_combo_cap080_budget050_practical_cv0.516844_unsubmitted.csv` |
| 5 | previous_balanced_budget050 | 319 | 0.514544 | 0.577317 | 0.098294 | 0.027384 | +0.0496 | medium | `submission_id_319_post317_combo_cap100_budget050_balanced_cv0.514544_unsubmitted.csv` |
| 6 | previous_scoreguard_budget050 | 320 | 0.511972 | 0.574745 | 0.113916 | 0.028794 | +0.0499 | medium | `submission_id_320_post317_combo_cap120_budget050_scoreguard_cv0.511972_unsubmitted.csv` |
| 7 | thirdpass_wideguard_budget050 | 326 | 0.503613 | 0.566386 | 0.227831 | 0.041335 | +0.0497 | frontier_stress | `submission_id_326_post322_combo3_cap240_budget050_wideguard_cv0.503613_unsubmitted.csv` |
| 8 | thirdpass_zerowide_budget000 | 327 | 0.511571 | 0.574344 | 0.270086 | 0.051897 | -0.0003 | frontier_stress | `submission_id_327_post322_combo3_cap280_budget000_zerowide_cv0.511571_unsubmitted.csv` |
| 9 | previous_frontier_budget160 | 322 | 0.489676 | 0.552449 | 0.197509 | 0.041722 | +0.1597 | high | `submission_id_322_post317_combo_cap200_budget160_frontier_cv0.489676_unsubmitted.csv` |
| 10 | previous_frontierstress | 317 | 0.481968 | 0.544741 | 0.196587 | 0.048606 | +0.2495 | frontier_stress | `submission_id_317_post313_secondpass_cap200_lam000_frontierstress_cv0.481968_unsubmitted.csv` |
| 11 | thirdpass_frontier_budget240 | 328 | 0.473842 | 0.536615 | 0.226416 | 0.048974 | +0.2399 | frontier_stress | `submission_id_328_post322_combo3_cap240_budget240_frontier_cv0.473842_unsubmitted.csv` |
| 12 | confirmed_public_anchor | 178 | 0.538638 | 0.601411 | 0.000000 | 0.000000 | +0.0000 | low | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 13 | failed_public_probe | 287 | 0.378934 | 0.441707 | 0.298460 | 0.115353 | +1.0000 | frontier_stress | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

## Recommendation

- ID323 local 0.516767, projection +0.0498, max delta 0.0776. It narrowly supersedes ID318 as practical candidate.
- ID324 local 0.514188, projection +0.0498, max delta 0.0996.
- ID325 local 0.510506, projection +0.0498, max delta 0.1497; strong but high movement.
- ID326 local 0.503613 under budget +0.05, but max delta 0.2278 makes it wide-stress.
- ID328 local 0.473842, projection +0.2399; current local frontier/stress only.

No DACON submission was made. If submitting next, ID323 is the current practical candidate; ID324/325 are bolder; ID326/328 should wait for public feedback.

No DACON submission was made.
