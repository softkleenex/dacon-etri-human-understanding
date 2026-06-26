# Candidate tier manifest post-ID313 aggregate-budget recovery

The aggregate-budget scan pushed local CV substantially lower while keeping an explicit measurement of movement toward the failed ID287 public direction. Lower local is now available, but the under-0.50 cards are high-risk stress artifacts, not first-submit candidates.

| rank | role | id | local CV | ID178-gap proxy | max Δ ID178 | mean Δ ID178 | failed projection | risk | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | aggregate_budget_scoreguard | 308 | 0.521679 | 0.584452 | 0.059530 | 0.016407 | +0.0340 | medium | `submission_id_308_post307_aggbudget_cap060_lam120_scoreguard_cv0.521679_unsubmitted.csv` |
| 2 | aggregate_budget_balanced | 309 | 0.523249 | 0.586022 | 0.049147 | 0.013501 | +0.0328 | medium | `submission_id_309_post307_aggbudget_cap050_lam120_balanced_cv0.523249_unsubmitted.csv` |
| 3 | aggregate_budget_wide_zero_projection | 310 | 0.521885 | 0.584658 | 0.114878 | 0.026107 | +0.0007 | frontier_stress | `submission_id_310_post307_aggbudget_cap120_lam160_zeroproj_cv0.521885_unsubmitted.csv` |
| 4 | aggregate_budget_midwide_zero_projection | 311 | 0.524195 | 0.586968 | 0.091284 | 0.018933 | -0.0000 | high | `submission_id_311_post307_aggbudget_cap100_lam160_zeroproj_cv0.524195_unsubmitted.csv` |
| 5 | orthogonal_aggressive_zero_projection | 306 | 0.526367 | 0.589140 | 0.074735 | 0.017771 | +0.0000 | medium | `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv` |
| 6 | orthogonal_balanced_zero_projection | 305 | 0.529570 | 0.592343 | 0.048772 | 0.012628 | +0.0000 | low | `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv` |
| 7 | orthogonal_safe_zero_projection | 304 | 0.532349 | 0.595122 | 0.029765 | 0.008162 | -0.0000 | low | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 8 | orthogonal_negative_info | 307 | 0.537792 | 0.600565 | 0.026555 | 0.005753 | -0.0170 | low | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 9 | aggregate_budget_scorepressure_under050 | 312 | 0.496824 | 0.559597 | 0.078960 | 0.026360 | +0.2113 | frontier_stress | `submission_id_312_post307_aggbudget_cap080_lam000_scorepressure_cv0.496824_unsubmitted.csv` |
| 10 | aggregate_budget_frontierstress_under049 | 313 | 0.488205 | 0.550978 | 0.113424 | 0.031438 | +0.2412 | frontier_stress | `submission_id_313_post307_aggbudget_cap120_lam000_frontierstress_cv0.488205_unsubmitted.csv` |
| 11 | pre_aggregate_recovery_positive_projection | 292 | 0.518466 | 0.581239 | 0.028837 | 0.012351 | +0.1024 | high | `submission_id_292_post287_base289_antifailed_ultrasafe_cv0.518466_unsubmitted.csv` |
| 12 | temporal_partial_positive_projection | 302 | 0.519801 | 0.582574 | 0.039357 | 0.011166 | +0.0937 | high | `submission_id_302_post300_temporal_cap040_tail55_max4_cv0.519801_unsubmitted.csv` |
| 13 | temporal_score_positive_projection | 303 | 0.505102 | 0.567875 | 0.049218 | 0.020352 | +0.1706 | high | `submission_id_303_post300_temporal_cap050_tail45_score_cv0.505102_unsubmitted.csv` |
| 14 | confirmed_public_anchor | 178 | 0.538638 | 0.601411 | 0.000000 | 0.000000 | +0.0000 | low | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 15 | demoted_failed_public_probe | 287 | 0.378934 | 0.441707 | 0.298460 | 0.115353 | +1.0000 | frontier_stress | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

## Recommendation

- Best controlled score-pressure: **ID308 local 0.521679 with aggregate failed projection +0.0340 and max delta 0.059530.**
- Best balanced controlled card: **ID309 local 0.523249 with aggregate failed projection +0.0328 and max delta 0.049147.**
- Best local-only frontier: ID313 local 0.488205, but it is frontier-stress and follows the ID287 failed direction strongly (+0.2412). Do not submit first.
- Under-0.50 local card: ID312 local 0.496824, but still high positive failed-direction projection (+0.2113). Treat as a local-score stress artifact.

Prefer ID309 for controlled score pressure or ID308 for slightly lower local; avoid ID312/ID313 until public feedback says positive failed-direction projection is tolerated.

Local-score frontier improved below 0.50, but Public 0.55 remains unproven without a new DACON result.

No DACON submission was made.
