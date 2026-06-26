# Candidate tier manifest post-ID317 second-pass aggregate budget

Second-pass aggregate-budget search reused ID308-ID313 as sources. It found a stronger practical candidate and a new local frontier, while preserving explicit ID287 failed-direction risk metrics.

| rank | role | id | local CV | ID178-gap proxy | max Δ ID178 | mean Δ ID178 | failed projection | risk | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | secondpass_practical_scoreguard | 314 | 0.520647 | 0.583420 | 0.078300 | 0.018700 | +0.0266 | medium | `submission_id_314_post313_secondpass_cap080_lam120_practical_cv0.520647_unsubmitted.csv` |
| 2 | firstpass_scoreguard | 308 | 0.521679 | 0.584452 | 0.059530 | 0.016407 | +0.0340 | medium | `submission_id_308_post307_aggbudget_cap060_lam120_scoreguard_cv0.521679_unsubmitted.csv` |
| 3 | firstpass_balanced | 309 | 0.523249 | 0.586022 | 0.049147 | 0.013501 | +0.0328 | medium | `submission_id_309_post307_aggbudget_cap050_lam120_balanced_cv0.523249_unsubmitted.csv` |
| 4 | orthogonal_balanced | 305 | 0.529570 | 0.592343 | 0.048772 | 0.012628 | +0.0000 | low | `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv` |
| 5 | orthogonal_aggressive | 306 | 0.526367 | 0.589140 | 0.074735 | 0.017771 | +0.0000 | medium | `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv` |
| 6 | secondpass_high_movement_scoreguard | 315 | 0.509564 | 0.572337 | 0.175338 | 0.035835 | +0.0416 | frontier_stress | `submission_id_315_post313_secondpass_cap180_lam120_scoreguard_cv0.509564_unsubmitted.csv` |
| 7 | secondpass_wide_zero_projection | 316 | 0.517120 | 0.579893 | 0.194423 | 0.036535 | -0.0036 | frontier_stress | `submission_id_316_post313_secondpass_cap200_lam160_zerowide_cv0.517120_unsubmitted.csv` |
| 8 | firstpass_under050_stress | 312 | 0.496824 | 0.559597 | 0.078960 | 0.026360 | +0.2113 | frontier_stress | `submission_id_312_post307_aggbudget_cap080_lam000_scorepressure_cv0.496824_unsubmitted.csv` |
| 9 | firstpass_frontierstress | 313 | 0.488205 | 0.550978 | 0.113424 | 0.031438 | +0.2412 | frontier_stress | `submission_id_313_post307_aggbudget_cap120_lam000_frontierstress_cv0.488205_unsubmitted.csv` |
| 10 | secondpass_frontierstress_local_best | 317 | 0.481968 | 0.544741 | 0.196587 | 0.048606 | +0.2495 | frontier_stress | `submission_id_317_post313_secondpass_cap200_lam000_frontierstress_cv0.481968_unsubmitted.csv` |
| 11 | orthogonal_safe | 304 | 0.532349 | 0.595122 | 0.029765 | 0.008162 | -0.0000 | low | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 12 | confirmed_public_anchor | 178 | 0.538638 | 0.601411 | 0.000000 | 0.000000 | +0.0000 | low | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 13 | failed_public_probe | 287 | 0.378934 | 0.441707 | 0.298460 | 0.115353 | +1.0000 | frontier_stress | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

## Recommendation

- ID314 local 0.520647, projection +0.0266, max delta 0.0783. It is lower than ID308/309 but still not as wild as ID315+.
- ID315 local 0.509564 with projection +0.0416, but max delta 0.1753 makes it a high-movement probe.
- ID316 local 0.517120 with projection -0.0036, but max delta 0.1944 makes it structurally wide.
- ID317 local 0.481968, projection +0.2495, max delta 0.1966. This is only a local frontier/stress artifact.

No DACON submission was made. If submitting, ID314 is the current practical score-pressure candidate; ID315/316/317 should wait for public feedback.

No DACON submission was made.
