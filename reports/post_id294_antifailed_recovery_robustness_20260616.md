# Post-ID294 anti-failed recovery robustness

IDs 292-294 preserve the clean ID178-anchor recovery move while applying a tiny negative blend away from public-failed ID287.

| rank | tier | id | local CV | ID178-gap proxy | post-failure proxy | max Δ ID178 | mean Δ ID178 | max Δ failed ID287 | file |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | ultrasafe_antifailed_probe | 292 | 0.518466 | 0.581239 | 0.581239 | 0.028837 | 0.012351 | 0.271978 | `submission_id_292_post287_base289_antifailed_ultrasafe_cv0.518466_unsubmitted.csv` |
| 2 | safest_clean_probe | 289 | 0.517635 | 0.580408 | 0.580408 | 0.029698 | 0.012847 | 0.270624 | `submission_id_289_post287_clean_anchor_cap030_cv0.517635_unsubmitted.csv` |
| 3 | balanced_antifailed_probe | 293 | 0.510898 | 0.573671 | 0.585579 | 0.039357 | 0.016941 | 0.261519 | `submission_id_293_post287_base290_antifailed_balanced_cv0.510898_unsubmitted.csv` |
| 4 | score_antifailed_probe | 294 | 0.507898 | 0.570671 | 0.589135 | 0.044288 | 0.018766 | 0.256489 | `submission_id_294_post287_base290_antifailed_score_cv0.507898_unsubmitted.csv` |
| 5 | balanced_clean_probe | 290 | 0.504921 | 0.567694 | 0.592715 | 0.049218 | 0.020598 | 0.251460 | `submission_id_290_post287_clean_anchor_cap050_cv0.504921_unsubmitted.csv` |
| 6 | wait_score_pressure | 291 | 0.498891 | 0.561664 | 0.600277 | 0.059485 | 0.024229 | 0.242264 | `submission_id_291_post287_clean_anchor_cap060_cv0.498891_unsubmitted.csv` |
| 7 | anchor | 178 | 0.538638 | 0.601411 | 0.601411 | 0.000000 | 0.000000 | 0.298460 | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 8 | demoted_failed | 287 | 0.378934 | 0.441707 | 0.798984 | 0.298460 | 0.115353 | 0.000000 | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

Recommendation: if submitting again, ID292 is the safest anti-failed recovery probe; ID293 is the balanced alternative; ID294 is the more score-oriented alternative. No DACON submission was made.
