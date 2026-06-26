# Post-ID297 wide anti-vector robustness

IDs 295-297 are direct anti-vector probes from confirmed Public anchor ID178 away from public-failed ID287. They intentionally sacrifice local CV to test whether the public distribution lies opposite the failed ID287 movement.

| rank | tier | id | local CV | ID178-gap proxy | max Δ ID178 | mean Δ ID178 | max Δ failed ID287 | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | recovery_lead | 292 | 0.518466 | 0.581239 | 0.028837 | 0.012351 | 0.271978 | `submission_id_292_post287_base289_antifailed_ultrasafe_cv0.518466_unsubmitted.csv` |
| 2 | clean_recovery_reference | 289 | 0.517635 | 0.580408 | 0.029698 | 0.012847 | 0.270624 | `submission_id_289_post287_clean_anchor_cap030_cv0.517635_unsubmitted.csv` |
| 3 | balanced_recovery | 293 | 0.510898 | 0.573671 | 0.039357 | 0.016941 | 0.261519 | `submission_id_293_post287_base290_antifailed_balanced_cv0.510898_unsubmitted.csv` |
| 4 | score_recovery | 294 | 0.507898 | 0.570671 | 0.044288 | 0.018766 | 0.256489 | `submission_id_294_post287_base290_antifailed_score_cv0.507898_unsubmitted.csv` |
| 5 | direct_antivector_mild | 295 | 0.548583 | 0.611356 | 0.014923 | 0.005768 | 0.313383 | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 6 | direct_antivector_logit_mild | 297 | 0.550861 | 0.613634 | 0.018164 | 0.007442 | 0.312665 | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |
| 7 | direct_antivector_stronger | 296 | 0.558807 | 0.621580 | 0.029846 | 0.011535 | 0.328306 | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 8 | anchor | 178 | 0.538638 | 0.601411 | 0.000000 | 0.000000 | 0.298460 | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 9 | demoted_failed | 287 | 0.378934 | 0.441707 | 0.298460 | 0.115353 | 0.000000 | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

Recommendation: ID292 remains the next score-recovery candidate. ID295 is the cleanest direct anti-vector information probe; ID296 is stronger but worse local and should only be used if we explicitly want to test the opposite-of-ID287 hypothesis. No DACON submission was made.
