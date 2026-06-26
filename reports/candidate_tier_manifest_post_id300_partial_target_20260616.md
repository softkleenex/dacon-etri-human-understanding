# Candidate tier manifest post-ID300 partial-target recovery

IDs 298-300 split the recovery movement by target group after ID287 failed. This helps test whether public failure is concentrated in Q targets, S targets, or S3.

| rank | role | id | local CV | ID178-gap proxy | max Δ ID178 | mean Δ ID178 | selected targets | file |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 1 | full_recovery_lead | 292 | 0.518466 | 0.581239 | 0.028837 | 0.012351 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_292_post287_base289_antifailed_ultrasafe_cv0.518466_unsubmitted.csv` |
| 2 | q_only_probe | 298 | 0.527647 | 0.590420 | 0.029697 | 0.006237 | `Q1,Q2,Q3` | `submission_id_298_post297_src289_q_all_partial_cv0.527647_unsubmitted.csv` |
| 3 | s_only_probe | 299 | 0.528626 | 0.591399 | 0.029698 | 0.006611 | `S1,S2,S3,S4` | `submission_id_299_post297_src289_s_all_partial_cv0.528626_unsubmitted.csv` |
| 4 | no_s3_balanced_probe | 300 | 0.511734 | 0.574507 | 0.039357 | 0.015862 | `Q1,Q2,Q3,S1,S2,S4` | `submission_id_300_post297_src293_no_s3_partial_cv0.511734_unsubmitted.csv` |
| 5 | full_balanced_recovery | 293 | 0.510898 | 0.573671 | 0.039357 | 0.016941 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_293_post287_base290_antifailed_balanced_cv0.510898_unsubmitted.csv` |
| 6 | full_score_recovery | 294 | 0.507898 | 0.570671 | 0.044288 | 0.018766 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_294_post287_base290_antifailed_score_cv0.507898_unsubmitted.csv` |
| 7 | direct_antivector_info | 295 | 0.548583 | 0.611356 | 0.014923 | 0.005768 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 8 | strong_antivector_info | 296 | 0.558807 | 0.621580 | 0.029846 | 0.011535 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 9 | confirmed_anchor | 178 | 0.538638 | 0.601411 | 0.000000 | 0.000000 | `-` | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 10 | demoted_failed | 287 | 0.378934 | 0.441707 | 0.298460 | 0.115353 | `-` | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

Recommendation: ID292 is still the score-recovery lead. ID298/ID299 are the clean Q-vs-S diagnostic pair. ID300 is a stronger partial candidate that avoids moving S3. No DACON submission was made.
