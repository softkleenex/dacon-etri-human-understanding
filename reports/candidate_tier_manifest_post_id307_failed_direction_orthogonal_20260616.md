# Candidate tier manifest post-ID307 failed-direction orthogonal recovery

ID287 public failure made the ID178→ID287 prediction direction a concrete risk signal. This manifest ranks new candidates by local CV, movement from ID178, and projection onto that failed direction.

| rank | role | id | local CV | ID178-gap proxy | max Δ ID178 | mean Δ ID178 | failed projection | selected targets | file |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | orthogonal_safe_zero_projection | 304 | 0.532349 | 0.595122 | 0.029765 | 0.008162 | -0.0000 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 2 | orthogonal_balanced_zero_projection | 305 | 0.529570 | 0.592343 | 0.048772 | 0.012628 | +0.0000 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv` |
| 3 | orthogonal_aggressive_zero_projection | 306 | 0.526367 | 0.589140 | 0.074735 | 0.017771 | +0.0000 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv` |
| 4 | orthogonal_mild_negative_projection_info | 307 | 0.537792 | 0.600565 | 0.026555 | 0.005753 | -0.0170 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 5 | anti_failed_ultrasafe_but_toward_failed | 292 | 0.518466 | 0.581239 | 0.028837 | 0.012351 | +0.1024 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_292_post287_base289_antifailed_ultrasafe_cv0.518466_unsubmitted.csv` |
| 6 | temporal_max4_zeroish_partial | 302 | 0.519801 | 0.582574 | 0.039357 | 0.011166 | +0.0937 | `Q2,Q3,S1,S2` | `submission_id_302_post300_temporal_cap040_tail55_max4_cv0.519801_unsubmitted.csv` |
| 7 | q_only_partial_diagnostic | 298 | 0.527647 | 0.590420 | 0.029697 | 0.006237 | +0.0541 | `Q1,Q2,Q3` | `submission_id_298_post297_src289_q_all_partial_cv0.527647_unsubmitted.csv` |
| 8 | s_only_partial_diagnostic | 299 | 0.528626 | 0.591399 | 0.029698 | 0.006611 | +0.0527 | `S1,S2,S3,S4` | `submission_id_299_post297_src289_s_all_partial_cv0.528626_unsubmitted.csv` |
| 9 | no_s3_partial_candidate | 300 | 0.511734 | 0.574507 | 0.039357 | 0.015862 | +0.1353 | `Q1,Q2,Q3,S1,S2,S4` | `submission_id_300_post297_src293_no_s3_partial_cv0.511734_unsubmitted.csv` |
| 10 | pure_antivector_mild_info | 295 | 0.548583 | 0.611356 | 0.014923 | 0.005768 | -0.0500 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 11 | pure_antivector_strong_info | 296 | 0.558807 | 0.621580 | 0.029846 | 0.011535 | -0.1000 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 12 | pure_antivector_logit_info | 297 | 0.550861 | 0.613634 | 0.018164 | 0.007442 | -0.0614 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |
| 13 | clean_anchor_recovery_duplicate_of_301 | 289 | 0.517635 | 0.580408 | 0.029698 | 0.012847 | +0.1068 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_289_post287_clean_anchor_cap030_cv0.517635_unsubmitted.csv` duplicate=ID301 |
| 14 | temporal_safe_duplicate_of_289 | 301 | 0.517635 | 0.580408 | 0.029698 | 0.012847 | +0.1068 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_301_post300_temporal_cap030_tail45_safe_cv0.517635_unsubmitted.csv` duplicate=ID289 |
| 15 | clean_anchor_balanced_toward_failed | 290 | 0.504921 | 0.567694 | 0.049218 | 0.020598 | +0.1712 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_290_post287_clean_anchor_cap050_cv0.504921_unsubmitted.csv` |
| 16 | temporal_score_toward_failed | 303 | 0.505102 | 0.567875 | 0.049218 | 0.020352 | +0.1706 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_303_post300_temporal_cap050_tail45_score_cv0.505102_unsubmitted.csv` |
| 17 | anti_failed_balanced_toward_failed | 293 | 0.510898 | 0.573671 | 0.039357 | 0.016941 | +0.1381 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_293_post287_base290_antifailed_balanced_cv0.510898_unsubmitted.csv` |
| 18 | anti_failed_score_toward_failed | 294 | 0.507898 | 0.570671 | 0.044288 | 0.018766 | +0.1547 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_294_post287_base290_antifailed_score_cv0.507898_unsubmitted.csv` |
| 19 | clean_anchor_aggressive_toward_failed | 291 | 0.498891 | 0.561664 | 0.059485 | 0.024229 | +0.2041 | `Q1,Q2,Q3,S1,S2,S3,S4` | `submission_id_291_post287_clean_anchor_cap060_cv0.498891_unsubmitted.csv` |
| 20 | confirmed_public_anchor | 178 | 0.538638 | 0.601411 | 0.000000 | 0.000000 | +0.0000 | `-` | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` public=0.601411 |
| 21 | demoted_failed_public_probe | 287 | 0.378934 | 0.441707 | 0.298460 | 0.115353 | +1.0000 | `-` | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` public=0.669879 |

## Recommendation

First public probe, if approved: **ID304**. ID304 improves local over ID178 while enforcing near-zero projection onto the observed ID287 public-failure direction and keeping max test movement under 0.03.

Second probe: ID305 if ID304 is not worse than ID178; ID307 if the goal is information about whether moving opposite ID287 helps public.

Avoid first: ID301 (prediction duplicate of ID289), ID303/292/293/294 (better local but still positive projection toward ID287 failure direction), ID306 until ID304/305 public behavior is known because cap0.08 movement is larger.

Not proven. Even ID306 local 0.526367 needs a much smaller public gap than ID178; submit only as an empirical probe, not as a claimed 0.55 solution.

No DACON submission was made.
