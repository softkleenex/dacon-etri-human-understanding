# Joint-risk candidate rerank — post ID341

Created: `2026-06-23 16:41:36 UTC+09:00`

This is report-only. It applies joint-distribution warning metrics to existing logged candidate CSVs. It creates no submission artifact, submission-log row, Kaggle run, or DACON submission.

## Input coverage

- Metric rows: `330`
- Unscored rows: `268`
- Recent rows `id >= 330`: `12`
- Skipped rows: `0`

## Tier counts

```json
{
  "low_joint_warning": 86,
  "reject_high_joint_risk": 84,
  "reject_high_anchor_distance": 53,
  "caution_joint_risk": 26,
  "frontier_overfit_warning": 16,
  "watch_one_warning": 3
}
```

## Controls

| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |
|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 178 | 0.538638 | 0.601411 | `already_scored_reference` | 0 |  | 0.037714 | 0.027899 | `` | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` |
| 287 | 0.378934 | 0.669879 | `already_scored_reference` | 8 | 0.115353 | 0.162286 | 0.031407 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` |
| 329 | 0.516718 | 0.601791 | `already_scored_reference` | 0 | 0.019372 | 0.031429 | 0.028556 | `` | `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv` |
| 337 | 0.538066 | 0.600621 | `already_scored_reference` | 0 | 0.001336 | 0.036571 | 0.028107 | `` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 339 | 0.536441 |  | `low_joint_warning` | 0 | 0.012602 | 0.036571 | 0.028070 | `` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 340 | 0.538426 | 0.601379 | `already_scored_reference` | 0 | 0.000494 | 0.037714 | 0.027931 | `` | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 341 | 0.538584 |  | `low_joint_warning` | 0 | 0.000124 | 0.037714 | 0.027907 | `` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |

## Recent candidates `id >= 330`

| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |
|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 330 | 0.511769 |  | `low_joint_warning` | 0 | 0.028048 | 0.025714 | 0.031718 | `` | `submission_id_330_post328_combo4_cap120_budget050_tightscore_cv0.511769_unsubmitted.csv` |
| 331 | 0.509930 |  | `frontier_overfit_warning` | 2 | 0.038757 | 0.025714 | 0.036313 | `target_mean_l1_to_train,pair11_l1_to_train` | `submission_id_331_post328_combo4_cap150_budget050_scoreguard_cv0.509930_unsubmitted.csv` |
| 332 | 0.503202 |  | `watch_one_warning` | 1 | 0.041433 | 0.031429 | 0.034764 | `pair11_l1_to_train` | `submission_id_332_post328_combo4_cap240_budget050_wideguard_cv0.503202_unsubmitted.csv` |
| 333 | 0.466963 |  | `reject_high_joint_risk` | 4 | 0.056313 | 0.036571 | 0.040118 | `target_mean_l1_to_train,pair11_l1_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178` | `submission_id_333_post328_combo4_cap320_budget280_frontier_cv0.466963_unsubmitted.csv` |
| 334 | 0.540347 |  | `low_joint_warning` | 0 | 0.010290 | 0.038857 | 0.027804 | `` | `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv` |
| 335 | 0.539690 |  | `low_joint_warning` | 0 | 0.007392 | 0.038857 | 0.027631 | `` | `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv` |
| 336 | 0.537663 |  | `low_joint_warning` | 0 | 0.004008 | 0.036000 | 0.028521 | `` | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 337 | 0.538066 | 0.600621 | `already_scored_reference` | 0 | 0.001336 | 0.036571 | 0.028107 | `` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 338 | 0.537861 |  | `low_joint_warning` | 0 | 0.004660 | 0.036000 | 0.028122 | `` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 339 | 0.536441 |  | `low_joint_warning` | 0 | 0.012602 | 0.036571 | 0.028070 | `` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 340 | 0.538426 | 0.601379 | `already_scored_reference` | 0 | 0.000494 | 0.037714 | 0.027931 | `` | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 341 | 0.538584 |  | `low_joint_warning` | 0 | 0.000124 | 0.037714 | 0.027907 | `` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |

## Top unscored low-warning candidates

| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |
|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 341 | 0.538584 |  | `low_joint_warning` | 0 | 0.000124 | 0.037714 | 0.027907 | `` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |
| 179 | 0.538290 |  | `low_joint_warning` | 0 | 0.003991 | 0.037714 | 0.027915 | `` | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 336 | 0.537663 |  | `low_joint_warning` | 0 | 0.004008 | 0.036000 | 0.028521 | `` | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 338 | 0.537861 |  | `low_joint_warning` | 0 | 0.004660 | 0.036000 | 0.028122 | `` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 190 | 0.537956 |  | `low_joint_warning` | 0 | 0.005034 | 0.036571 | 0.028241 | `` | `submission_id_190_id173_remote_s1_random_conv_refresh_overlay_cv0.537956_unsubmitted.csv` |
| 307 | 0.537792 |  | `low_joint_warning` | 0 | 0.005753 | 0.034857 | 0.028361 | `` | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 295 | 0.548583 |  | `low_joint_warning` | 0 | 0.005768 | 0.034857 | 0.028081 | `` | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 298 | 0.527647 |  | `low_joint_warning` | 0 | 0.006237 | 0.037714 | 0.028062 | `` | `submission_id_298_post297_src289_q_all_partial_cv0.527647_unsubmitted.csv` |
| 191 | 0.538009 |  | `low_joint_warning` | 0 | 0.006327 | 0.036571 | 0.028104 | `` | `submission_id_191_id190_tempered_id177_remote_s1_anchor_blend_cv0.538009_unsubmitted.csv` |
| 299 | 0.528626 |  | `low_joint_warning` | 0 | 0.006611 | 0.040000 | 0.027345 | `` | `submission_id_299_post297_src289_s_all_partial_cv0.528626_unsubmitted.csv` |
| 335 | 0.539690 |  | `low_joint_warning` | 0 | 0.007392 | 0.038857 | 0.027631 | `` | `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv` |
| 297 | 0.550861 |  | `low_joint_warning` | 0 | 0.007442 | 0.033714 | 0.028319 | `` | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |
| 304 | 0.532349 |  | `low_joint_warning` | 0 | 0.008162 | 0.033143 | 0.028681 | `` | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 334 | 0.540347 |  | `low_joint_warning` | 0 | 0.010290 | 0.038857 | 0.027804 | `` | `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv` |
| 302 | 0.519801 |  | `low_joint_warning` | 0 | 0.011166 | 0.040571 | 0.028174 | `` | `submission_id_302_post300_temporal_cap040_tail55_max4_cv0.519801_unsubmitted.csv` |
| 296 | 0.558807 |  | `low_joint_warning` | 0 | 0.011535 | 0.033714 | 0.028288 | `` | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 292 | 0.518466 |  | `low_joint_warning` | 0 | 0.012351 | 0.040000 | 0.027575 | `` | `submission_id_292_post287_base289_antifailed_ultrasafe_cv0.518466_unsubmitted.csv` |
| 339 | 0.536441 |  | `low_joint_warning` | 0 | 0.012602 | 0.036571 | 0.028070 | `` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 305 | 0.529570 |  | `low_joint_warning` | 0 | 0.012628 | 0.032571 | 0.029209 | `` | `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv` |
| 289 | 0.517635 |  | `low_joint_warning` | 0 | 0.012847 | 0.040000 | 0.027571 | `` | `submission_id_289_post287_clean_anchor_cap030_cv0.517635_unsubmitted.csv` |
| 301 | 0.517635 |  | `low_joint_warning` | 0 | 0.012847 | 0.040000 | 0.027571 | `` | `submission_id_301_post300_temporal_cap030_tail45_safe_cv0.517635_unsubmitted.csv` |
| 180 | 0.529579 |  | `low_joint_warning` | 0 | 0.012873 | 0.034286 | 0.028454 | `` | `submission_id_180_id179_tempered_id172_extreme_anchor_probe_cv0.529579_unsubmitted.csv` |
| 309 | 0.523249 |  | `low_joint_warning` | 0 | 0.013501 | 0.034857 | 0.030515 | `` | `submission_id_309_post307_aggbudget_cap050_lam120_balanced_cv0.523249_unsubmitted.csv` |
| 300 | 0.511734 |  | `low_joint_warning` | 0 | 0.015862 | 0.040571 | 0.027880 | `` | `submission_id_300_post297_src293_no_s3_partial_cv0.511734_unsubmitted.csv` |
| 308 | 0.521679 |  | `low_joint_warning` | 0 | 0.016407 | 0.033714 | 0.029338 | `` | `submission_id_308_post307_aggbudget_cap060_lam120_scoreguard_cv0.521679_unsubmitted.csv` |
| 293 | 0.510898 |  | `low_joint_warning` | 0 | 0.016941 | 0.041143 | 0.027721 | `` | `submission_id_293_post287_base290_antifailed_balanced_cv0.510898_unsubmitted.csv` |
| 181 | 0.526430 |  | `low_joint_warning` | 0 | 0.017042 | 0.034286 | 0.028435 | `` | `submission_id_181_id179_targetwise_delta06_id172_extreme_probe_cv0.526430_unsubmitted.csv` |
| 306 | 0.526367 |  | `low_joint_warning` | 0 | 0.017771 | 0.030286 | 0.030018 | `` | `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv` |
| 193 | 0.536429 |  | `low_joint_warning` | 0 | 0.018262 | 0.036571 | 0.028217 | `` | `submission_id_193_id191_sequence_summary_embedding_overlay_cv0.536429_unsubmitted.csv` |
| 192 | 0.526261 |  | `low_joint_warning` | 0 | 0.018522 | 0.034286 | 0.028574 | `` | `submission_id_192_id191_targetwise_delta06_id172_extreme_probe_cv0.526261_unsubmitted.csv` |

## Local-CV frontier with warnings

| id | local | public | tier | warnings | Δ178 mean | ext05/95 | pair11 | flags | file |
|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 250 | 0.261184 |  | `reject_high_joint_risk` | 10 | 0.244204 | 0.598286 | 0.035429 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_250_id242_iso_non_s3_s4base_ultra_frontier_cv0.261184_unsubmitted.csv` |
| 251 | 0.263237 |  | `reject_high_joint_risk` | 10 | 0.241984 | 0.598286 | 0.035572 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_251_id242_quantile_non_s3_s4q100_ultra_frontier_cv0.263237_unsubmitted.csv` |
| 252 | 0.274420 |  | `reject_high_joint_risk` | 9 | 0.222174 | 0.528571 | 0.032506 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_252_id249_iso_mixed_second_stage_frontier_cv0.274420_unsubmitted.csv` |
| 285 | 0.356330 |  | `reject_high_joint_risk` | 10 | 0.139552 | 0.274286 | 0.033689 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_285_post282_base278_tempguard_calibration_cv0.356330_unsubmitted.csv` |
| 242 | 0.357651 |  | `reject_high_joint_risk` | 9 | 0.171876 | 0.090857 | 0.055264 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_242_id238_q240_s4iso_recal_frontier_cv0.357651_unsubmitted.csv` |
| 243 | 0.360420 |  | `reject_high_joint_risk` | 9 | 0.168209 | 0.069143 | 0.055327 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_243_id238_q240_all_recal_frontier_cv0.360420_unsubmitted.csv` |
| 288 | 0.372773 |  | `reject_high_joint_risk` | 9 | 0.122496 | 0.176000 | 0.030596 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_288_post285_base278_fixedtemp_frontier_cv0.372773_unsubmitted.csv` |
| 249 | 0.374465 |  | `reject_high_joint_risk` | 8 | 0.139964 | 0.046286 | 0.041226 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178` | `submission_id_249_id235_frontier_twise_cap040_logit_bridge_cv0.374465_unsubmitted.csv` |
| 241 | 0.383459 |  | `reject_high_joint_risk` | 10 | 0.161926 | 0.090857 | 0.049022 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_241_id238_q160_s4iso_recal_frontier_cv0.383459_unsubmitted.csv` |
| 284 | 0.385005 |  | `reject_high_joint_risk` | 10 | 0.115276 | 0.190286 | 0.034841 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_284_post282_base280_tempguard_calibration_cv0.385005_unsubmitted.csv` |
| 278 | 0.389120 |  | `reject_high_joint_risk` | 8 | 0.102841 | 0.096571 | 0.029842 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_278_post274_refined_cap300_p005_blend_cv0.389120_unsubmitted.csv` |
| 248 | 0.391735 |  | `reject_high_joint_risk` | 7 | 0.127223 | 0.034286 | 0.036638 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_01_99,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178` | `submission_id_248_id235_frontier_twise_cap030_logit_bridge_cv0.391735_unsubmitted.csv` |
| 282 | 0.393400 |  | `reject_high_joint_risk` | 8 | 0.099366 | 0.092571 | 0.029629 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400_unsubmitted.csv` |
| 286 | 0.395368 |  | `reject_high_joint_risk` | 8 | 0.102636 | 0.130857 | 0.030271 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_286_post285_base280_fixedtemp_guard_cv0.395368_unsubmitted.csv` |
| 277 | 0.399355 |  | `reject_high_joint_risk` | 8 | 0.094559 | 0.101143 | 0.029634 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_277_post274_refined_cap280_p010_blend_cv0.399355_unsubmitted.csv` |
| 281 | 0.399583 |  | `reject_high_joint_risk` | 8 | 0.094659 | 0.088000 | 0.029379 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_281_post278_gapaware_cap260_sp005_dp100_blend_cv0.399583_unsubmitted.csv` |
| 280 | 0.406704 |  | `reject_high_joint_risk` | 7 | 0.089384 | 0.084571 | 0.029100 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_280_post278_gapaware_cap240_sp005_dp100_blend_cv0.406704_unsubmitted.csv` |
| 276 | 0.411217 |  | `reject_high_joint_risk` | 7 | 0.085222 | 0.087429 | 0.029149 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_276_post274_refined_cap260_p010_blend_cv0.411217_unsubmitted.csv` |
| 279 | 0.414169 |  | `reject_high_joint_risk` | 7 | 0.083903 | 0.081143 | 0.028804 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_279_post278_gapaware_cap220_sp005_dp100_blend_cv0.414169_unsubmitted.csv` |
| 245 | 0.415035 |  | `reject_high_joint_risk` | 7 | 0.115981 | 0.034286 | 0.036035 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_01_99,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178` | `submission_id_245_id235_id242_alpha0375_bridge_frontier_cv0.415035_unsubmitted.csv` |
| 247 | 0.415523 |  | `reject_high_joint_risk` | 8 | 0.117523 | 0.106857 | 0.033047 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178` | `submission_id_247_id235_frontier_twise_cap020_logit_bridge_cv0.415523_unsubmitted.csv` |
| 275 | 0.416609 |  | `reject_high_joint_risk` | 7 | 0.081705 | 0.079429 | 0.028921 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_275_post274_refined_cap220_p005_blend_cv0.416609_unsubmitted.csv` |
| 240 | 0.417349 |  | `reject_high_joint_risk` | 10 | 0.142642 | 0.090857 | 0.037371 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_240_id238_q80_s4iso_recal_frontier_cv0.417349_unsubmitted.csv` |
| 274 | 0.423144 |  | `reject_high_joint_risk` | 7 | 0.077239 | 0.074857 | 0.028412 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_274_post271_ultra_cap200_p005_blend_cv0.423144_unsubmitted.csv` |
| 257 | 0.433079 |  | `reject_high_joint_risk` | 7 | 0.071503 | 0.074857 | 0.028048 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_257_id178_anchor_fine_cap018_frontier_bridge_cv0.433079_unsubmitted.csv` |
| 273 | 0.433773 |  | `reject_high_joint_risk` | 7 | 0.069633 | 0.070286 | 0.028492 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_273_post271_ultra_cap180_p005_blend_cv0.433773_unsubmitted.csv` |
| 271 | 0.435576 |  | `reject_high_joint_risk` | 7 | 0.069400 | 0.074286 | 0.027943 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,mean_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_271_post266_riskpen_cap018_p010_blend_cv0.435576_unsubmitted.csv` |
| 244 | 0.437262 |  | `reject_high_joint_risk` | 6 | 0.110052 | 0.034286 | 0.032972 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_01_99,mean_abs_delta_vs_id178,max_abs_delta_vs_id178` | `submission_id_244_id235_id242_alpha0225_bridge_frontier_cv0.437262_unsubmitted.csv` |
| 239 | 0.437967 |  | `reject_high_joint_risk` | 9 | 0.121326 | 0.198286 | 0.031287 | `target_mean_l1_to_train,target_mean_max_abs_to_train,extreme_rate_05_95,extreme_rate_01_99,prediction_sharpness,temporal_volatility_ratio_to_train,mean_abs_delta_vs_id178,max_abs_delta_vs_id178,prediction_entropy_low` | `submission_id_239_id238_iso_recal_frontier_cv0.437967_unsubmitted.csv` |
| 246 | 0.438034 |  | `reject_high_joint_risk` | 6 | 0.109251 | 0.034286 | 0.033045 | `target_mean_l1_to_train,target_mean_max_abs_to_train,pair11_l1_to_train,extreme_rate_01_99,mean_abs_delta_vs_id178,max_abs_delta_vs_id178` | `submission_id_246_id235_id243_alpha0225_bridge_frontier_cv0.438034_unsubmitted.csv` |

## Decision

- Best low-warning ID: `341`
- Best low-warning file: `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv`
- Frontier candidates have joint warnings: `True`
- Create submission artifact: `False`
- Submit now: `False`
- Next action: Use this rerank as a warning overlay before any manual submission decision. It does not override the post-ID341 transfer gate; candidates still need explicit approval and artifact-contract review.

