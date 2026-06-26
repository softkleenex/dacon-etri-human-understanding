# Post-ID338 public-feedback next action

Created: `2026-06-24 00:21:14 UTC+09:00`

This is report-only. It creates no submission artifact, base prediction, notebook, submission-log row, Kaggle run, or DACON submission.

Correction: authenticated DACON submission history shows ID342 is the current best public anchor. This report keeps ID338-distance tables only as neighborhood diagnostics.

## Public feedback

- Best confirmed ID: `342`
- Best confirmed Public: `0.5960832519`
- Delta vs ID337: `-0.0045373767`
- Delta vs ID178: `-0.0053280490`
- Interpretation: Authenticated submission history shows ID342 feature-similarity shrink is the current public anchor. ID338 remains useful only as a neighborhood screening anchor; ID339 and ID338 are scored comparison probes.

## Recent candidates

| id | local | public | tier | family | proxy | mean delta338 | max delta338 | failed proj | targets | file |
|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 334 | 0.540347 |  | `reviewable_id338_neighbor` | `sequence_temporal_anchor` | 0.607420 | 0.012426 | 0.125184 | 0.0039 | `S4,S1,Q3` | `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv` |
| 335 | 0.539690 |  | `reviewable_id338_neighbor` | `sequence_temporal_anchor` | 0.606009 | 0.009531 | 0.112063 | 0.0032 | `S1,S4,Q3` | `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv` |
| 336 | 0.537663 |  | `low_id338_distance` | `sequence_temporal_anchor` | 0.602685 | 0.004949 | 0.062902 | -0.0017 | `S1,Q3,S2` | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 337 | 0.538066 | 0.6006206286 | `known_public_reference` | `sequence_temporal_anchor` | 0.601940 | 0.003900 | 0.049233 | -0.0006 | `S1,Q3,S2` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 338 | 0.537861 | 0.5992145177 | `known_public_reference` | `sequence_temporal_anchor` | 0.599280 | 0.000000 | 0.000000 | 0.0005 | `` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 339 | 0.536441 | 0.5985187787 | `known_public_reference` | `sequence_temporal_anchor` | 0.606495 | 0.013654 | 0.097051 | -0.0120 | `Q2,Q3,Q1` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 340 | 0.538426 | 0.6013791753 | `known_public_reference` | `sequence_temporal_anchor` | 0.603083 | 0.005154 | 0.073989 | -0.0003 | `S1,Q3,Q2` | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 341 | 0.538584 |  | `low_id338_distance` | `sequence_temporal_anchor` | 0.602965 | 0.004783 | 0.073989 | -0.0001 | `S1,Q3,S2` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |
| 342 | 0.534000 | 0.5960832519 | `known_public_reference` | `independent_feature_similarity` | 0.607530 | 0.017470 | 0.090826 | -0.0051 | `S4,Q1,Q2` | `submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv` |
| 343 | 0.537854 |  | `low_id338_distance` | `q3_s3_related_probe` | 0.599434 | 0.000113 | 0.003652 | 0.0006 | `S3` | `submission_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.csv` |
| 344 | 0.537839 |  | `low_id338_distance` | `q3_s3_related_probe` | 0.599896 | 0.000453 | 0.014610 | 0.0008 | `S3` | `submission_id_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.csv` |

## Same-batch scored rows

| id | local | public | tier | family | proxy | mean delta338 | max delta338 | failed proj | targets | file |
|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 339 | 0.536441 | 0.5985187787 | `known_public_reference` | `sequence_temporal_anchor` | 0.606495 | 0.013654 | 0.097051 | -0.0120 | `Q2,Q3,Q1` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 342 | 0.534000 | 0.5960832519 | `known_public_reference` | `independent_feature_similarity` | 0.607530 | 0.017470 | 0.090826 | -0.0051 | `S4,Q1,Q2` | `submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv` |

## Top ID338 neighbors

| id | local | public | tier | family | proxy | mean delta338 | max delta338 | failed proj | targets | file |
|---:|---:|---:|---|---|---:|---:|---:|---:|---|---|
| 343 | 0.537854 |  | `low_id338_distance` | `q3_s3_related_probe` | 0.599434 | 0.000113 | 0.003652 | 0.0006 | `S3` | `submission_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.csv` |
| 344 | 0.537839 |  | `low_id338_distance` | `q3_s3_related_probe` | 0.599896 | 0.000453 | 0.014610 | 0.0008 | `S3` | `submission_id_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.csv` |
| 179 | 0.538290 |  | `low_id338_distance` | `sequence_temporal_anchor` | 0.600841 | 0.001316 | 0.035699 | 0.0011 | `S1` | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 191 | 0.538009 |  | `low_id338_distance` | `other` | 0.602654 | 0.002515 | 0.078290 | 0.0024 | `S1` | `submission_id_191_id190_tempered_id177_remote_s1_anchor_blend_cv0.538009_unsubmitted.csv` |
| 336 | 0.537663 |  | `low_id338_distance` | `sequence_temporal_anchor` | 0.602685 | 0.004949 | 0.062902 | -0.0017 | `S1,Q3,S2` | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 341 | 0.538584 |  | `low_id338_distance` | `sequence_temporal_anchor` | 0.602965 | 0.004783 | 0.073989 | -0.0001 | `S1,Q3,S2` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |
| 307 | 0.537792 |  | `low_id338_distance` | `failed_direction_combo` | 0.604014 | 0.007806 | 0.076729 | -0.0170 | `S1,Q3,S4` | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 295 | 0.548583 |  | `reviewable_id338_neighbor` | `failed_direction_combo` | 0.604541 | 0.008316 | 0.088847 | -0.0500 | `S1,Q3,S2` | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 304 | 0.532349 |  | `low_id338_distance` | `failed_direction_combo` | 0.604551 | 0.010225 | 0.068827 | -0.0000 | `S1,Q3,S4` | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 190 | 0.537956 |  | `reviewable_id338_neighbor` | `sequence_temporal_anchor` | 0.604898 | 0.006041 | 0.115855 | 0.0023 | `S1,Q3,S2` | `submission_id_190_id173_remote_s1_random_conv_refresh_overlay_cv0.537956_unsubmitted.csv` |
| 297 | 0.550861 |  | `reviewable_id338_neighbor` | `failed_direction_combo` | 0.604954 | 0.009652 | 0.088377 | -0.0614 | `S1,Q3,S2` | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |
| 335 | 0.539690 |  | `reviewable_id338_neighbor` | `sequence_temporal_anchor` | 0.606009 | 0.009531 | 0.112063 | 0.0032 | `S1,S4,Q3` | `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv` |
| 305 | 0.529570 |  | `low_id338_distance` | `failed_direction_combo` | 0.606326 | 0.014161 | 0.069810 | 0.0000 | `S1,Q3,Q2` | `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv` |
| 177 | 0.540607 |  | `low_id338_distance` | `sequence_temporal_anchor` | 0.606341 | 0.014762 | 0.075416 | 0.0012 | `S2,Q3,S1` | `submission_id_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.csv` |
| 296 | 0.558807 |  | `reviewable_id338_neighbor` | `failed_direction_combo` | 0.606588 | 0.013318 | 0.103705 | -0.1000 | `S1,Q3,Q2` | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 334 | 0.540347 |  | `reviewable_id338_neighbor` | `sequence_temporal_anchor` | 0.607420 | 0.012426 | 0.125184 | 0.0039 | `S4,S1,Q3` | `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv` |
| 193 | 0.536429 |  | `reviewable_id338_neighbor` | `sequence_temporal_anchor` | 0.607845 | 0.015597 | 0.121314 | -0.0105 | `Q2,Q1,Q3` | `submission_id_193_id191_sequence_summary_embedding_overlay_cv0.536429_unsubmitted.csv` |
| 198 | 0.536380 |  | `reviewable_id338_neighbor` | `q3_s3_related_probe` | 0.608137 | 0.016508 | 0.121314 | -0.0102 | `Q2,Q1,Q3` | `submission_id_198_id193_s3_smartphone_light_compact_overlay_cv0.536380_unsubmitted.csv` |
| 306 | 0.526367 |  | `reviewable_id338_neighbor` | `failed_direction_combo` | 0.608910 | 0.018649 | 0.086710 | 0.0000 | `S1,Q3,Q2` | `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv` |
| 200 | 0.536190 |  | `reviewable_id338_neighbor` | `sequence_temporal_anchor` | 0.609268 | 0.018701 | 0.135645 | -0.0129 | `Q2,Q1,Q3` | `submission_id_200_id198_sax_paa_sequence_overlay_cv0.536190_unsubmitted.csv` |

## Decision

- Wait state: no same-batch score wait remains; Kaggle v3 is closed
- 23:30 KST spend rule: The user-approved 23:30 KST spend rule has already been executed for 2026-06-23 via IDs 338, 342, and 339.
- Create submission artifact: `False`
- Submit now: `False`
- Next modeling action: Do not continue the same S1-only or Q-sequence cascade. If another no-submit cycle is available, build an ID342-aware Q3/S3 independent-source probe from materially new raw-source features, while still using ID338 distance and known failed-direction projection as secondary safety screens.

