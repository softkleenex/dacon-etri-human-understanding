# Post-ID342 public-gap penalty proxy redesign

Created: `2026-06-25 02:13:59 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- OOF files scanned: `247`
- Scored rows: `8`
- Stable gap median: `0.062318739`
- Stable-gap LOO MAE: `0.03194482177540838`
- Penalty-proxy LOO MAE: `0.10625828050414168`
- Stable-gap pairwise agreement: `0.5357142857142857`
- Penalty-proxy pairwise agreement: `1.0`
- Public-0.55 penalty proxy pass count: `0`
- Credible Public-0.55 penalty proxy pass count: `0`
- Validation redesign gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Use the penalty proxy as a stricter review lens, not as a candidate generator. If no credible Public-0.55 row appears, continue with validation redesign or a materially new source; do not submit the local-frontier rows that only pass before shape penalties.

## Penalty Weights

| feature | weight |
|---|---:|
| `mean_abs_delta_vs_id342` | `0.158756481` |
| `max_abs_delta_vs_id342` | `0.000000000` |
| `max_target_mean_abs_delta_vs_id342` | `0.000000000` |
| `max_abs_mean_rate_shift` | `0.000000000` |
| `extreme_probability_rate` | `0.000000000` |

## Top Proxy Rows

| id | proxy | gap to 0.55 | label | file |
|---:|---:|---:|---|---|
| `304` | `0.594667839` | `0.044667839` | `outside_stable_target_move_envelope` | `base_oof_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349.csv` |
| `221` | `0.595555998` | `0.045555998` | `outside_stable_mean_move_envelope` | `base_oof_id_221_id218_public_feature_similarity_overlay_cv0.532990.csv` |
| `342` | `0.596318745` | `0.046318745` | `stable_public_shape_envelope` | `base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv` |
| `212` | `0.596787357` | `0.046787357` | `outside_stable_rate_shift_envelope` | `base_oof_id_212_id206_shapelet_motif_sequence_wide_weight_overlay_cv0.534469.csv` |
| `209` | `0.596848808` | `0.046848808` | `outside_stable_rate_shift_envelope` | `base_oof_id_209_id206_shapelet_motif_sequence_overlay_cv0.534530.csv` |
| `206` | `0.598082651` | `0.048082651` | `outside_stable_rate_shift_envelope` | `base_oof_id_206_id203_wavelet_spectral_sequence_overlay_cv0.535764.csv` |
| `203` | `0.598298872` | `0.048298872` | `outside_stable_rate_shift_envelope` | `base_oof_id_203_id200_s4_recurrence_gaf_sequence_overlay_cv0.535980.csv` |
| `200` | `0.598508597` | `0.048508597` | `outside_stable_rate_shift_envelope` | `base_oof_id_200_id198_sax_paa_sequence_overlay_cv0.536190.csv` |
| `198` | `0.598698766` | `0.048698766` | `stable_public_shape_envelope` | `base_oof_id_198_id193_s3_smartphone_light_compact_overlay_cv0.536380.csv` |
| `193` | `0.598747389` | `0.048747389` | `stable_public_shape_envelope` | `base_oof_id_193_id191_sequence_summary_embedding_overlay_cv0.536429.csv` |
| `339` | `0.598759458` | `0.048759458` | `outside_stable_extreme_probability_envelope` | `base_oof_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441.csv` |
| `298` | `0.599919989` | `0.049919989` | `outside_stable_mean_move_envelope` | `base_oof_id_298_post297_src289_q_all_partial_cv0.527647.csv` |

## Top Credible Shape Rows

| id | proxy | gap to 0.55 | file |
|---:|---:|---:|---|
| `342` | `0.596318745` | `0.046318745` | `base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv` |
| `198` | `0.598698766` | `0.048698766` | `base_oof_id_198_id193_s3_smartphone_light_compact_overlay_cv0.536380.csv` |
| `193` | `0.598747389` | `0.048747389` | `base_oof_id_193_id191_sequence_summary_embedding_overlay_cv0.536429.csv` |
| `344` | `0.600157529` | `0.050157529` | `base_oof_id_344_id338_s3_phone_wearable_w0020_cv0.537839.csv` |
| `343` | `0.600172699` | `0.050172699` | `base_oof_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854.csv` |
| `338` | `0.600179633` | `0.050179633` | `base_oof_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861.csv` |
| `191` | `0.600328216` | `0.050328216` | `base_oof_id_191_id190_tempered_id177_remote_s1_anchor_blend_cv0.538009.csv` |
| `337` | `0.600385136` | `0.050385136` | `base_oof_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066.csv` |
| `179` | `0.600608667` | `0.050608667` | `base_oof_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290.csv` |
| `340` | `0.600744875` | `0.050744875` | `base_oof_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426.csv` |
| `341` | `0.600902399` | `0.050902399` | `base_oof_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584.csv` |
