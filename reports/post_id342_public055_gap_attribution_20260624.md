# Post-ID342 Public-0.55 gap attribution

Created: `2026-06-24 21:57:51 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- OOF files scanned: `247`
- Stable public gap median: `0.062318739`
- Required full local for Public 0.55: `0.487681261`
- ID342 local margin above required: `+0.046318745`
- Existing rows below required local: `64`
- Credible rows below required local: `0`
- Best projected candidate: `ID250` / `0.323502620`
- Best low-movement projected candidate: `ID211` / `0.582424274`
- DACON submit now: `False`
- Next action: Do not submit existing low-local frontier rows. Public 0.55 requires either a validated local score near the frontier without frontier-family movement risk, or a new mechanism that compresses the public-minus-local gap; continue with validation redesign/public-gap attribution.

## Top Projected Rows

| id | local | projected public | gap 0.55 | mean move | max move | risk | file |
|---:|---:|---:|---:|---:|---:|---|---|
| `250` | `0.261184` | `0.323503` | `-0.226497` | `0.232629` | `0.871435` | `frontier_family_unscored` | `base_oof_id_250_id242_iso_non_s3_s4base_ultra_frontier_cv0.261184.csv` |
| `251` | `0.263237` | `0.325556` | `-0.224444` | `0.230526` | `0.867199` | `frontier_family_unscored` | `base_oof_id_251_id242_quantile_non_s3_s4q100_ultra_frontier_cv0.263237.csv` |
| `252` | `0.274420` | `0.336739` | `-0.213261` | `0.217815` | `0.871435` | `frontier_family_unscored` | `base_oof_id_252_id249_iso_mixed_second_stage_frontier_cv0.274420.csv` |
| `285` | `0.356330` | `0.418649` | `-0.131351` | `0.143640` | `0.494176` | `large_distribution_move` | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| `242` | `0.357651` | `0.419969` | `-0.130031` | `0.162656` | `0.744625` | `frontier_family_unscored` | `base_oof_id_242_id238_q240_s4iso_recal_frontier_cv0.357651.csv` |
| `243` | `0.360420` | `0.422739` | `-0.127261` | `0.159569` | `0.704769` | `frontier_family_unscored` | `base_oof_id_243_id238_q240_all_recal_frontier_cv0.360420.csv` |
| `288` | `0.372773` | `0.435092` | `-0.114908` | `0.125657` | `0.389695` | `frontier_family_unscored` | `base_oof_id_288_post285_base278_fixedtemp_frontier_cv0.372773.csv` |
| `249` | `0.374465` | `0.436784` | `-0.113216` | `0.133734` | `0.603115` | `frontier_family_unscored` | `base_oof_id_249_id235_frontier_twise_cap040_logit_bridge_cv0.374465.csv` |
| `287` | `0.378934` | `0.441253` | `-0.108747` | `0.119407` | `0.362732` | `scored_failed_frontier_trap` | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| `241` | `0.383459` | `0.445778` | `-0.104222` | `0.152927` | `0.744625` | `frontier_family_unscored` | `base_oof_id_241_id238_q160_s4iso_recal_frontier_cv0.383459.csv` |
| `284` | `0.385005` | `0.447324` | `-0.102676` | `0.120460` | `0.443909` | `large_distribution_move` | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| `278` | `0.389120` | `0.451439` | `-0.098561` | `0.105452` | `0.343984` | `large_distribution_move` | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |
| `248` | `0.391735` | `0.454054` | `-0.095946` | `0.122824` | `0.551317` | `frontier_family_unscored` | `base_oof_id_248_id235_frontier_twise_cap030_logit_bridge_cv0.391735.csv` |
| `282` | `0.393400` | `0.455719` | `-0.094281` | `0.101805` | `0.338669` | `large_distribution_move` | `base_oof_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400.csv` |
| `286` | `0.395368` | `0.457687` | `-0.092313` | `0.107067` | `0.321193` | `large_distribution_move` | `base_oof_id_286_post285_base280_fixedtemp_guard_cv0.395368.csv` |
| `277` | `0.399355` | `0.461673` | `-0.088327` | `0.099530` | `0.338669` | `large_distribution_move` | `base_oof_id_277_post274_refined_cap280_p010_blend_cv0.399355.csv` |
| `281` | `0.399583` | `0.461902` | `-0.088098` | `0.097326` | `0.323180` | `large_distribution_move` | `base_oof_id_281_post278_gapaware_cap260_sp005_dp100_blend_cv0.399583.csv` |
| `280` | `0.406704` | `0.469023` | `-0.080977` | `0.092316` | `0.303473` | `large_distribution_move` | `base_oof_id_280_post278_gapaware_cap240_sp005_dp100_blend_cv0.406704.csv` |
| `276` | `0.411217` | `0.473536` | `-0.076464` | `0.091704` | `0.310246` | `large_distribution_move` | `base_oof_id_276_post274_refined_cap260_p010_blend_cv0.411217.csv` |
| `279` | `0.414169` | `0.476488` | `-0.073512` | `0.087229` | `0.283435` | `large_distribution_move` | `base_oof_id_279_post278_gapaware_cap220_sp005_dp100_blend_cv0.414169.csv` |

## Low-Movement Rows

| id | local | projected public | gap 0.55 | mean move | max move | risk | file |
|---:|---:|---:|---:|---:|---:|---|---|
| `211` | `0.520106` | `0.582424` | `+0.032424` | `0.024741` | `0.117868` | `not_close_to_public055_threshold` | `base_oof_id_211_id209_targetwise_delta08_id172_extreme_probe_cv0.520106.csv` |
| `222` | `0.521966` | `0.584285` | `+0.034285` | `0.021982` | `0.118659` | `not_close_to_public055_threshold` | `base_oof_id_222_id221_targetwise_delta06_id172_extreme_probe_cv0.521966.csv` |
| `210` | `0.523400` | `0.585719` | `+0.035719` | `0.021479` | `0.103918` | `not_close_to_public055_threshold` | `base_oof_id_210_id209_targetwise_delta06_id172_extreme_probe_cv0.523400.csv` |
| `304` | `0.532349` | `0.594668` | `+0.044668` | `0.019386` | `0.118016` | `not_close_to_public055_threshold` | `base_oof_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349.csv` |
| `342` | `0.534000` | `0.596319` | `+0.046319` | `0.000000` | `0.000000` | `not_close_to_public055_threshold` | `base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv` |
| `212` | `0.534469` | `0.596787` | `+0.046787` | `0.019046` | `0.104201` | `not_close_to_public055_threshold` | `base_oof_id_212_id206_shapelet_motif_sequence_wide_weight_overlay_cv0.534469.csv` |
| `209` | `0.534530` | `0.596849` | `+0.046849` | `0.017963` | `0.081188` | `not_close_to_public055_threshold` | `base_oof_id_209_id206_shapelet_motif_sequence_overlay_cv0.534530.csv` |
| `206` | `0.535764` | `0.598083` | `+0.048083` | `0.018681` | `0.112836` | `not_close_to_public055_threshold` | `base_oof_id_206_id203_wavelet_spectral_sequence_overlay_cv0.535764.csv` |
| `203` | `0.535980` | `0.598299` | `+0.048299` | `0.018408` | `0.115491` | `not_close_to_public055_threshold` | `base_oof_id_203_id200_s4_recurrence_gaf_sequence_overlay_cv0.535980.csv` |
| `307` | `0.537792` | `0.600111` | `+0.050111` | `0.018926` | `0.119484` | `not_close_to_public055_threshold` | `base_oof_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792.csv` |
