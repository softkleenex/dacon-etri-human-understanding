# Post-ID342 validation-weighted pseudo-public split audit

Created: `2026-06-25 06:32:38 UTC+09:00`

This is report-only. It creates no predictions, feature matrices, submission artifacts, Kaggle runs, or DACON submissions.

## Decision

- OOF files scanned: `247`
- Scored rows: `8`
- Lens configs evaluated: `15`
- Review-lens pass count: `0`
- Primary config: `full_oof__id342_target_bottleneck`
- Primary stable LOO MAE: `0.0007592943485617195`
- Primary stable pairwise: `1.0`
- Primary all-scored pairwise: `0.7857142857142857`
- Primary recent slate order: `[342, 339, 338]`
- Anchor weighted projection: `0.5962115319988912`
- Public-0.55 weighted pass count: `0`
- Credible Public-0.55 weighted pass count: `0`
- Public-0.58 review pass count: `22`
- Credible Public-0.58 review pass count: `0`
- Validation-weighted gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: Use the primary validation-weighted lens as a stricter audit lens only. If it finds no credible Public-0.55 or Public-0.58 stable-shape row, close this validation-redesign branch for candidate creation and require a materially new source hypothesis.

## Lens Configs

| rank | config | pass | stable LOO MAE | stable pairwise | all pairwise | slate order | failed margin |
|---:|---|---:|---:|---:|---:|---|---:|
| `1` | `full_oof__id342_target_bottleneck` | `False` | `0.000759294` | `1.000` | `0.786` | `[342, 339, 338]` | `-0.014237111` |
| `2` | `subject_hole_loso_mean__id342_target_bottleneck` | `False` | `0.000759294` | `1.000` | `0.786` | `[342, 339, 338]` | `-0.014237111` |
| `3` | `subject_hole_loso_mean__stable_target_reliability` | `False` | `0.000846344` | `1.000` | `0.786` | `[342, 339, 338]` | `-0.015161577` |
| `4` | `full_oof__stable_target_reliability` | `False` | `0.000846344` | `1.000` | `0.786` | `[342, 339, 338]` | `-0.015161577` |
| `5` | `full_oof__uniform` | `False` | `0.000847312` | `1.000` | `0.786` | `[342, 339, 338]` | `-0.013967973` |
| `6` | `subject_hole_loso_mean__uniform` | `False` | `0.000847312` | `1.000` | `0.786` | `[342, 339, 338]` | `-0.013967973` |
| `7` | `primary_worst__stable_target_reliability` | `False` | `0.003154617` | `0.933` | `0.750` | `[342, 339, 338]` | `-0.022554000` |
| `8` | `primary_mean__stable_target_reliability` | `False` | `0.001148380` | `0.867` | `0.714` | `[342, 339, 338]` | `-0.016999623` |
| `9` | `test_neighbor_mean__stable_target_reliability` | `False` | `0.001727191` | `0.867` | `0.714` | `[342, 339, 338]` | `-0.018495308` |
| `10` | `primary_mean__id342_target_bottleneck` | `False` | `0.000686620` | `0.733` | `0.643` | `[342, 339, 338]` | `-0.015788062` |
| `11` | `primary_mean__uniform` | `False` | `0.000733648` | `0.733` | `0.643` | `[342, 339, 338]` | `-0.015491031` |
| `12` | `test_neighbor_mean__uniform` | `False` | `0.000862320` | `0.733` | `0.643` | `[342, 339, 338]` | `-0.017044055` |

## Target Weights

| target | weight |
|---|---:|
| `Q1` | `0.146027` |
| `Q2` | `0.147568` |
| `Q3` | `0.150676` |
| `S1` | `0.124838` |
| `S2` | `0.141547` |
| `S3` | `0.131071` |
| `S4` | `0.158274` |

## Top Weighted Rows

| id | projection | gap to 0.55 | label | file |
|---:|---:|---:|---|---|
| `313` | `0.558898896` | `+0.008898896` | `legacy_public_guided_family` | `base_oof_id_313_post307_aggbudget_cap120_lam000_frontierstress_cv0.488205.csv` |
| `312` | `0.566786985` | `+0.016786985` | `legacy_public_guided_family` | `base_oof_id_312_post307_aggbudget_cap080_lam000_scorepressure_cv0.496824.csv` |
| `255` | `0.568810378` | `+0.018810378` | `legacy_public_guided_family` | `base_oof_id_255_id178_anchor_fine_cap012_frontier_bridge_cv0.469816.csv` |
| `291` | `0.568976397` | `+0.018976397` | `outside_stable_mean_move_envelope` | `base_oof_id_291_post287_clean_anchor_cap060_cv0.498891.csv` |
| `256` | `0.570779957` | `+0.020779957` | `legacy_public_guided_family` | `base_oof_id_256_id178_anchor_fine_cap014_frontier_bridge_cv0.464121.csv` |
| `290` | `0.571124860` | `+0.021124860` | `outside_stable_mean_move_envelope` | `base_oof_id_290_post287_clean_anchor_cap050_cv0.504921.csv` |
| `303` | `0.571248164` | `+0.021248164` | `outside_stable_mean_move_envelope` | `base_oof_id_303_post300_temporal_cap050_tail45_score_cv0.505102.csv` |
| `247` | `0.572231072` | `+0.022231072` | `legacy_public_guided_family` | `base_oof_id_247_id235_frontier_twise_cap020_logit_bridge_cv0.415523.csv` |
| `253` | `0.573374421` | `+0.023374421` | `legacy_public_guided_family` | `base_oof_id_253_id178_anchor_cap012_frontier_bridge_cv0.477131.csv` |
| `294` | `0.573394506` | `+0.023394506` | `outside_stable_mean_move_envelope` | `base_oof_id_294_post287_base290_antifailed_score_cv0.507898.csv` |
| `282` | `0.574473902` | `+0.024473902` | `outside_stable_mean_move_envelope` | `base_oof_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400.csv` |
| `278` | `0.575048305` | `+0.025048305` | `outside_stable_mean_move_envelope` | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |

## Top Credible Shape Rows

| id | projection | gap to 0.55 | file |
|---:|---:|---:|---|
| `342` | `0.596211532` | `+0.046211532` | `base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv` |
| `198` | `0.598748436` | `+0.048748436` | `base_oof_id_198_id193_s3_smartphone_light_compact_overlay_cv0.536380.csv` |
| `193` | `0.598793047` | `+0.048793047` | `base_oof_id_193_id191_sequence_summary_embedding_overlay_cv0.536429.csv` |
| `344` | `0.600279417` | `+0.050279417` | `base_oof_id_344_id338_s3_phone_wearable_w0020_cv0.537839.csv` |
| `343` | `0.600293335` | `+0.050293335` | `base_oof_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854.csv` |
| `338` | `0.600299697` | `+0.050299697` | `base_oof_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861.csv` |
| `191` | `0.600429538` | `+0.050429538` | `base_oof_id_191_id190_tempered_id177_remote_s1_anchor_blend_cv0.538009.csv` |
| `337` | `0.600492349` | `+0.050492349` | `base_oof_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066.csv` |
| `179` | `0.600674614` | `+0.050674614` | `base_oof_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290.csv` |
| `340` | `0.600773044` | `+0.050773044` | `base_oof_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426.csv` |
| `341` | `0.600935762` | `+0.050935762` | `base_oof_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584.csv` |
