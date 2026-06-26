# Post-ID329 public-anchor rerank

Created: `2026-06-17 01:23:38` KST

This is a no-submission diagnostic. It re-ranks existing base prediction artifacts using ID178 and ID329 as public anchors and ID287 as the failed direction.

Base-test IDs scanned: `236`.

## Tier counts

| tier | count |
|---|---:|
| `diagnostic_only` | 55 |
| `reject_high_public_risk` | 150 |
| `reviewable_low_movement` | 13 |
| `same_family_freeze_after_id329` | 18 |

## Top reviewable low-movement candidates

| rank | id | local | proxy | family | mean Δ178 | max Δ178 | proj failed | status | file |
|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| 1 | 177 | 0.540607 | 0.613871 | `sequence_temporal_family` | 0.018594 | 0.115165 | +0.0012 | `generated_unsubmitted` | `submission_id_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.csv` |
| 2 | 179 | 0.538290 | 0.617308 | `anchor_bridge_family` | 0.003991 | 0.051824 | +0.0011 | `generated_unsubmitted` | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 3 | 295 | 0.548583 | 0.617918 | `anchor_bridge_family` | 0.005768 | 0.014923 | -0.0500 | `generated_unsubmitted` | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 4 | 297 | 0.550861 | 0.617927 | `anchor_bridge_family` | 0.007442 | 0.018164 | -0.0614 | `generated_unsubmitted` | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |
| 5 | 296 | 0.558807 | 0.619069 | `anchor_bridge_family` | 0.011535 | 0.029846 | -0.1000 | `generated_unsubmitted` | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 6 | 307 | 0.537792 | 0.623287 | `failed_direction_combo_family` | 0.005753 | 0.026555 | -0.0170 | `generated_unsubmitted` | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 7 | 304 | 0.532349 | 0.624226 | `failed_direction_combo_family` | 0.008162 | 0.029765 | -0.0000 | `generated_unsubmitted` | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 8 | 181 | 0.526430 | 0.624398 | `other_family` | 0.017042 | 0.085372 | +0.0479 | `generated_unsubmitted` | `submission_id_181_id179_targetwise_delta06_id172_extreme_probe_cv0.526430_unsubmitted.csv` |
| 9 | 305 | 0.529570 | 0.625923 | `failed_direction_combo_family` | 0.012628 | 0.048772 | +0.0000 | `generated_unsubmitted` | `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv` |
| 10 | 180 | 0.529579 | 0.626004 | `anchor_bridge_family` | 0.012873 | 0.072340 | +0.0348 | `generated_unsubmitted` | `submission_id_180_id179_tempered_id172_extreme_anchor_probe_cv0.529579_unsubmitted.csv` |
| 11 | 306 | 0.526367 | 0.628018 | `failed_direction_combo_family` | 0.017771 | 0.074735 | +0.0000 | `generated_unsubmitted` | `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv` |
| 12 | 170 | 0.546579 | 0.657187 | `calibration_frontier_family` | 0.036503 | 0.118629 | +0.0142 | `generated_unsubmitted` | `submission_id_170_id168_outer_supported_partial_calibration_probe_cv0.546579_unsubmitted.csv` |

## Top non-same-family reviewable candidates

| rank | id | local | proxy | family | mean Δ178 | max Δ178 | proj failed | file |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 177 | 0.540607 | 0.613871 | `sequence_temporal_family` | 0.018594 | 0.115165 | +0.0012 | `submission_id_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.csv` |
| 2 | 179 | 0.538290 | 0.617308 | `anchor_bridge_family` | 0.003991 | 0.051824 | +0.0011 | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 3 | 295 | 0.548583 | 0.617918 | `anchor_bridge_family` | 0.005768 | 0.014923 | -0.0500 | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 4 | 297 | 0.550861 | 0.617927 | `anchor_bridge_family` | 0.007442 | 0.018164 | -0.0614 | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |
| 5 | 296 | 0.558807 | 0.619069 | `anchor_bridge_family` | 0.011535 | 0.029846 | -0.1000 | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 6 | 181 | 0.526430 | 0.624398 | `other_family` | 0.017042 | 0.085372 | +0.0479 | `submission_id_181_id179_targetwise_delta06_id172_extreme_probe_cv0.526430_unsubmitted.csv` |
| 7 | 180 | 0.529579 | 0.626004 | `anchor_bridge_family` | 0.012873 | 0.072340 | +0.0348 | `submission_id_180_id179_tempered_id172_extreme_anchor_probe_cv0.529579_unsubmitted.csv` |
| 8 | 170 | 0.546579 | 0.657187 | `calibration_frontier_family` | 0.036503 | 0.118629 | +0.0142 | `submission_id_170_id168_outer_supported_partial_calibration_probe_cv0.546579_unsubmitted.csv` |

## Interpretation

- The proxy is deliberately conservative and should not be read as a predicted leaderboard score.
- Same-family constrained-combo candidates remain frozen after ID329.
- The next useful work is manual review of non-same-family, low-movement candidates and the demographic/static-feature strict CV audit; no DACON submission was made.
