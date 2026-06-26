# Post-ID335 public-anchor rerank

Created: `2026-06-17 22:27:52 UTC+09:00`

This is a no-submission diagnostic. The proxy is a screening heuristic, not a leaderboard prediction.

## Tier counts

- `reject_high_public_risk`: 174
- `diagnostic_only`: 43
- `reviewable_low_movement`: 12
- `known_public_reference`: 2
- `same_family_freeze_after_id329`: 5
- `prepared_sequence_probe_reference`: 2

## Top reviewable low-movement candidates

| rank | id | local | proxy | family | mean Δ178 | max Δ178 | proj failed | status | file |
|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| 1 | 307 | 0.537792 | 0.605472 | `failed_direction_combo_family` | 0.005753 | 0.026555 | -0.0170 | `generated_unsubmitted` | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 2 | 295 | 0.548583 | 0.605688 | `failed_direction_combo_family` | 0.005768 | 0.014923 | -0.0500 | `generated_unsubmitted` | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 3 | 304 | 0.532349 | 0.605919 | `failed_direction_combo_family` | 0.008162 | 0.029765 | -0.0000 | `generated_unsubmitted` | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 4 | 297 | 0.550861 | 0.606165 | `failed_direction_combo_family` | 0.007442 | 0.018164 | -0.0614 | `generated_unsubmitted` | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |
| 5 | 179 | 0.53829 | 0.606255 | `sequence_temporal_family` | 0.003991 | 0.051824 | +0.0011 | `generated_unsubmitted` | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 6 | 305 | 0.52957 | 0.607127 | `failed_direction_combo_family` | 0.012628 | 0.048772 | +0.0000 | `generated_unsubmitted` | `submission_id_305_post303_orthogonal_cap050_proj000_balanced_cv0.529570_unsubmitted.csv` |
| 7 | 296 | 0.558807 | 0.607791 | `failed_direction_combo_family` | 0.011535 | 0.029846 | -0.1000 | `generated_unsubmitted` | `submission_id_296_post287_anchor_antivector_prob_m010_cv0.558807_unsubmitted.csv` |
| 8 | 306 | 0.526367 | 0.608861 | `failed_direction_combo_family` | 0.017771 | 0.074735 | +0.0000 | `generated_unsubmitted` | `submission_id_306_post303_orthogonal_cap080_proj000_aggressive_cv0.526367_unsubmitted.csv` |
| 9 | 311 | 0.524195 | 0.609719 | `other_family` | 0.018933 | 0.091284 | -0.0000 | `generated_unsubmitted` | `submission_id_311_post307_aggbudget_cap100_lam160_zeroproj_cv0.524195_unsubmitted.csv` |
| 10 | 177 | 0.540607 | 0.612052 | `sequence_temporal_family` | 0.018594 | 0.115165 | +0.0012 | `generated_unsubmitted` | `submission_id_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.csv` |
| 11 | 310 | 0.521885 | 0.612363 | `other_family` | 0.026107 | 0.114878 | +0.0007 | `generated_unsubmitted` | `submission_id_310_post307_aggbudget_cap120_lam160_zeroproj_cv0.521885_unsubmitted.csv` |
| 12 | 170 | 0.546579 | 0.619720 | `calibration_frontier_family` | 0.036503 | 0.118629 | +0.0142 | `generated_unsubmitted` | `submission_id_170_id168_outer_supported_partial_calibration_probe_cv0.546579_unsubmitted.csv` |

## Prepared sequence references

| id | local | proxy | mean Δ178 | max Δ178 | proj failed | file |
|---:|---:|---:|---:|---:|---:|---|
| 334 | 0.540347 | 0.610986 | 0.010290 | 0.119120 | +0.0039 | `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv` |
| 335 | 0.539690 | 0.609846 | 0.007392 | 0.111949 | +0.0032 | `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv` |

## Recommendation

Do not submit from this rerank alone. If continuing locally, inspect top low-movement non-failed-family candidates against ID334/ID335 stress behavior, then require a decision note before any generated artifact or public probe.

