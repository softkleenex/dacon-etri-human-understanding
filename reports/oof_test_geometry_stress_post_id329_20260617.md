# OOF test-geometry stress diagnostic — post ID329

Created: `2026-06-17 21:50:00 UTC+09:00`

This diagnostic scans existing base OOF predictions only. No model was trained, no submission CSV was created, and no DACON submission was made.

## Verdict

**Do not use post-hoc OOF subset loss as a promotion gate.** It fails the most important sanity check: ID287 still looks excellent by these losses even though its confirmed Public score collapsed to `0.6698787318`.

## Anchor check

| id | overall OOF | boundary OOF | band OOF | public | status |
|---:|---:|---:|---:|---:|---|
| 178 | 0.538638 | 0.586290 | 0.541604 | 0.601411 | `cv_lb_gap_risk` |
| 287 | 0.378934 | 0.426040 | 0.389428 | 0.669879 | `cv_lb_gap_risk` |
| 329 | 0.516718 | 0.558058 | 0.520499 | 0.601791 | `cv_lb_gap_risk` |

ID287 boundary rank: `11` / `236`
ID287 band rank: `9` / `236`

## Top boundary OOF rows

| rank | id | overall | boundary | band | public | file |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 250 | 0.261184 | 0.288405 | 0.270703 |  | `base_oof_id_250_id242_iso_non_s3_s4base_ultra_frontier_cv0.261184.csv` |
| 2 | 251 | 0.263237 | 0.290434 | 0.272594 |  | `base_oof_id_251_id242_quantile_non_s3_s4q100_ultra_frontier_cv0.263237.csv` |
| 3 | 252 | 0.274420 | 0.305184 | 0.285964 |  | `base_oof_id_252_id249_iso_mixed_second_stage_frontier_cv0.274420.csv` |
| 4 | 242 | 0.357651 | 0.378854 | 0.364805 |  | `base_oof_id_242_id238_q240_s4iso_recal_frontier_cv0.357651.csv` |
| 5 | 243 | 0.360420 | 0.380950 | 0.366081 |  | `base_oof_id_243_id238_q240_all_recal_frontier_cv0.360420.csv` |
| 6 | 241 | 0.383459 | 0.397022 | 0.390760 |  | `base_oof_id_241_id238_q160_s4iso_recal_frontier_cv0.383459.csv` |
| 7 | 249 | 0.374465 | 0.398821 | 0.383290 |  | `base_oof_id_249_id235_frontier_twise_cap040_logit_bridge_cv0.374465.csv` |
| 8 | 285 | 0.356330 | 0.400947 | 0.367480 |  | `base_oof_id_285_post282_base278_tempguard_calibration_cv0.356330.csv` |
| 9 | 248 | 0.391735 | 0.418376 | 0.401832 |  | `base_oof_id_248_id235_frontier_twise_cap030_logit_bridge_cv0.391735.csv` |
| 10 | 288 | 0.372773 | 0.420182 | 0.384668 |  | `base_oof_id_288_post285_base278_fixedtemp_frontier_cv0.372773.csv` |
| 11 | 287 | 0.378934 | 0.426040 | 0.389428 | 0.669879 | `base_oof_id_287_post285_base282_fixedtemp_balanced_cv0.378934.csv` |
| 12 | 278 | 0.389120 | 0.431325 | 0.398661 |  | `base_oof_id_278_post274_refined_cap300_p005_blend_cv0.389120.csv` |
| 13 | 284 | 0.385005 | 0.433033 | 0.395570 |  | `base_oof_id_284_post282_base280_tempguard_calibration_cv0.385005.csv` |
| 14 | 282 | 0.393400 | 0.435688 | 0.402138 |  | `base_oof_id_282_post278_gapaware_cap280_sp005_dp100_blend_cv0.393400.csv` |
| 15 | 240 | 0.417349 | 0.438547 | 0.423449 |  | `base_oof_id_240_id238_q80_s4iso_recal_frontier_cv0.417349.csv` |

## Interpretation

- The test-geometry subset is useful for describing where the hidden rows live, but post-hoc OOF predictions can remain contaminated by the same full-OOF/non-nested effects that made ID287 fail.
- A valid future gate must be fold-nested: candidate selection, calibration, and blending must be learned without the outer validation rows.
- No Public 0.58-plausible candidate is promoted by this diagnostic.

