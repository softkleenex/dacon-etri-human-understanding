# Post-ID342 stable-lens existing-candidate triage

Created: `2026-06-24 04:53:37 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Decision

- Scanned OOF candidates: `247`
- Unscored candidates: `239`
- Stable-lens review pass count: `0`
- Lineage near-miss count: `2`
- Lineage near-miss IDs: `[222, 221]`
- Best unscored candidate: `250`
- Best unscored full OOF / delta vs ID342: `0.261184` / `-0.272816124`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Reason: No existing unscored candidate passed the stable-lens review. Strong local rows remain dominated by failed-frontier, public-scored, missing-test, or movement-risk checks.

## Public-Scored References

| id | full OOF | public | delta 342 | failed cos | test mean delta | pass | risks |
|---:|---:|---:|---:|---:|---:|:---:|---|
| 342 | 0.534000 | 0.596083252 | +0.000000 | 0.000 | 0.000000 | `False` | `already_has_public_feedback,material_local_gain_vs_id342,not_already_public_scored` |
| 339 | 0.536441 | 0.598518779 | +0.002441 | 0.420 | 0.014734 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,material_local_gain_vs_id342,not_already_public_scored,not_closer_to_failed_frontier_than_id342` |
| 338 | 0.537861 | 0.599214518 | +0.003861 | 0.627 | 0.017470 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,material_local_gain_vs_id342,not_already_public_scored,not_closer_to_failed_frontier_than_id342` |
| 337 | 0.538066 | 0.600620629 | +0.004066 | 0.681 | 0.018285 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,material_local_gain_vs_id342,not_already_public_scored,not_closer_to_failed_frontier_than_id342` |
| 340 | 0.538426 | 0.601379175 | +0.004426 | 0.689 | 0.018028 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,material_local_gain_vs_id342,not_already_public_scored,not_closer_to_failed_frontier_than_id342` |
| 178 | 0.538638 | 0.601411301 | +0.004638 | 0.689 | 0.018463 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,lineage_not_public_guided_or_unnested,material_local_gain_vs_id342,not_already_public_scored` |
| 329 | 0.516718 | 0.601790858 | -0.017282 | 1.000 | 0.025343 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_already_public_scored` |
| 287 | 0.378934 | 0.669878732 | -0.155066 | 1.000 | 0.121174 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_already_public_scored` |

## Passing Rows

| rank | id | full OOF | delta 342 | proxy | gap 0.55 | failed cos | test mean delta | risks |
|---:|---:|---:|---:|---:|---:|---:|---:|---|

## Lineage Near Misses

| rank | id | full OOF | delta 342 | proxy | failed cos | test mean delta | blocking note |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 222 | 0.521966 | -0.012034 | 0.584049 | 0.148 | 0.021032 | Public-structure tempered calibration probe: target-wise clean weights {'Q1': 0.897, 'Q2': 0.857, 'Q3': 0.843, 'S1': 0.83, 'S2': 0.885, 'S3': 0.68, 'S4': 0.841}. Not strict Subject-Hole CV, not fold-nested; no DACON subm |
| 2 | 221 | 0.532990 | -0.001010 | 0.595073 | 0.000 | 0.018463 | Public-structure feature-similarity overlay on ID218; target-wise configs selected from reports/id218_public_feature_similarity_narrow_grid_20260616.json. Not strict/nested; no DACON submission was made. / 2026-06-21 DAC |

## Top Local Rows

| rank | id | full OOF | delta 342 | public | failed cos | failed proj | test mean delta | pass | risks |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 250 | 0.261184 | -0.272816 |  | 0.902 | 3.723 | 0.244621 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_frontier_low_loss,oof_movement_within_anchor_band` |
| 2 | 251 | 0.263237 | -0.270763 |  | 0.901 | 3.712 | 0.242351 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_frontier_low_loss,oof_movement_within_anchor_band` |
| 3 | 252 | 0.274420 | -0.259580 |  | 0.918 | 3.510 | 0.222971 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 4 | 285 | 0.356330 | -0.177670 |  | 0.965 | 1.804 | 0.144080 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 5 | 242 | 0.357651 | -0.176349 |  | 0.748 | 3.189 | 0.170896 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 6 | 243 | 0.360420 | -0.173580 |  | 0.743 | 3.171 | 0.166995 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 7 | 288 | 0.372773 | -0.161227 |  | 0.997 | 1.752 | 0.127111 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 8 | 249 | 0.374465 | -0.159535 |  | 0.796 | 2.677 | 0.139807 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 9 | 287 | 0.378934 | -0.155066 | 0.669878732 | 1.000 | 1.629 | 0.121174 | `False` | `already_has_public_feedback,failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_already_public_scored` |
| 10 | 241 | 0.383459 | -0.150541 |  | 0.614 | 2.897 | 0.161769 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_frontier_low_loss,oof_movement_within_anchor_band` |
| 11 | 284 | 0.385005 | -0.148995 |  | 0.969 | 1.517 | 0.120888 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 12 | 278 | 0.389120 | -0.144880 |  | 0.987 | 1.746 | 0.106903 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 13 | 248 | 0.391735 | -0.142265 |  | 0.782 | 2.395 | 0.127271 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 14 | 282 | 0.393400 | -0.140600 |  | 0.990 | 1.685 | 0.103458 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 15 | 286 | 0.395368 | -0.138632 |  | 0.999 | 1.498 | 0.108835 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 16 | 277 | 0.399355 | -0.134645 |  | 0.988 | 1.588 | 0.099451 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 17 | 281 | 0.399583 | -0.134417 |  | 0.991 | 1.612 | 0.099000 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 18 | 280 | 0.406704 | -0.127296 |  | 0.992 | 1.530 | 0.094070 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 19 | 276 | 0.411217 | -0.122783 |  | 0.984 | 1.501 | 0.090702 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 20 | 279 | 0.414169 | -0.119831 |  | 0.992 | 1.454 | 0.088989 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 21 | 245 | 0.415035 | -0.118965 |  | 0.715 | 2.131 | 0.116015 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 22 | 247 | 0.415523 | -0.118477 |  | 0.725 | 2.087 | 0.117900 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 23 | 275 | 0.416609 | -0.117391 |  | 0.988 | 1.465 | 0.086917 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 24 | 240 | 0.417349 | -0.116651 |  | 0.619 | 2.311 | 0.142465 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 25 | 274 | 0.423144 | -0.110856 |  | 0.984 | 1.400 | 0.081664 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |

## Review Rows

| rank | id | full OOF | delta 342 | proxy | gap 0.55 | failed cos | test mean delta | pass | risks |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| 1 | 250 | 0.261184 | -0.272816 | 0.323267 | -0.226733 | 0.902 | 0.244621 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_frontier_low_loss,oof_movement_within_anchor_band` |
| 2 | 251 | 0.263237 | -0.270763 | 0.325320 | -0.224680 | 0.901 | 0.242351 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_frontier_low_loss,oof_movement_within_anchor_band` |
| 3 | 252 | 0.274420 | -0.259580 | 0.336503 | -0.213497 | 0.918 | 0.222971 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 4 | 285 | 0.356330 | -0.177670 | 0.418414 | -0.131586 | 0.965 | 0.144080 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 5 | 242 | 0.357651 | -0.176349 | 0.419734 | -0.130266 | 0.748 | 0.170896 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 6 | 243 | 0.360420 | -0.173580 | 0.422503 | -0.127497 | 0.743 | 0.166995 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 7 | 288 | 0.372773 | -0.161227 | 0.434856 | -0.115144 | 0.997 | 0.127111 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 8 | 249 | 0.374465 | -0.159535 | 0.436548 | -0.113452 | 0.796 | 0.139807 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 9 | 241 | 0.383459 | -0.150541 | 0.445543 | -0.104457 | 0.614 | 0.161769 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_frontier_low_loss,oof_movement_within_anchor_band` |
| 10 | 284 | 0.385005 | -0.148995 | 0.447089 | -0.102911 | 0.969 | 0.120888 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 11 | 278 | 0.389120 | -0.144880 | 0.451203 | -0.098797 | 0.987 | 0.106903 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 12 | 248 | 0.391735 | -0.142265 | 0.453818 | -0.096182 | 0.782 | 0.127271 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 13 | 282 | 0.393400 | -0.140600 | 0.455484 | -0.094516 | 0.990 | 0.103458 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 14 | 286 | 0.395368 | -0.138632 | 0.457451 | -0.092549 | 0.999 | 0.108835 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 15 | 277 | 0.399355 | -0.134645 | 0.461438 | -0.088562 | 0.988 | 0.099451 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 16 | 281 | 0.399583 | -0.134417 | 0.461666 | -0.088334 | 0.991 | 0.099000 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 17 | 280 | 0.406704 | -0.127296 | 0.468787 | -0.081213 | 0.992 | 0.094070 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 18 | 276 | 0.411217 | -0.122783 | 0.473300 | -0.076700 | 0.984 | 0.090702 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 19 | 279 | 0.414169 | -0.119831 | 0.476253 | -0.073747 | 0.992 | 0.088989 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 20 | 245 | 0.415035 | -0.118965 | 0.477118 | -0.072882 | 0.715 | 0.116015 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 21 | 247 | 0.415523 | -0.118477 | 0.477607 | -0.072393 | 0.725 | 0.117900 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 22 | 275 | 0.416609 | -0.117391 | 0.478692 | -0.071308 | 0.988 | 0.086917 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 23 | 240 | 0.417349 | -0.116651 | 0.479433 | -0.070567 | 0.619 | 0.142465 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 24 | 274 | 0.423144 | -0.110856 | 0.485227 | -0.064773 | 0.984 | 0.081664 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
| 25 | 257 | 0.433079 | -0.100921 | 0.495163 | -0.054837 | 0.984 | 0.078031 | `False` | `failed_direction_cosine_low,failed_direction_projection_low,frontier_low_loss_band,lineage_not_public_guided_or_unnested,not_closer_to_failed_frontier_than_id342,not_frontier_low_loss` |
