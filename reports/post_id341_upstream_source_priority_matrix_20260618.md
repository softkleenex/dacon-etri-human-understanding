# Post-ID341 upstream source priority matrix

Created: `2026-06-18 05:13:44 UTC+09:00`

This is report-only. It trains no model, creates no submission artifact, and makes no DACON submission.

## Rebuild queue

| rank | id | action | gain vs ID178 | wins vs ID341 | proj failed ID287 | reason |
|---:|---:|---|---:|---:|---:|---|
| 1 | 221 | `rebuild_clean_root_feature_similarity` | +0.005648 | 21 | -0.0102 | ID221 is weaker than later calibration artifacts but is an upstream prerequisite for ID226/ID238. |
| 2 | 226 | `rebuild_primary_convex_backbone` | +0.046867 | 23 | +0.2995 | ID226 carries the main reusable upstream signal and sits below ID227/228/237 and downstream A1/B1 artifacts. |
| 3 | 227 | `replay_only_after_clean_226` | +0.074229 | 23 | +0.4971 | This is an ID226 calibration derivative; rebuilding it before a clean ID226 would preserve the same full-OOF failure mode. |
| 3 | 237 | `replay_only_after_clean_226` | +0.071947 | 23 | +0.5164 | This is an ID226 calibration derivative; rebuilding it before a clean ID226 would preserve the same full-OOF failure mode. |
| 3 | 228 | `replay_only_after_clean_226` | +0.056393 | 23 | +0.3934 | This is an ID226 calibration derivative; rebuilding it before a clean ID226 would preserve the same full-OOF failure mode. |

## Source matrix

| id | layer | status | local log | full OOF | gain vs 178 | gain vs 341 | fold wins vs 341 | max Δ341 test | proj failed | action |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 221 | `upstream_model_layer` | `generated_unsubmitted` | 0.532990 | 0.532990 | +0.005648 | +0.005593 | 21/23 | 0.187821 | -0.0102 | `rebuild_clean_root_feature_similarity` |
| 226 | `upstream_model_layer` | `generated_unsubmitted` | 0.491771 | 0.491771 | +0.046867 | +0.046813 | 23/23 | 0.535822 | +0.2995 | `rebuild_primary_convex_backbone` |
| 227 | `upstream_model_layer` | `generated_unsubmitted` | 0.464409 | 0.464409 | +0.074229 | +0.074175 | 23/23 | 0.608673 | +0.4971 | `replay_only_after_clean_226` |
| 228 | `upstream_model_layer` | `generated_unsubmitted` | 0.482245 | 0.482245 | +0.056393 | +0.056339 | 23/23 | 0.577488 | +0.3934 | `replay_only_after_clean_226` |
| 237 | `upstream_model_layer` | `generated_unsubmitted` | 0.466691 | 0.466691 | +0.071947 | +0.071893 | 23/23 | 0.608673 | +0.5164 | `replay_only_after_clean_226` |
| 238 | `a1_replay_layer` | `generated_unsubmitted` | 0.462365 | 0.462365 | +0.076273 | +0.076219 | 23/23 | 0.579587 | +0.4610 | `hold_until_clean_upstream_layer_exists` |
| 239 | `a1_replay_layer` | `generated_unsubmitted` | 0.437967 | 0.437967 | +0.100670 | +0.100616 | 23/23 | 0.733672 | +0.6369 | `hold_until_clean_upstream_layer_exists` |
| 241 | `a1_replay_layer` | `generated_unsubmitted` | 0.383459 | 0.383459 | +0.155178 | +0.155124 | 23/23 | 0.733672 | +0.8455 | `hold_until_clean_upstream_layer_exists` |
| 254 | `b1_replay_layer` | `generated_unsubmitted` | 0.471985 | 0.471985 | +0.066653 | +0.066599 | 23/23 | 0.146734 | +0.3383 | `dependency_audit_only_for_now` |
| 255 | `b1_replay_layer` | `generated_unsubmitted` | 0.469816 | 0.469816 | +0.068821 | +0.068767 | 23/23 | 0.115233 | +0.3765 | `dependency_audit_only_for_now` |
| 256 | `b1_replay_layer` | `generated_unsubmitted` | 0.464121 | 0.464121 | +0.074517 | +0.074463 | 23/23 | 0.131771 | +0.3865 | `dependency_audit_only_for_now` |
| 257 | `b1_replay_layer` | `generated_unsubmitted` | 0.433079 | 0.433079 | +0.105558 | +0.105504 | 23/23 | 0.176775 | +0.6075 | `dependency_audit_only_for_now` |
| 258 | `b1_replay_layer` | `generated_unsubmitted` | 0.457647 | 0.457647 | +0.080991 | +0.080937 | 23/23 | 0.159601 | +0.4397 | `dependency_audit_only_for_now` |
| 266 | `b1_replay_layer` | `generated_unsubmitted` | 0.457894 | 0.457894 | +0.080743 | +0.080689 | 23/23 | 0.150039 | +0.4386 | `dependency_audit_only_for_now` |
| 178 | `public_anchor_layer` | `cv_lb_gap_risk` | 0.538638 | 0.538638 | +0.000000 | -0.000054 | 4/23 | 0.002025 | +0.0000 | `reference_only` |
| 287 | `public_anchor_layer` | `cv_lb_gap_risk` | 0.378934 | 0.378934 | +0.159704 | +0.159650 | 23/23 | 0.298460 | +1.0000 | `reference_only` |
| 329 | `public_anchor_layer` | `cv_lb_gap_risk` | 0.516718 | 0.516718 | +0.021919 | +0.021865 | 23/23 | 0.079192 | +0.0499 | `reference_only` |
| 341 | `public_anchor_layer` | `generated_unsubmitted` | 0.538584 | 0.538584 | +0.000054 | +0.000000 | 0/23 | 0.000000 | -0.0001 | `reference_only` |

## Upstream target deltas versus ID178

| id | Q1 | Q2 | Q3 | S1 | S2 | S3 | S4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 221 | -0.007854 | -0.006508 | -0.004859 | -0.004513 | -0.000676 | -0.001404 | -0.013719 |
| 226 | -0.054903 | -0.064994 | -0.051075 | -0.038218 | -0.052389 | -0.025778 | -0.040712 |
| 227 | -0.062220 | -0.071595 | -0.054851 | -0.047342 | -0.060042 | -0.025778 | -0.197774 |
| 228 | -0.062220 | -0.071595 | -0.054851 | -0.047342 | -0.060042 | -0.025778 | -0.072924 |
| 237 | -0.058512 | -0.068558 | -0.053463 | -0.043209 | -0.056332 | -0.025778 | -0.197774 |

## Decision

- Any existing artifact promotable: `False`
- First rebuild focus: `ID221 clean feature-similarity root, ID226 clean target-wise convex backbone`
- Hold until clean backbone: `ID227, ID228, ID237, ID238, ID239, ID241`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run heavy rebuild now: `False`
- Next action: Use this matrix to extend the Kaggle upstream runner with report-first ID221/ID226 clean rebuild design. Do not create ID342 or replay A1/B1 artifacts until a clean upstream backbone exists and passes transfer gates.

