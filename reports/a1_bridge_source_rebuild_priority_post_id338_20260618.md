# A1 bridge source rebuild priority — post ID338

Created: `2026-06-18 01:47:29 UTC+09:00`

This is report-only. It creates no CSV, notebook, base prediction, submission-log row, or DACON submission.

## Split summary

| split | all-extreme best | q240-free best | q240-free gap | all q240-choice rate | qfree sources |
|---|---:|---:|---:|---:|---|
| `subject_hole_fold_internal` | 0.395423 / robust 0.418444 | 0.414272 / robust 0.432545 | +0.018849 / robust +0.014101 | 71.4% | `238:1, 239:5, 241:25` |
| `loso_fold_internal` | 0.395415 / robust 0.418415 | 0.414440 / robust 0.432730 | +0.019026 / robust +0.014315 | 71.4% | `238:3, 239:10, 241:50` |

## q240-free target map

| target | predominant q240-free choice | target loss gap vs all-extreme | read |
|---|---|---:|---|
| `Q1` | `bridge:241:logit:a0.475` (5/5 folds) | +0.035583 | q240_free_recalibration |
| `Q2` | `bridge:241:logit:a0.463` (5/5 folds) | +0.019743 | q240_free_recalibration |
| `Q3` | `bridge:241:prob:a0.537` (5/5 folds) | +0.030041 | q240_free_recalibration |
| `S1` | `bridge:241:logit:a0.400` (5/5 folds) | +0.016667 | q240_free_recalibration |
| `S2` | `bridge:241:logit:a0.425` (4/5 folds) | +0.029912 | q240_free_recalibration |
| `S3` | `base:235` (4/5 folds) | +0.000000 | bridge_base |
| `S4` | `bridge:239:prob:a0.550` (5/5 folds) | +0.000000 | q240_free_recalibration |

## Recommendations

### 1. `A1_q240_free_source_layer_rebuild`

- Source IDs: `[238, 239, 241]`
- Why: This is the cleanest immediate A1 lane: it gives up about 0.018849 subject-hole and 0.019026 LOSO local score versus the q240-dominated all-extreme row, but avoids direct ID242/243 dependence.
- Allowed next action: Rebuild source alternatives 238/239/241 inside frozen outer folds, then replay the cap0.30 target-wise bridge selector report-only.
- Blocked action: Do not create a DACON CSV from the existing q240-free OOF artifacts.

### 2. `A1_s4_s3_micro_ablation`

- Source IDs: `[238, 239]`
- Why: S4 consistently selects ID239 and S3 mostly stays on base with occasional ID238. This is lower-upside than the q240-free full target map, but cheaper and useful as a sanity check before a larger source rebuild.
- Allowed next action: Run a report-only nested S3/S4 ablation if compute budget is tight.
- Blocked action: Do not tune S3/S4 by public-anchor movement alone.

### 3. `A1_q240_branch_hold`

- Source IDs: `[242, 243]`
- Why: The all-extreme row is strongest, with q240 choices in roughly 71.4% of subject-hole fold-target choices and 71.4% of LOSO choices. That is signal, but it is also exactly the blocked full-OOF q240 calibration chain.
- Allowed next action: Keep as an explanatory stress target only; revisit after lower calibration/source generation is rebuilt without full-OOF shortcuts.
- Blocked action: Do not submit or descend into ID242/252 artifacts from this evidence.

## Decision

Primary next lane: `A1_q240_free_source_layer_rebuild`.

The A1 signal survives, but the strongest all-extreme selector is dominated by the blocked q240 chain. The safest useful next work is a q240-free source-layer rebuild, not a leaderboard submission.

No submission artifact or DACON submission was created.
