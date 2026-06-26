# True nested rebuild blueprint — post ID335

Created: `2026-06-17 22:49:51 UTC+09:00`

This is report-only. It creates no CSV, notebook, submission-log row, base prediction, or DACON submission.

## Why this exists

The fold-internal failed-direction diagnostic preserved a meaningful local signal, but the selected source artifacts were still generated with full-OOF/non-nested procedures. This report identifies what must be rebuilt before that signal can become submission evidence.

## Fold-internal signal to explain

| split | best setting | overall | robust | selected source ids |
|---|---|---:|---:|---|
| `subject_hole_fold_internal` | `id310_like_cap120_lam160` | 0.522506 | 0.548380 | `[242, 243, 249, 252, 268, 276, 278]` |
| `loso_fold_internal` | `id310_like_cap120_lam160` | 0.523274 | 0.549054 | `[242, 249, 252, 268, 276, 278]` |

## Selected source artifact audit

| id | local CV | family | risk | immediate upstream ids | method |
|---:|---:|---|---|---|---|
| 242 | 0.357651 | `second_stage_full_oof_calibration` | `extreme_non_nested` | `[238]` | Second-stage full-OOF calibration on ID238 |
| 243 | 0.360420 | `second_stage_full_oof_calibration` | `extreme_non_nested` | `[238]` | Second-stage full-OOF calibration on ID238 |
| 249 | 0.374465 | `targetwise_delta_capped_bridge` | `extreme_non_nested` | `[235]` | Target-wise delta-capped prob/logit bridge public probe |
| 252 | 0.274420 | `second_stage_full_oof_calibration` | `extreme_non_nested` | `[249]` | Second-stage full-OOF calibration on ID249 |
| 268 | 0.448976 | `targetwise_pair_prob_logit_blend` | `high_non_nested` | `[178, 254, 255, 256, 257, 258, 266]` | Target-wise full-OOF pair probability/logit blend probe |
| 276 | 0.411217 | `targetwise_pair_prob_logit_blend` | `high_non_nested` | `[178, 252, 255, 258]` | Target-wise full-OOF pair probability/logit blend probe |
| 278 | 0.389120 | `targetwise_pair_prob_logit_blend` | `high_non_nested` | `[178, 252, 257, 258]` | Target-wise full-OOF pair probability/logit blend probe |

## Dependency edges

| upstream id | selected source id | upstream also selected? |
|---:|---:|---|
| 238 | 242 | no |
| 238 | 243 | no |
| 235 | 249 | no |
| 249 | 252 | yes |
| 178 | 268 | no |
| 254 | 268 | no |
| 255 | 268 | no |
| 256 | 268 | no |
| 257 | 268 | no |
| 258 | 268 | no |
| 266 | 268 | no |
| 178 | 276 | no |
| 252 | 276 | yes |
| 255 | 276 | no |
| 258 | 276 | no |
| 178 | 278 | no |
| 252 | 278 | yes |
| 257 | 278 | no |
| 258 | 278 | no |

## Required rebuild contract

1. Freeze outer subject-hole and LOSO splits before looking at scores.
2. For every outer fold, generate or select each candidate family using only the outer-training rows.
3. Fit target calibrators, bridge weights, projection penalties, and source choices inside that outer-training partition only.
4. Score only the untouched outer-validation rows, then aggregate target, subject, and date-slice robustness.
5. Do not create a DACON-facing artifact unless the nested result also explains ID178, ID287, and ID329 public behavior.

## Rebuild lanes

### A_delta_capped_bridge_selection_rebuild

Check whether the target-wise capped bridge survives when bridge choices are learned inside outer folds.

- Source IDs: `[249]`
- Allowed: `['choose per-target bridge families and caps inside the outer-training rows only', 'score only on untouched outer-validation rows', 'reject any cap that relies on full-OOF target loss or hidden-test movement alone']`
- Blocked: `["reusing ID249's full-OOF target-wise selection as proof", 'treating a capped test delta as a substitute for nested validation']`

### B_pair_blend_without_new_calibration

Check whether the pair/logit blend signal survives when target choices are learned inside outer folds.

- Source IDs: `[268, 276, 278]`
- Allowed: `['choose target-wise single/pair/logit blend configs inside the outer-training rows only', 'score only on untouched outer-validation rows', 'carry full-train test predictions only after the nested OOF score is reviewed']`
- Blocked: `['using full OOF target loss to choose a target config', 'promoting a submission CSV directly from existing ID268/276/278 artifacts']`

### C_second_stage_calibration_rebuild

Test whether the extreme calibration signal is real or only a full-OOF artifact.

- Source IDs: `[242, 243, 252]`
- Allowed: `['fit isotonic/quantile calibrators only on inner-training predictions', 'choose calibration family and quantile count inside inner folds', 'penalize extreme probabilities and per-target instability before scoring']`
- Blocked: `['fitting isotonic or quantile maps on the same OOF rows used for evaluation', 'treating local CV below 0.40 as leaderboard evidence without anchor explanation']`

### D_anchor_explanation_gate

Reject any rebuild that cannot explain the known public outcomes.

- Source IDs: `[178, 287, 329]`
- Required checks: `['ID287-like local-frontier movement must be penalized', 'ID329-like local gain must not be mistaken for public improvement', 'candidate must improve robust nested score without large movement away from ID178']`

## Decision

The selected source IDs preserve signal under fold-internal replay, but every selected source is non-nested and most are high/extreme-risk full-OOF calibration or target-wise blend artifacts. They justify a nested rebuild blueprint, not a submission.
