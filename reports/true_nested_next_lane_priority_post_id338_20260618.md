# True nested rebuild next-lane priority

Created: `2026-06-18 01:31:18 UTC+09:00`

This is report-only. It consolidates existing diagnostics and creates no CSV, notebook, base prediction, or DACON submission.

## Ranked lanes

| rank | lane | group | priority | signal | risk | cost | anchor penalty | read |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `S1_controlled_probe_holdout` | `probe_only` | -0.009 | 0.015 | 0.010 | 0.006 | 0.008 | ID337 is the cleanest low-movement controlled probe, but the signal is tiny. This lane can spend public budget only with explicit approval; it will not repair validation. |
| 2 | `A1_delta_capped_bridge_midrisk_rebuild` | `primary` | -0.015 | 0.082 | 0.045 | 0.026 | 0.026 | Delta-capped bridge has a smooth risk ladder: ID247/248 are safer than ID249. A nested bridge-choice rebuild can test whether the ladder survives without using full-OOF target choices. |
| 3 | `B1_pair_logit_direct_no_252_model_level_rebuild` | `primary` | -0.019 | 0.115 | 0.065 | 0.045 | 0.024 | Strong fold-internal pair/logit signal survives at the selector layer, but the dependency audit shows the source artifacts still trace into high/extreme full-OOF bridge/calibration chains. |
| 4 | `C0_lower_layer_id238_safe_calibration` | `deprioritized` | -0.048 | 0.010 | 0.020 | 0.020 | 0.018 | The lower ID238 layer has only tiny conservative temperature-shift salvage signal. This is useful for explaining why the ID242 chain is fragile, but not a high-upside next run. |
| 5 | `C2_upper_second_stage_calibration_id242_chain` | `blocked` | -0.090 | 0.160 | 0.145 | 0.045 | 0.060 | The ID242/252 chain has enormous local signal, but it depends on extreme non-nested calibration. Lower-layer diagnostics show ID242-like calibration on ID238 collapses under fold-internal refitting. |

## Next execution recommendation

Primary lane: `A1_delta_capped_bridge_midrisk_rebuild`.

The immediate execution lane is the mid-risk delta-capped bridge rebuild. The pair/logit selector signal remains the downstream map, but the B1 dependency audit shows its current sources still trace into high/extreme full-OOF bridge/calibration chains. Rebuild the bridge/source layer first, then replay pair/logit selection. Keep ID337 as a separate explicit-approval public probe candidate, not as validation repair.

## Lane details

### S1_controlled_probe_holdout

- Group: `probe_only`
- Next step: Keep ID337 as the exact safety-first probe candidate; do not submit automatically.
- Blocked by: `['Requires explicit user approval for exact CSV before DACON submission.']`
- Evidence report: `reports/recent_s1_probe_stress_review_20260618.json`

### A1_delta_capped_bridge_midrisk_rebuild

- Group: `primary`
- Next step: Replay capped bridge selection with caps around 0.15/0.20/0.30 inside outer folds, rejecting ID249-style aggressive movement unless it improves nested robust score and anchor geometry.
- Blocked by: `['Needs exact bridge family reconstruction from ID235/244-249 scripts.']`
- Evidence report: `reports/post_id249_targetwise_delta_capped_bridge_robustness_20260616.json`

### B1_pair_logit_direct_no_252_model_level_rebuild

- Group: `primary`
- Next step: Do not reuse source IDs 254/255/256/257/258/266 as clean sources. First rebuild the bridge/source layers below them inside outer folds, then replay target-wise pair/logit selection report-only.
- Blocked by: `['B1 source artifacts are not dependency-clean; see b1_pair_blend_source_dependency_audit.']`
- Evidence report: `reports/fold_internal_pair_blend_rebuild_post_id335_20260617.json`

### C0_lower_layer_id238_safe_calibration

- Group: `deprioritized`
- Next step: Do not prioritize unless the pair/bridge lanes fail or a stricter calibration baseline is needed.
- Blocked by: `[]`
- Evidence report: `reports/id238_conservative_calibration_post_id335_20260617.json`

### C2_upper_second_stage_calibration_id242_chain

- Group: `blocked`
- Next step: Keep blocked. Do not rebuild upper ID252 descendants until a lower-layer model-level source replaces the q240/isotonic full-OOF shortcut.
- Blocked by: `['ID242 lower-layer calibration is not validated.', 'ID287/ID329 public failures warn against local-frontier calibration.']`
- Evidence report: `reports/fold_internal_second_stage_calibration_post_id335_20260617.json`

