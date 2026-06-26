# DACON three-slot slate - 2026-06-23

Created: `2026-06-23 22:59:03 UTC+09:00`

This is an exact-path/memo policy report. It validates selected artifacts and writes no submission.

## Decision

- Slate ready: `True`
- Submit slots allowed by this policy: `3`
- Public 0.58 plausible candidate exists: `False`
- Requires exact DACON CSV approval: `True`
- Kaggle network now: `False`
- Next action: Submit the three files one by one only after exact SUB_PATH/SUB_MEMO approval. Stop immediately on day_max_count, API failure, or unexpected public deterioration.

## Public Context

- Recent public floor: `0.6006206286219425`
- ID178 public: `0.6014113009`
- ID337 public: `0.6006206286219425`
- ID340 public: `0.601379175253277`
- ID329 public: `0.6017908577`
- ID287 public collapse: `0.6698787318`

## Exact Slate

| slot | id | local | proxy | joint | max d178 | role | exact path | exact memo |
|---:|---:|---:|---:|---|---:|---|---|---|
| 1 | 338 | 0.537861 | 0.605302 | `low_joint_warning` | 0.073989 | `conservative_anchor_variant` | `submissions/submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` | `20260623 slate 1/3 id338 id179 s1 randomconv safety probe` |
| 2 | 342 | 0.534000 | 0.609942 | `low_joint_warning` | 0.093911 | `independent_feature_similarity_probe` | `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv` | `20260623 slate 2/3 id342 id178 id221 featuresim shrink050 independent probe` |
| 3 | 339 | 0.536441 | 0.608259 | `low_joint_warning` | 0.097051 | `informative_sequence_summary_probe` | `submissions/submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` | `20260623 slate 3/3 id339 q123 seqsum id337 s1 information probe` |

## Domain Rationale

### Slot 1 - ID338

- Rationale: Tests whether the ID179 tempered anchor plus the same low-movement S1 source improves on ID337.
- Domain read: S1 is a sleep-quality/sleep-state target where the random-conv temporal sequence source was the only recent remotely reproduced signal. This slot keeps all non-S1 targets anchored.
- Transfer/joint blocks: `transfer_proxy_above_public_058_gate, transfer_tier:recent_sequence_probe_unsubmitted`
- Ready for exact approval: `True`

### Slot 2 - ID342

- Rationale: Uses the unconsumed ID221 strict-feature/feature-similarity family, shrunk 50% toward ID178 to keep max movement below 0.10.
- Domain read: Feature-similarity is a non-sequence behavioral/source-family probe: it tests whether subject/day covariate similarity transfers where recent sequence micro-probes compressed near public 0.60.
- Transfer/joint blocks: `transfer_proxy_above_public_058_gate, transfer_tier:diagnostic_only`
- Ready for exact approval: `True`

### Slot 3 - ID339

- Rationale: More informative but higher-movement Q1/Q2/Q3 sequence-summary probe with ID337 S1; useful if the first two slots do not move public below the 0.60 band.
- Domain read: Q targets are subjective condition/routine responses. The sequence-summary source summarizes daily sensor trajectories, so this slot probes whether broader temporal shape helps those subjective labels.
- Transfer/joint blocks: `transfer_proxy_above_public_058_gate, transfer_tier:recent_sequence_probe_unsubmitted`
- Ready for exact approval: `True`

## Artifact Checks

| slot | id | csv | notebook | metadata | base oof | base test | errors |
|---:|---:|---|---|---|---|---|---|
| 1 | 338 | `submissions/submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` | `notebooks/history/submission_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.ipynb` | `reports/experiment_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.json` | `artifacts/base_predictions/base_oof_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861.csv` | `artifacts/base_predictions/base_test_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861.csv` | `none` |
| 2 | 342 | `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv` | `notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb` | `reports/experiment_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.json` | `artifacts/base_predictions/base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv` | `artifacts/base_predictions/base_test_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv` | `none` |
| 3 | 339 | `submissions/submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` | `notebooks/history/submission_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.ipynb` | `reports/experiment_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.json` | `artifacts/base_predictions/base_oof_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441.csv` | `artifacts/base_predictions/base_test_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441.csv` | `none` |

