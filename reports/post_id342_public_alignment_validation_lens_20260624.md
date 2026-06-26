# Post-ID342 public-alignment validation lens

Created: `2026-06-24 04:53:35 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Candidate Scores

| id | full OOF | public | public-full gap | source | note |
|---:|---:|---:|---:|---|---|
| 178 | 0.538638 | 0.601411301 | +0.062774 | `recorded_control_public_anchor` | Former public anchor. |
| 287 | 0.378934 | 0.669878732 | +0.290945 | `recorded_failed_frontier` | Severe local-frontier public failure. |
| 329 | 0.516718 | 0.601790858 | +0.085073 | `recorded_guarded_probe` | Practical guarded probe that did not beat ID178. |
| 337 | 0.538066 | 0.600620629 | +0.062554 | `recorded_micro_probe` | Low-movement S1 random-conv safety probe. |
| 338 | 0.537861 | 0.599214518 | +0.061354 | `dacon_three_slot_submission_result` |  |
| 339 | 0.536441 | 0.598518779 | +0.062078 | `dacon_three_slot_submission_result` |  |
| 340 | 0.538426 | 0.601379175 | +0.062953 | `recorded_micro_probe` | Q2 sequence-summary safety probe. |
| 342 | 0.534000 | 0.596083252 | +0.062083 | `dacon_three_slot_submission_result` |  |

## Lens Alignment

| group | best lens | spearman | pearson | pairwise | rank MAE | order |
|---|---|---:|---:|---:|---:|---|
| all_scored | `full_oof` | -0.167 | -0.991 | 0.536 | 3.000 | `[287, 329, 342, 339, 338, 337, 340, 178]` |
| exclude_id287 | `full_oof` | 0.250 | -0.226 | 0.714 | 1.714 | `[329, 342, 339, 338, 337, 340, 178]` |
| exclude_failed_frontiers | `full_oof` | 1.000 | 0.965 | 1.000 | 0.000 | `[342, 339, 338, 337, 340, 178]` |
| recent_20260623_slate | `full_oof` | 1.000 | 0.987 | 1.000 | 0.000 | `[342, 339, 338]` |

## Recent Slate

- Public order: `[342, 339, 338]`
- Exact-match local lenses: `['full_oof', 'primary_mean', 'primary_worst', 'subject_hole_5', 'subject_timeseries_5', 'actual_test_band_holdout', 'test_boundary_neighbor2_holdout', 'test_boundary_neighbor5_holdout', 'loso_10']`

## Target Lens Leaders

| rank | target | lens type | spearman | pairwise | order |
|---:|---|---|---:|---:|---|
| 1 | `Q1` | `full` | 0.941 | 1.000 | `[342, 339, 338, 178, 337, 340]` |
| 2 | `Q1` | `primary_mean` | 0.941 | 1.000 | `[342, 339, 338, 178, 337, 340]` |
| 3 | `Q3` | `full` | 0.941 | 1.000 | `[342, 339, 338, 178, 337, 340]` |
| 4 | `S1` | `full` | 0.883 | 0.923 | `[342, 338, 337, 339, 178, 340]` |
| 5 | `S2` | `full` | 0.676 | 0.889 | `[342, 338, 178, 337, 339, 340]` |

## Decision

- Best all-scored lens: `full_oof`
- Best lens excluding ID287: `full_oof`
- Best stable lens excluding ID287/ID329: `full_oof`
- Trusted lens candidates: `['full_oof', 'subject_hole_5', 'loso_10']`
- ID342 gap to Public 0.55: `+0.046083252`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Gate policy: Do not trust full local CV alone. Require any future candidate to beat ID342 on the best stable public-alignment lens, preserve the recent slate order where relevant, and explicitly report behavior with ID287 and ID329 excluded so the gate is not driven by failed-frontier artifacts.
- Reason: The diagnostic is calibration-only. It explains which local lenses are less misleading, but it does not create a new independent signal or reduce ID342's public gap to 0.55.
