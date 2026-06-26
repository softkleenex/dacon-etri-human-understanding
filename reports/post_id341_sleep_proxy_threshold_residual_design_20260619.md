# Post-ID341 sleep-proxy threshold residual design

Created: `2026-06-23 18:17:39 UTC+09:00`

This is report-only. It creates no predictions, feature matrix, submission CSV, base prediction artifact, notebook/history artifact, submission-log row, Kaggle push/run, or DACON submission.

## Decision

- Design ready: `True`
- Metadata feasible: `True`
- Design backlog recorded: `True`
- Local no-approval score route exists: `False`
- Open design lane now: `False`
- Feature matrix allowed now: `False`
- Train model now: `False`
- Compute target associations now: `False`
- Public 0.58 reviewable: `False`
- Manual probe if user spends submission: `None`
- Manual probe family status: `consumed_after_id341_id340_id337_public_micro_probe_slate`
- Create ID342/new submission ID: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Keep this as design backlog only. The score-relevant route remains the exact-approved Kaggle no-submit upstream rebuild.

## Feature block schema coverage

| block | shared columns | schema ready | preview |
|---|---:|---:|---|
| `sleep_stage_duration` | 43 | `True` | `ch2025_mLight_light_max_max, ch2025_mLight_light_mean_max, ch2025_mLight_light_mean_mean, ch2025_wLight_light_max_max, ch2025_wLight_light_mean_max` |
| `sleep_efficiency_fragmentation` | 2 | `True` | `wHr_sleep_efficiency_proxy, wHr_sleep_fragmentation_index` |
| `sleep_hr_microstructure` | 26 | `True` | `ch2025_wHr_hr_rmssd_max, ch2025_wHr_hr_rmssd_mean, slp_hr_awake_ratio, slp_hr_deep_ratio, slp_hr_early_mean` |
| `sleep_pedo_light` | 38 | `True` | `ch2025_mLight_light_max_max, ch2025_mLight_light_mean_max, ch2025_mLight_light_mean_mean, ch2025_wLight_light_max_max, ch2025_wLight_light_mean_max` |

## Target design rows

| target | feasible | proxy blocks | risk |
|---|---:|---|---|
| `S2` | `True` | `sleep_efficiency_fragmentation,sleep_stage_duration` | processed sleep-efficiency proxies already overlap with prior sleep-boundary/circadian lanes |
| `S3` | `True` | `sleep_stage_duration,sleep_pedo_light` | SOL is not directly observed; threshold residuals would be proxy-only and public-transfer fragile |
| `S4` | `True` | `sleep_efficiency_fragmentation,sleep_hr_microstructure` | WASO-like fragmentation was already part of weak/closed sleep-family evidence |
| `Q1` | `True` | `sleep_stage_duration,sleep_hr_microstructure,sleep_pedo_light` | subjective target may not transfer from objective sleep proxy without subject-normalized validation |

## Context checks

- safe_wait_ready: `True`
- next_route_triage_ready: `True`
- sleep_backlog_feasible_but_not_open: `True`
- q3s3_local_lanes_closed: `True`
- sleep_boundary_not_promotable: `True`
- circadian_lane_closed: `True`
- paper_family_lane_closed: `True`
- public_gap_blocks_cv_only: `True`

