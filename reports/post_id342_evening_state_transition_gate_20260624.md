# Post-ID342 evening state-transition gate

Created: `2026-06-24 05:11:03 UTC+09:00`

This is report-only. It creates no predictions, submission artifacts, Kaggle runs, or DACON submissions.

## Domain read

- Q3 is pre-sleep stress and S3 is sleep-onset-latency adherence. This gate uses pre-sleep screen/app/HR activity and settling into darkness/charging/stillness, with previous-day label-free deviations and fold-internal subject baselines.
- Web-researched sources:
  - [ETRI Lifelog Dataset 2024](https://arxiv.org/html/2508.03698v1): ETRI defines Q3 as pre-sleep stress and S3 as sleep-onset-latency guideline adherence; the baseline section reports that task-specific time-zone features beat all-feature bundles.
  - [National Sleep Foundation - What Is Sleep Quality?](https://www.thensf.org/what-is-sleep-quality/): NSF frames sleep quality around latency, awakenings, WASO, and efficiency, with adult latency screening at falling asleep within 30 minutes.
  - [National Sleep Foundation - Consistent Sleep Schedules](https://www.thensf.org/sleep-schedules-sleep-timing-guideline/): NSF's sleep-timing guidance motivates regularity and previous-day deviation features rather than only same-day absolute sensor totals.
  - [Sleep Foundation - Electronics and Sleep](https://www.sleepfoundation.org/how-sleep-works/how-electronics-affect-sleep): Pre-bed device light and alerting effects motivate screen/light/usage settling features for S3 and Q3, while keeping the gate strictly empirical.

## Feature contract

- Rows: `700` (`450` train, `250` test)
- Numeric feature columns used: `520`
- Mean missing rate: `0.14731593406593405`

| family | feature count | examples |
|---|---:|---|
| `all_transition` | 520 | `est_pedo_presleep_step_sum_minus_evening_delta_prev1`, `est_pedo_presleep_distance_sum_minus_evening_delta_prev1`, `est_pedo_presleep_step_sum_minus_evening`, `est_pedo_presleep_step_sum_minus_evening_prev1`, `est_pedo_presleep_step_sum_minus_evening_dev_past3` |
| `q3_arousal_usage` | 456 | `est_pedo_presleep_step_sum_minus_evening_delta_prev1`, `est_pedo_presleep_distance_sum_minus_evening_delta_prev1`, `est_pedo_presleep_step_sum_minus_evening`, `est_pedo_presleep_step_sum_minus_evening_prev1`, `est_pedo_presleep_step_sum_minus_evening_dev_past3` |
| `s3_settling_sol` | 323 | `est_pedo_presleep_step_sum_minus_evening_delta_prev1`, `est_pedo_presleep_distance_sum_minus_evening_delta_prev1`, `est_pedo_presleep_step_sum_minus_evening`, `est_pedo_presleep_step_sum_minus_evening_prev1`, `est_pedo_presleep_step_sum_minus_evening_dev_past3` |

## Target scan

| target | selected | eligible | best config | wins | mean delta | max delta |
|---|---|---:|---|---:|---:|---:|
| `Q3` | `False` | 0 | `Q3_id342_evening_state_q3_arousal_usage_lr_c0_03_w0.005` | 2 | 0.000015535 | 0.000215616 |
| `S3` | `False` | 0 | `S3_id342_evening_state_s3_settling_sol_lr_c0_30_w0.005` | 2 | 0.000063167 | 0.000386532 |

## Decision

- Selected targets: `[]`
- Estimated all-target delta vs ID342: `0.000000000`
- Projected Public proxy: `0.596083252`
- Gap to Public 0.55: `0.046083252`
- Final gate pass: `False`
- Kaggle now: `False`
- DACON submit now: `False`
- Next action: If this gate fails, close lightweight evening state-transition features as another ID342-aware local lane. If both Q3 and S3 pass, use Kaggle no-submit only for parity review before any artifact.
