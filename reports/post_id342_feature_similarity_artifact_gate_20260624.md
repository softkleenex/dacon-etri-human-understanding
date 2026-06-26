# Post-ID342 feature-similarity artifact gate

Created: `2026-06-24 03:41:50 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Reviewed Rows

| rank | row | pass | local Δ342 | mean Δ | max Δ | proxy | gap 0.55 | max test Δ342 | failed checks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `id178_id221_alpha0.65_q123_subjective_rest050` | `False` | -0.000332 | -0.000554 | -0.000291 | 0.595752 | +0.045752 | 0.023876 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 2 | `id178_id221_alpha0.65_q1_s4_public_response_rest050` | `False` | -0.000375 | -0.000501 | -0.000375 | 0.595708 | +0.045708 | 0.028173 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 3 | `id178_id221_alpha0.65_q1_q3_s4_local_gain_rest050` | `False` | -0.000459 | -0.000444 | -0.000115 | 0.595624 | +0.045624 | 0.028173 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 4 | `id178_id221_alpha0.65_q2_rest050` | `False` | -0.000113 | -0.000321 | -0.000113 | 0.595970 | +0.045970 | 0.021297 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 5 | `id178_id221_alpha0.65_q1_rest050` | `False` | -0.000134 | -0.000291 | -0.000134 | 0.595949 | +0.045949 | 0.023497 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 6 | `id178_id221_alpha0.65_s4_rest050` | `False` | -0.000241 | -0.000211 | -0.000123 | 0.595843 | +0.045843 | 0.028173 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 7 | `id178_id221_alpha0.65_sleep_objective_rest050` | `False` | -0.000326 | -0.000158 | +0.000001 | 0.595757 | +0.045757 | 0.028173 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 8 | `id178_id221_alpha1.00_s2_rest050` | `False` | -0.000031 | -0.000141 | -0.000031 | 0.596052 | +0.046052 | 0.019567 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 9 | `id178_id221_alpha0.80_s2_rest050` | `False` | -0.000023 | -0.000090 | -0.000023 | 0.596060 | +0.046060 | 0.011740 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 10 | `id178_id221_alpha1.00_s3_rest050` | `False` | -0.000065 | -0.000077 | +0.000002 | 0.596018 | +0.046018 | 0.024298 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |
| 11 | `id178_id221_alpha0.80_s3_rest050` | `False` | -0.000047 | -0.000054 | -0.000005 | 0.596036 | +0.046036 | 0.014579 | `local_delta_material_enough,public_proxy_low_enough,public_055_gap_low_enough_for_remote,not_post_public_neighborhood_scan` |

## Decision

- Artifact gate pass count: `0`
- Best reviewed row: `id178_id221_alpha0.65_q123_subjective_rest050`
- Best projected public proxy: `0.595751629`
- Best gap to Public 0.55: `+0.045751629`
- Artifact review allowed: `False`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Reason: No reviewed row passed the strict artifact gate. The local gains are too small, the public proxy remains near ID342 and far from 0.55, and the scan is explicitly post-public ID342-neighborhood tuning.
