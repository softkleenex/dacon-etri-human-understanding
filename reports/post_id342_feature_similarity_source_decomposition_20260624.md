# Post-ID342 feature-similarity source decomposition

Created: `2026-06-24 03:37:58 UTC+09:00`

This is report-only. It creates no submission artifacts, base predictions, Kaggle runs, or DACON submissions.

## Reference Losses

- ID178 local: `0.538638`
- ID221 local: `0.532990`
- ID342 local: `0.534000`
- ID342 reconstruction max error from ID178/ID221 alpha 0.50: `0.000000000000`

## Top Rows

| rank | config | targets | alpha | eligible | local Δ342 | wins | mean Δ | max Δ | max test Δ342 | proxy |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `id178_id221_alpha0.65_q123_subjective_rest050` | `Q1,Q2,Q3` | 0.65 | `True` | -0.000332 | 5 | -0.000554 | -0.000291 | 0.023876 | 0.595752 |
| 2 | `id178_id221_alpha0.65_q1_s4_public_response_rest050` | `Q1,S4` | 0.65 | `True` | -0.000375 | 5 | -0.000501 | -0.000375 | 0.028173 | 0.595708 |
| 3 | `id178_id221_alpha0.65_q1_q3_s4_local_gain_rest050` | `Q1,Q3,S4` | 0.65 | `True` | -0.000459 | 5 | -0.000444 | -0.000115 | 0.028173 | 0.595624 |
| 4 | `id178_id221_alpha0.65_q2_rest050` | `Q2` | 0.65 | `True` | -0.000113 | 5 | -0.000321 | -0.000113 | 0.021297 | 0.595970 |
| 5 | `id178_id221_alpha0.65_q1_rest050` | `Q1` | 0.65 | `True` | -0.000134 | 5 | -0.000291 | -0.000134 | 0.023497 | 0.595949 |
| 6 | `id178_id221_alpha0.65_s4_rest050` | `S4` | 0.65 | `True` | -0.000241 | 5 | -0.000211 | -0.000123 | 0.028173 | 0.595843 |
| 7 | `id178_id221_alpha0.65_sleep_objective_rest050` | `S1,S2,S3,S4` | 0.65 | `True` | -0.000326 | 4 | -0.000158 | +0.000001 | 0.028173 | 0.595757 |
| 8 | `id178_id221_alpha1.00_s2_rest050` | `S2` | 1.00 | `True` | -0.000031 | 5 | -0.000141 | -0.000031 | 0.019567 | 0.596052 |
| 9 | `id178_id221_alpha0.80_s2_rest050` | `S2` | 0.80 | `True` | -0.000023 | 5 | -0.000090 | -0.000023 | 0.011740 | 0.596060 |
| 10 | `id178_id221_alpha1.00_s3_rest050` | `S3` | 1.00 | `True` | -0.000065 | 4 | -0.000077 | +0.000002 | 0.024298 | 0.596018 |
| 11 | `id178_id221_alpha0.80_s3_rest050` | `S3` | 0.80 | `True` | -0.000047 | 5 | -0.000054 | -0.000005 | 0.014579 | 0.596036 |
| 12 | `id178_id221_alpha1.00_q123_subjective_rest050` | `Q1,Q2,Q3` | 1.00 | `False` | -0.000603 | 5 | -0.001266 | -0.000381 | 0.079586 | 0.595481 |

## Decision

- Eligible rows: `11`
- Best config: `id178_id221_alpha0.65_q123_subjective_rest050`
- Best changed targets: `['Q1', 'Q2', 'Q3']`
- Best local delta vs ID342: `-0.000332`
- Best mean/max primary delta vs ID342: `-0.000554` / `-0.000291`
- Projected public proxy from ID342: `0.595751629`
- Gap to Public 0.55 after proxy: `+0.045751629`
- Create submission artifact: `False`
- Submit now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- Next action: Keep this as report-only decomposition. If a row is locally eligible, it still needs a stricter artifact/joint-risk gate because it is an ID342-neighborhood post-public scan, not independent Public 0.55 evidence.
