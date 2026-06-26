# Validation-transfer gate — post remote parity

Created: `2026-06-18 03:10:41 UTC+09:00`

This is report-only. It creates no predictions, submission CSV, notebook artifact, submission-log row, or DACON submission.

## Public gap read

- Real public rows used: `23`
- Safe scored gap median: `0.016339`
- Recent ID178/ID329 gap floor: `0.062773`
- Public `0.58` plausibility gate: stress estimate `<= 0.590` and not quarantined

Recent anchor behavior matters more than old safe rows: ID178 and ID329 both landed around Public `0.601` despite very different local CVs. That is why the gate uses a stress estimate rather than local CV alone.

## Candidate gate

| id | local | proxy | local+recent-gap | stress | pass 0.58 gate | family | quarantined | file |
|---:|---:|---:|---:|---:|---|---|---|---|
| 337 | 0.538066 | 0.605022 | 0.600839 | 0.605022 | `False` | `sequence_temporal_family` | `False` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 307 | 0.537792 | 0.605472 | 0.600565 | 0.605472 | `False` | `failed_direction_combo_family` | `True` | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 304 | 0.532349 | 0.605919 | 0.595122 | 0.605919 | `False` | `failed_direction_combo_family` | `True` | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 179 | 0.538290 | 0.606255 | 0.601063 | 0.606255 | `False` | `sequence_temporal_family` | `False` | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 338 | 0.537861 | 0.606833 | 0.600634 | 0.606833 | `False` | `sequence_temporal_family` | `False` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 336 | 0.537663 | 0.607945 | 0.600436 | 0.607945 | `False` | `sequence_temporal_family` | `False` | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 339 | 0.536441 | 0.609547 | 0.599214 | 0.609547 | `False` | `other_family` | `False` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 295 | 0.548583 | 0.605688 | 0.611356 | 0.611356 | `False` | `failed_direction_combo_family` | `True` | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 297 | 0.550861 | 0.606165 | 0.613634 | 0.613634 | `False` | `failed_direction_combo_family` | `True` | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |

## Decision

- Any candidate passes Public `0.58` transfer gate: `False`
- Submit now: `False`
- Next action: Do not create ID340 from local CV alone. The next useful experiment must define a new source or validation split that lowers the transfer stress estimate, not merely blend toward lower local CV.

