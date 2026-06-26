# Post-ID341 public transfer audit

Created: `2026-06-21 16:48:16 UTC+09:00`

This is report-only. It creates no predictions, submission CSV, notebook/history artifact, submission-log row, or DACON submission.

## Public evidence read

- Plausible public rows used: `25`
- Recent anchor gap floor, IDs 178/329: `0.062773`
- Local/public correlation on plausible rows: `0.212010`
- Current no-submit control: ID341, proxy `0.603830`

Plausible public evidence still says local CV is a weak selector. ID178 and ID329 both landed near Public 0.601 despite different local CVs; ID287 proves full-OOF calibration can collapse.

## Recent/control candidates

| id | local | public | proxy | Δ178 mean/max | Δ341 mean/max | proj failed | tier | file |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 178 | 0.538638 | 0.601411 | 0.601450 | 0.000000/0.000000 | 0.000124/0.002025 | +0.0000 | `known_public_anchor` | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` |
| 287 | 0.378934 | 0.669879 | 0.845570 | 0.115353/0.298460 | 0.115367/0.298460 | +1.0000 | `known_public_gap_or_collapse` | `submission_id_287_post285_base282_fixedtemp_balanced_cv0.378934_unsubmitted.csv` |
| 329 | 0.516718 | 0.601791 | 0.644729 | 0.019372/0.079192 | 0.019359/0.079192 | +0.0499 | `known_public_gap_or_collapse` | `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv` |
| 334 | 0.540347 |  | 0.609113 | 0.010290/0.119120 | 0.010413/0.119120 | +0.0039 | `recent_sequence_probe_unsubmitted` | `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv` |
| 335 | 0.539690 |  | 0.607864 | 0.007392/0.111949 | 0.007516/0.111949 | +0.0032 | `recent_sequence_probe_unsubmitted` | `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv` |
| 336 | 0.537663 |  | 0.606237 | 0.004008/0.112756 | 0.004132/0.112756 | -0.0017 | `recent_sequence_probe_unsubmitted` | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 337 | 0.538066 | 0.600621 | 0.603030 | 0.001336/0.037585 | 0.001460/0.037585 | -0.0006 | `known_public_anchor` | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 338 | 0.537861 |  | 0.605302 | 0.004660/0.073989 | 0.004783/0.073989 | +0.0005 | `recent_sequence_probe_unsubmitted` | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 339 | 0.536441 |  | 0.608259 | 0.012602/0.097051 | 0.012482/0.095026 | -0.0120 | `recent_sequence_probe_unsubmitted` | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 340 | 0.538426 | 0.601379 | 0.601780 | 0.000494/0.008102 | 0.000371/0.006076 | -0.0003 | `known_public_anchor` | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |
| 341 | 0.538584 |  | 0.603830 | 0.000124/0.002025 | 0.000000/0.000000 | -0.0001 | `current_control_no_submit` | `submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv` |

## Best numeric proxy rows after quarantine

| rank | id | local | proxy | family | tier | Δ178 max | Δ341 max | file |
|---:|---:|---:|---:|---|---|---:|---:|---|
| 1 | 179 | 0.538290 | 0.604506 | `sequence_anchor_family` | `diagnostic_only` | 0.051824 | 0.051824 | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 2 | 338 | 0.537861 | 0.605302 | `sequence_anchor_family` | `recent_sequence_probe_unsubmitted` | 0.073989 | 0.073989 | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 3 | 336 | 0.537663 | 0.606237 | `sequence_anchor_family` | `recent_sequence_probe_unsubmitted` | 0.112756 | 0.112756 | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 4 | 190 | 0.537956 | 0.607471 | `sequence_anchor_family` | `diagnostic_only` | 0.127403 | 0.127403 | `submission_id_190_id173_remote_s1_random_conv_refresh_overlay_cv0.537956_unsubmitted.csv` |
| 5 | 191 | 0.538009 | 0.607699 | `other_family` | `diagnostic_only` | 0.121839 | 0.121839 | `submission_id_191_id190_tempered_id177_remote_s1_anchor_blend_cv0.538009_unsubmitted.csv` |
| 6 | 335 | 0.539690 | 0.607864 | `sequence_anchor_family` | `recent_sequence_probe_unsubmitted` | 0.111949 | 0.111949 | `submission_id_335_id178_alpha_s4half_stressguard_cv0.539690_unsubmitted.csv` |
| 7 | 339 | 0.536441 | 0.608259 | `sequence_anchor_family` | `recent_sequence_probe_unsubmitted` | 0.097051 | 0.095026 | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 8 | 334 | 0.540347 | 0.609113 | `sequence_anchor_family` | `recent_sequence_probe_unsubmitted` | 0.119120 | 0.119120 | `submission_id_334_id178_rollback_s1_s4_sequence_stressguard_cv0.540347_unsubmitted.csv` |
| 9 | 311 | 0.524195 | 0.610009 | `other_family` | `diagnostic_only` | 0.091284 | 0.091284 | `submission_id_311_post307_aggbudget_cap100_lam160_zeroproj_cv0.524195_unsubmitted.csv` |
| 10 | 193 | 0.536429 | 0.610774 | `sequence_anchor_family` | `diagnostic_only` | 0.121839 | 0.121839 | `submission_id_193_id191_sequence_summary_embedding_overlay_cv0.536429_unsubmitted.csv` |
| 11 | 177 | 0.540607 | 0.610905 | `sequence_anchor_family` | `diagnostic_only` | 0.115165 | 0.115165 | `submission_id_177_id173_multiseed_random_conv_sequence_overlay_v4_cv0.540607_unsubmitted.csv` |
| 12 | 198 | 0.536380 | 0.611048 | `other_family` | `diagnostic_only` | 0.121839 | 0.121839 | `submission_id_198_id193_s3_smartphone_light_compact_overlay_cv0.536380_unsubmitted.csv` |
| 13 | 200 | 0.536190 | 0.612148 | `sequence_anchor_family` | `diagnostic_only` | 0.136414 | 0.136414 | `submission_id_200_id198_sax_paa_sequence_overlay_cv0.536190_unsubmitted.csv` |
| 14 | 203 | 0.535980 | 0.612928 | `sequence_anchor_family` | `diagnostic_only` | 0.136414 | 0.136414 | `submission_id_203_id200_s4_recurrence_gaf_sequence_overlay_cv0.535980_unsubmitted.csv` |
| 15 | 310 | 0.521885 | 0.613046 | `other_family` | `diagnostic_only` | 0.114878 | 0.114878 | `submission_id_310_post307_aggbudget_cap120_lam160_zeroproj_cv0.521885_unsubmitted.csv` |
| 16 | 206 | 0.535764 | 0.613568 | `sequence_anchor_family` | `diagnostic_only` | 0.136414 | 0.136414 | `submission_id_206_id203_wavelet_spectral_sequence_overlay_cv0.535764_unsubmitted.csv` |
| 17 | 309 | 0.523249 | 0.613595 | `other_family` | `diagnostic_only` | 0.049147 | 0.049749 | `submission_id_309_post307_aggbudget_cap050_lam120_balanced_cv0.523249_unsubmitted.csv` |
| 18 | 176 | 0.542270 | 0.614114 | `sequence_anchor_family` | `diagnostic_only` | 0.142515 | 0.142515 | `submission_id_176_id173_multiseed_random_conv_sequence_overlay_v3_cv0.542270_unsubmitted.csv` |
| 19 | 180 | 0.529579 | 0.614554 | `calibration_family` | `diagnostic_only` | 0.072340 | 0.072340 | `submission_id_180_id179_tempered_id172_extreme_anchor_probe_cv0.529579_unsubmitted.csv` |
| 20 | 314 | 0.520647 | 0.614834 | `other_family` | `diagnostic_only` | 0.078300 | 0.078300 | `submission_id_314_post313_secondpass_cap080_lam120_practical_cv0.520647_unsubmitted.csv` |

## Decision

- Any candidate passes Public 0.58 transfer gate: `False`
- Submit now: `False`
- Create submission artifact now: `False`
- Push Kaggle now: `False`
- Next action: Do not continue local OOF recombination or same-family shrinking. The next score-relevant work is the report-only Kaggle upstream source rebuild, or a genuinely new source family evaluated through this transfer gate.

