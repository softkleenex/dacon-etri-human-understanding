# Post-remote-parity next lane decision

Created: `2026-06-18 03:09:30 UTC+09:00`

Inputs:
- Rerank: `reports/post_remote_parity_public_anchor_rerank_post_id339_20260618.json`
- Remote parity: `reports/remote_candidate_parity_post_id339_summary_20260618.json`

## Decision

- Submit now: **no**
- Public `0.58`-plausible candidate exists: **no**
- Mac/M4 environment suspicion for ID337/ID339: **cleared enough for candidate parity**, but this is not leaderboard evidence.
- If the user explicitly forces one controlled public probe: **ID337**.
- Exact file: `submissions/submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv`

## Why

Post-parity rerank still puts ID337 first among low-movement reviewable candidates, but its conservative proxy is around `0.605`, not `0.58`. ID339 is better locally but has a worse proxy and broader target movement. The failed-direction/orthogonal candidates near the top stay quarantined because ID287 and ID329 already showed that this family can look locally attractive while failing publicly.

## Candidate read

| id | local | proxy | family | max Δ178 | proj failed | status | file |
|---:|---:|---:|---|---:|---:|---|---|
| 337 | 0.538066 | 0.605022 | `sequence_temporal_family` | 0.037585 | -0.0006 | controlled-probe only | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 339 | 0.536441 | 0.609547 | `other_family` | 0.097051 | -0.0120 | proxy is not Public-0.58 plausible; multi-target movement and worse proxy than ID337 | `submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |
| 338 | 0.537861 | 0.606833 | `sequence_temporal_family` | 0.073989 | +0.0005 | proxy is not Public-0.58 plausible; higher movement than ID337 for similar S1 signal | `submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 336 | 0.537663 | 0.607945 | `sequence_temporal_family` | 0.112756 | -0.0017 | proxy is not Public-0.58 plausible; higher movement than ID337 for similar S1 signal | `submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv` |
| 179 | 0.538290 | 0.606255 | `sequence_temporal_family` | 0.051824 | +0.0011 | proxy is not Public-0.58 plausible | `submission_id_179_id178_tempered_id177_random_conv_anchor_blend_cv0.538290_unsubmitted.csv` |
| 307 | 0.537792 | 0.605472 | `failed_direction_combo_family` | 0.026555 | -0.0170 | failed-direction/orthogonal family is quarantined after ID287/ID329 public failure; proxy is not Public-0.58 plausible | `submission_id_307_post303_orthogonal_cap030_neg010_probe_cv0.537792_unsubmitted.csv` |
| 295 | 0.548583 | 0.605688 | `failed_direction_combo_family` | 0.014923 | -0.0500 | failed-direction/orthogonal family is quarantined after ID287/ID329 public failure; proxy is not Public-0.58 plausible | `submission_id_295_post287_anchor_antivector_prob_m005_cv0.548583_unsubmitted.csv` |
| 304 | 0.532349 | 0.605919 | `failed_direction_combo_family` | 0.029765 | -0.0000 | failed-direction/orthogonal family is quarantined after ID287/ID329 public failure; proxy is not Public-0.58 plausible | `submission_id_304_post303_orthogonal_cap030_proj000_safe_cv0.532349_unsubmitted.csv` |
| 297 | 0.550861 | 0.606165 | `failed_direction_combo_family` | 0.018164 | -0.0614 | failed-direction/orthogonal family is quarantined after ID287/ID329 public failure; proxy is not Public-0.58 plausible | `submission_id_297_post287_anchor_antivector_logit_m005_cv0.550861_unsubmitted.csv` |

## Next lane

Run a report-only validation-transfer diagnostic before creating ID340: compare current low-movement sequence candidates against scored public anchors by family and target movement; only create a new artifact if it improves a predeclared cross-family transfer gate, not just local CV.

No DACON submission was made.
