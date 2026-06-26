# Post-ID341 Public-0.60 wall root-cause diagnosis

Created: `2026-06-21 16:54:45 UTC+09:00`

This is report-only. It creates no predictions, feature matrix, submission artifact, Kaggle run, or DACON submission.

## Decision

- Wall explained by current evidence: `True`
- Local-CV-only path closed: `True`
- Same-family micro-probe path closed: `True`
- Q3/S3 independent signal required: `True`
- Primary score route: `explicit_approved_clean_upstream_id221_to_id226`
- Requires exact Kaggle approval: `True`
- Submit now: `False`
- Create submission artifact: `False`
- Next action: Do not spend more DACON submissions or local micro-overlays. The next useful score work is either the exact-approved no-submit upstream rebuild, or a materially new Q3/S3 independent source that passes this wall diagnosis and the transfer gates before artifact review.

## Recent public band

- Best recent public ID: `337`
- Best recent public score: `0.600620628622`
- Local span across ID178/329/337/340: `0.02192`
- Public span across ID178/329/337/340: `0.001170229078`
- All recent micro probes near 0.60: `True`

| id | local CV | public | public-local gap | file |
|---:|---:|---:|---:|---|
| 178 | 0.538638 | 0.6014113009 | 0.0627733009 | `submission_id_178_id173_fine_highc_random_conv_sequence_overlay_cv0.538638_unsubmitted.csv` |
| 329 | 0.516718 | 0.6017908577 | 0.0850728577 | `submission_id_329_post328_combo4_cap080_budget050_practical_cv0.516718_unsubmitted.csv` |
| 337 | 0.538066 | 0.600620628622 | 0.062554628622 | `submission_id_337_id178_s1_seed45502_randomconv_w005_safety_cv0.538066_unsubmitted.csv` |
| 340 | 0.538426 | 0.601379175253 | 0.062953175253 | `submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv` |

## Causes

| cause | active | evidence |
|---|:---:|---|
| `validation_surface_mismatch` | `True` | Recent local CV changes around ID178/337/340 span far more locally than publicly, while ID329/287 show lower local CV can fail to transfer. |
| `q3_s3_uncovered` | `True` | Remaining-route inventory keeps Q3/S3 uncovered and local no-approval score routes closed. |
| `q2_only_lead_insufficient` | `True` | The env/social/routine local smoke selected Q2 only; Q3/S3 stay empty and Q2 proxy remains above 0.60. |
| `leaderboard_probe_family_consumed` | `True` | ID341/340/337 micro-probes have already been spent and stayed near the 0.60 public band. |

## Target-gap evidence

```json
{
  "q3_s3_still_uncovered": true,
  "q2_only_signal_found": true,
  "q2_best_proxy": 0.6019107525231998,
  "q2_not_public_058": true,
  "local_no_approval_score_route_exists": false,
  "only_score_relevant_open_route": "explicit_approved_clean_upstream_id221_to_id226"
}
```

