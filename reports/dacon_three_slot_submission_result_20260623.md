# DACON three-slot submission result - 2026-06-23

Created: `2026-06-23 23:31:16 UTC+09:00`
Updated: `2026-06-23 23:36:37 UTC+09:00`
Corrected: `2026-06-24 00:00 KST authenticated submission history`

## Decision

- User submission-spend line: `23:30 KST`
- Kaggle v3 status before spend: `KernelWorkerStatus.COMPLETE`
- Kaggle v3 artifact promotion: `False`
- Policy report: `reports/dacon_three_slot_slate_20260623.json`
- Submission log updated: `True`
- Confirmed best public score after batch: `0.5960832519`
- Confirmed best submission after batch: ID `342`, DACON submission `1467937`
- Confirmed team rank after batch: `220`
- Score source: authenticated DACON submission history provided by operator on `2026-06-24`

## Attempts

| slot | id | status | HTTP | public | response | exact path |
|---:|---:|---|---:|---:|---|---|
| 1 | 338 | `scored_cv_lb_gap_risk` | 200 | `0.5992145177` | `sub_id=1467936; isSubmitted=True; detail=Success` | `submissions/submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv` |
| 2 | 342 | `scored_team_best_cv_lb_gap_risk` | 200 | `0.5960832519` | `sub_id=1467937; isSubmitted=True; detail=Success` | `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv` |
| 3 | 339 | `scored_cv_lb_gap_risk` | 200 | `0.5985187787` | `sub_id=1467938; isSubmitted=True; detail=Success` | `submissions/submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv` |

## Interpretation

The batch broke the recent Public `0.600` wall through ID342, not ID338. The previous confirmed best was `0.6006206286219425`, so the improvement is `-0.004537376722`. This corrects the earlier unauthenticated leaderboard inference that attributed the team-best row to ID338.
