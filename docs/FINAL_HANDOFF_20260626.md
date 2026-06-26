# Final handoff - 2026-06-26

This document freezes the post-deadline operating state for the DACON ETRI Human
Understanding AI Paper Challenge workspace.

## Final submission selection

Select ID342 unless a later official/private result supersedes it.

| role | id | DACON submission id | file | local CV | public LB |
|---|---:|---:|---|---:|---:|
| Best confirmed public anchor | 342 | 1467937 | `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv` | 0.534000 | 0.5960832519 |

ID342 is an ID178/ID221 feature-similarity shrink at alpha `0.50`. It remains
the best confirmed public score in the local submission log.

## Final-day emergency slate

The final Q3 app-name lexicon anti-signal slate improved the local proxy but did
not transfer to the public leaderboard.

| id | DACON submission id | submitted at (KST) | local CV | public LB | public delta vs ID342 | status |
|---:|---:|---|---:|---:|---:|---|
| 346 | 1468652 | 2026-06-26 09:32:18 | 0.533659 | 0.5963391483 | +0.0002558964 | Worse than ID342 |
| 350 | 1468654 | 2026-06-26 09:39:56 | 0.533699 | 0.5963352712 | +0.0002520193 | Worse than ID342 |
| 347 | 1468655 | 2026-06-26 09:40:20 | 0.533813 | 0.5961947552 | +0.0001115033 | Best of the emergency slate, worse than ID342 |

ID348 and ID349 were generated but DACON rejected them with the daily submission
count limit. They should remain generated-unsubmitted artifacts.

## Interpretation

The final slate is a clear CV/LB gap example. A target-specific local gate found
a plausible Q3 anti-signal, but public feedback moved in the wrong direction.
Future analysis should treat local improvements near ID342 as hypotheses until
they explain the public ordering of known scored submissions.

Do not promote candidates solely because they beat ID342 locally. The useful
next research question is why the ID178/ID221 feature-similarity shrink
transferred better than later Q3/S3/S4 local signals.

## Reproducibility pointers

- Canonical submission log: `submissions/submission_log.csv`
- Final emergency result note: `reports/dacon_final_emergency_q3_app_lexicon_results_20260626.md`
- Final emergency generator: `scripts/promote_post_id342_q3_app_lexicon_candidate.py`
- Current workflow review: `docs/WORKFLOW_REVIEW.md`
- Public 0.55 research plan: `docs/PUBLIC_055_PLAN.md`

Each tracked submission should keep the expected mapping:

1. `submissions/submission_log.csv` row
2. `submissions/*.csv` artifact when intentionally retained
3. `notebooks/history/*.ipynb` matching notebook-history stub when present
4. `reports/decision_*.md` or equivalent decision note for promoted artifacts

## Security and Git state

- Hardcoded DACON credentials were removed from the current tree.
- Git history was rewritten and force-pushed so the known hardcoded DACON token
  assignment no longer appears in `scripts/submit_calibrated.py` history.
- The DACON token should still be treated as compromised and rotated in DACON.
- `.env`, `.venv`, raw data, processed data, and large generated artifacts stay
  ignored by Git.

## Post-competition follow-up

When official final/private scores become available:

1. Record the private score and selected-final status in `submissions/submission_log.csv`.
2. Add a short final-result report under `reports/`.
3. Keep ID342 as the public anchor unless official final results prove otherwise.
4. Use final/private feedback to evaluate whether the CV/LB gap was public-only
   or a broader validation mismatch.
