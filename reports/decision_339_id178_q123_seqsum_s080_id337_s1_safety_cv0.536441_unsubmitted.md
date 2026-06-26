# Decision note — submission 339 id178_q123_seqsum_s080_id337_s1_safety

Created: `2026-06-18 02:17:30 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled public-anchor probe hypothesis: **yes, still unsubmitted**

## Evidence

- Local OOF CV: `0.536441`
- Robust score: `0.562034`
- Max Δ vs ID178: `0.097051`
- Failed-direction projection: `-0.012001`
- Worst target: `S4` `0.602700`

## Rationale

Q1/Q2/Q3 use the ID178 sequence-summary overlay with weights scaled by `0.80`. S1 uses the existing ID337 safety source. S2/S3/S4 remain ID178. This is the first full-contract promotion of the post-A1 sequence-summary cap-sensitivity lead, but it must not be treated as a leaderboard improvement until the user explicitly approves and DACON scores it.

## Artifact contract

- Submission CSV: `submissions/submission_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.csv`
- Notebook: `notebooks/history/submission_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.ipynb`
- Metadata: `reports/experiment_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441.csv`
- Base test: `artifacts/base_predictions/base_test_id_339_id178_q123_seqsum_s080_id337_s1_safety_cv0.536441.csv`
