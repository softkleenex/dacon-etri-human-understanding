# Decision note — submission 341 id178_q2_seqsum_lr_c03_w00025_micro_safety

Created: `2026-06-18 03:52:48 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled low-movement public probe hypothesis: **yes, still unsubmitted**

## Evidence

- Local Subject-Hole CV: `0.538584`
- Robust score: `0.564131`
- Mean Δ vs ID178: `0.000124`
- Max Δ vs ID178: `0.002025`
- Failed-direction projection: `-0.000085`
- Worst target: `S4` `0.602700`

## Rationale

Only `Q2` is blended with sequence-summary `lr_c0_3` at weight `0.0025`. All other targets remain ID178. This is the safest target-subset ID340 construction lead from the report-only geometry scan, but it is not a Public 0.58 claim and needs explicit approval before any DACON submission.

## Artifact contract

- Submission CSV: `submissions/submission_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.csv`
- Notebook: `notebooks/history/submission_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.ipynb`
- Metadata: `reports/experiment_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584.csv`
- Base test: `artifacts/base_predictions/base_test_id_341_id178_q2_seqsum_lr_c03_w00025_micro_safety_cv0.538584.csv`
