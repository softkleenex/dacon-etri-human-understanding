# Decision note — submission 340 id178_q2_seqsum_lr_c03_w001_safety

Created: `2026-06-18 03:41:08 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled low-movement public probe hypothesis: **yes, still unsubmitted**

## Evidence

- Local Subject-Hole CV: `0.538426`
- Robust score: `0.563952`
- Mean Δ vs ID178: `0.000494`
- Max Δ vs ID178: `0.008102`
- Failed-direction projection: `-0.000339`
- Worst target: `S4` `0.602700`

## Rationale

Only `Q2` is blended with sequence-summary `lr_c0_3` at weight `0.01`. All other targets remain ID178. This is the safest target-subset ID340 construction lead from the report-only geometry scan, but it is not a Public 0.58 claim and needs explicit approval before any DACON submission.

## Artifact contract

- Submission CSV: `submissions/submission_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.csv`
- Notebook: `notebooks/history/submission_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.ipynb`
- Metadata: `reports/experiment_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426.csv`
- Base test: `artifacts/base_predictions/base_test_id_340_id178_q2_seqsum_lr_c03_w001_safety_cv0.538426.csv`
