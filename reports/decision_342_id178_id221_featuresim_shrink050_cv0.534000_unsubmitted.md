# Decision note — submission 342 id178_id221_featuresim_shrink050

Created: `2026-06-23 16:44:59 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled independent-source public probe hypothesis: **yes, still unsubmitted**

## Evidence

- Local OOF CV: `0.534000`
- Robust score: `0.558647`
- Mean delta vs ID178: `0.018463`
- Max delta vs ID178: `0.093911`
- Failed-direction projection: `-0.005082`
- Worst target: `S4` `0.591629`

## Rationale

ID221 brings feature-similarity signal from the strict-feature family, but the original unshrunk test geometry was too large for a direct probe. This candidate blends it with ID178 at alpha `0.50` across all targets. The intent is to test whether non-sequence behavioral similarity transfers publicly while keeping anchor movement restrained.

## Artifact contract

- Submission CSV: `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv`
- Notebook: `notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb`
- Metadata: `reports/experiment_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv`
- Base test: `artifacts/base_predictions/base_test_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv`
