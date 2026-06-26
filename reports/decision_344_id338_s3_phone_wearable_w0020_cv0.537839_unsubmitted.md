# Decision note - submission 344 id338_s3_phone_wearable_w0020

Created: `2026-06-24 00:07:41 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Reserve controlled S3 public probe hypothesis: **yes, still unsubmitted**

## Evidence

- Local OOF CV: `0.537839`
- Local CV delta vs ID338 base: `-0.00002210`
- S3 delta vs ID338 base: `-0.00015472`
- Robust score: `0.563496`
- Mean delta vs ID338: `0.00045335`
- Max delta vs ID338: `0.01460969`
- Failed-direction projection: `+0.00081723`
- Worst target: `S4` `0.602685`

## Rationale

The phone/wearable discordance source is domain-relevant for S3 because it contrasts phone rest state with wearable activation/light. The selected weight is `0.02`, so the artifact preserves ID338 while adding one independent S3 signal. This is a reserve candidate, not a submission recommendation by itself.

## Artifact contract

- Submission CSV: `submissions/submission_id_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.csv`
- Notebook: `notebooks/history/submission_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.ipynb`
- Metadata: `reports/experiment_344_id338_s3_phone_wearable_w0020_cv0.537839_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_344_id338_s3_phone_wearable_w0020_cv0.537839.csv`
- Base test: `artifacts/base_predictions/base_test_id_344_id338_s3_phone_wearable_w0020_cv0.537839.csv`
