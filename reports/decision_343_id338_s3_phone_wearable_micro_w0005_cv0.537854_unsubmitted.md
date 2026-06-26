# Decision note - submission 343 id338_s3_phone_wearable_micro_w0005

Created: `2026-06-23 23:55:40 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Reserve controlled S3 public probe hypothesis: **yes, still unsubmitted**

## Evidence

- Local OOF CV: `0.537854`
- Local CV delta vs ID338 base: `-0.00000693`
- S3 delta vs ID338 base: `-0.00004854`
- Robust score: `0.563573`
- Mean delta vs ID338: `0.00011334`
- Max delta vs ID338: `0.00365242`
- Failed-direction projection: `+0.00061630`
- Worst target: `S4` `0.602685`

## Rationale

The phone/wearable discordance source is domain-relevant for S3 because it contrasts phone rest state with wearable activation/light. The selected weight is only `0.005`, so the artifact preserves ID338 almost entirely while adding one independent S3 signal. This is a reserve candidate, not a submission recommendation by itself.

## Artifact contract

- Submission CSV: `submissions/submission_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.csv`
- Notebook: `notebooks/history/submission_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.ipynb`
- Metadata: `reports/experiment_343_id338_s3_phone_wearable_micro_w0005_cv0.537854_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854.csv`
- Base test: `artifacts/base_predictions/base_test_id_343_id338_s3_phone_wearable_micro_w0005_cv0.537854.csv`
