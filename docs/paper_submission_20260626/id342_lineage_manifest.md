# ID342 Lineage Manifest

Status: lineage manifest for the current confirmed public anchor. This document
does not create or modify predictions, submission CSVs, Kaggle runs, DACON
submissions, or submission-log rows.

## Canonical Submission-Log Row

- Row ID: `342`
- Date: `2026-06-23 16:44:59`
- Model description: `ID178/ID221 feature-similarity shrink a0.50`
- Local CV: `0.534000`
- Public score: `0.5960832519`
- Submitted: `True`
- Status: `cv_lb_gap_risk`
- Risk reason: `Absolute CV-LB gap is >= 0.03.`
- Notebook file:
  `notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb`
- Submission file:
  `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv`

## DACON Confirmation

- DACON submission ID: `1467937`
- Submitted at: `2026-06-23 23:30:37 KST`
- Confirmed score: `0.5960832519`
- Confirmation report:
  `reports/dacon_three_slot_submission_result_20260623.md`

## Artifact Contract

From `reports/decision_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.md`:

- Submission CSV:
  `submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv`
- Notebook:
  `notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb`
- Metadata:
  `reports/experiment_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.json`
- Base OOF:
  `artifacts/base_predictions/base_oof_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv`
- Base test:
  `artifacts/base_predictions/base_test_id_342_id178_id221_featuresim_shrink050_cv0.534000.csv`

## Interpretation

ID342 is a public-confirmed anchor, not a prize-range or Public `0.55` result.
Its local/Public gap is large:

```text
0.5960832519 - 0.534000 = 0.0620832519
```

For paper and model-description purposes, this row supports a conservative
claim: the ID178/ID221 feature-similarity shrink transferred better than the
other 2026-06-23 slate probes, but it did not solve the remaining Q3/S3/S4
representation gap.

## Invariant

Any future paper, model description, or handoff package must keep the following
one-to-one mapping intact:

```text
notebooks/history/submission_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.ipynb
  -> submissions/submission_id_342_id178_id221_featuresim_shrink050_cv0.534000_unsubmitted.csv
  -> submissions/submission_log.csv row id=342
  -> DACON submission id 1467937 / public 0.5960832519
```
