# Model Description Draft

Status: draft model-description material for the 5th ETRI Human Understanding
AI Paper Challenge. This document is not a DACON submission and does not create
or modify any prediction file, submission CSV, Kaggle run, or submission-log
row.

## Confirmed Public Anchor

- Submission ID: `342`
- DACON submission ID: `1467937`
- Public score: `0.5960832519`
- Local CV: `0.534000`
- Submitted at: `2026-06-23 23:30:37 KST`
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

## Model Summary

The confirmed model is an anchor-controlled probability blend:

```text
ID342 = 0.50 * ID221 + 0.50 * ID178
```

ID178 is the stable public-anchor lineage. ID221 contributes an independent
feature-similarity signal from the strict-feature family. The blend shrinks the
ID221 movement back toward ID178 to reduce the risk of large public-frontier
movement. All seven targets are blended under the same alpha.

The model should be described as a conservative ensemble and risk-control
workflow, not as a single end-to-end neural architecture.

## Target Definitions

- `Q1`: subjective sleep quality relative to participant average.
- `Q2`: subjective physical fatigue before sleep; `1` means lower fatigue.
- `Q3`: subjective stress before sleep; `1` means lower stress.
- `S1`: total sleep time guideline adherence.
- `S2`: sleep efficiency guideline adherence.
- `S3`: sleep onset latency guideline adherence.
- `S4`: wake after sleep onset guideline adherence.

## Validation Policy

Candidate promotion required more than full local CV improvement. The review
stack used:

- full out-of-fold multi-target log loss,
- subject-hole split checks,
- leave-one-subject-out style stress checks,
- test-boundary neighbor holdouts,
- target-specific loss deltas,
- public-gap proxy estimates,
- failed-frontier correlation checks against IDs such as `287` and `329`,
- submission-log and artifact lineage checks.

This policy was necessary because ID287 and ID329 showed that low local CV can
be misleading for the public leaderboard.

## Negative Evidence Summary

The following branches did not justify a new artifact or DACON submission:

- KMA ASOS historical weather: valid source, tiny unstable S4 movement only.
- AirKorea 2024 Daejeon archive: complete historical coverage, no review pass.
- Historical ETRI survey/sleep priors: Q2 coupling only, no direct Q3/S3/S4
  route.
- Missingness-aware and self-supervised subject-state routes: broad coverage,
  zero stable downstream passes.
- MIS-LSTM-style block sequence proxy: selected only a small S4 signal.
- Registered material-source fusion: no local promotion gate pass.
- Stable-envelope and target-reallocation optimizers: improved projections but
  remained far from Public `0.55`.

## Reproduction Boundary

The package must not include private API keys, DACON/Kaggle credentials, or raw
external weather caches. External weather and AirKorea caches stay under ignored
`data/external/weather/`.

Recommended local checks before handing over any result package:

```bash
make paper-submission-package-audit
make post-id342-deadline-operating-policy
make docs-audit
git diff --check
```

Do not run `make dacon-submit` from this model-description draft.

Lineage details are fixed in
`docs/paper_submission_20260626/id342_lineage_manifest.md`.
