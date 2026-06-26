# Decision note — submission 338 id179_s1_seed45502_randomconv_w005_balanced

Created: `2026-06-18 01:14:14 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled public-anchor probe hypothesis: **yes, still unsubmitted**

## Rationale

This artifact keeps ID179 unchanged for six targets and replaces only `S1` with a random-conv sequence blend from seed `45502`, `512` kernels, model `lr_c0_1`, weight `0.05`. The same seed survived replay on ID178/179/190/191 anchors, but all evidence remains public-structure-local rather than true public validation.

## Artifact contract

- Submission CSV: `submissions/submission_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.csv`
- Notebook: `notebooks/history/submission_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.ipynb`
- Metadata: `reports/experiment_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861.csv`
- Base test: `artifacts/base_predictions/base_test_id_338_id179_s1_seed45502_randomconv_w005_balanced_cv0.537861.csv`
- Local OOF CV: `0.537861`
