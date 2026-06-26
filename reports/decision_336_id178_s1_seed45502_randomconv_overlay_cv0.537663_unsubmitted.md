# Decision note — submission 336 id178_s1_seed45502_randomconv_overlay

Created: `2026-06-18 00:48:03 UTC+09:00`

## Decision

- DACON submission: **no**
- Public 0.58 plausible: **no**
- Controlled public-anchor probe hypothesis: **yes, still unsubmitted**

## Rationale

This artifact keeps ID178 unchanged for six targets and replaces only `S1` with a random-conv sequence blend from seed `45502`, `512` kernels, model `lr_c0_1`, weight `0.15`. The same seed survived replay on ID178/179/190/191 anchors, but all evidence remains public-structure-local rather than true public validation.

## Artifact contract

- Submission CSV: `submissions/submission_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.csv`
- Notebook: `notebooks/history/submission_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.ipynb`
- Metadata: `reports/experiment_336_id178_s1_seed45502_randomconv_overlay_cv0.537663_unsubmitted.json`
- Base OOF: `artifacts/base_predictions/base_oof_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663.csv`
- Base test: `artifacts/base_predictions/base_test_id_336_id178_s1_seed45502_randomconv_overlay_cv0.537663.csv`
- Local OOF CV: `0.537663`
