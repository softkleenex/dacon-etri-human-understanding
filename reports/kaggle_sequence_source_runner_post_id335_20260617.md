# Kaggle sequence-source runner post-ID335

Dedicated report-only Kaggle runner for the post-ID335 S1 random-conv sequence-source lane.

- Runner directory: `notebooks/kaggle_sequence_source_runner`
- Notebook: `notebooks/kaggle_sequence_source_runner/Sequence_Source_Post_ID335_Runner.ipynb`
- Metadata: `notebooks/kaggle_sequence_source_runner/kernel-metadata.json`
- Kaggle slug: `softkleenex/dacon-etri-sequence-source-post-id335`
- Push target: `make kaggle-push-sequence-source-runner`
- Pull target: `make kaggle-pull-sequence-source-output`
- Current state: pushed to Kaggle, completed, pulled, and summarized.
- Output dir: `artifacts/kaggle_outputs/sequence_source_post_id335`
- Summary: `reports/kaggle_sequence_source_remote_summary_post_id335_20260617.md`

## Jobs

| base id | kernels | seed | output |
|---:|---:|---:|---|
| 173 | 512 | 45501 | `reports/id173_random_conv_sequence_post_id335_s1_nk512_seed45501_20260617.json` |
| 173 | 512 | 45502 | `reports/id173_random_conv_sequence_post_id335_s1_nk512_seed45502_20260617.json` |
| 173 | 768 | 45601 | `reports/id173_random_conv_sequence_post_id335_s1_nk768_seed45601_20260617.json` |
| 173 | 768 | 45602 | `reports/id173_random_conv_sequence_post_id335_s1_nk768_seed45602_20260617.json` |
| 173 | 1024 | 45701 | `reports/id173_random_conv_sequence_post_id335_s1_nk1024_seed45701_20260617.json` |
| 173 | 1024 | 45702 | `reports/id173_random_conv_sequence_post_id335_s1_nk1024_seed45702_20260617.json` |

## Result

All six report-only diagnostics completed on Kaggle. The best overall diagnostic was `nk512_seed45502`, with ID173 base mean `0.545253`, targetwise best mean `0.542888`, and improvement `+0.002365` versus the ID173 base under the public-structure diagnostic lens.

The strongest target-specific signal was S1: `blend_lr_c0_1_0.2` improved S1 by `+0.010855` versus ID173. Q1 and S4 also showed smaller signals, while S3 stayed at base.

## Guardrails

This runner created no DACON submission, no submission-log row, no promoted notebook, and no base OOF/test artifact. The result is useful source-generation evidence, but it is not a Public `0.58` claim and does not authorize artifact promotion by itself. A promoted candidate would require a separate unique submission CSV, notebook/history artifact, submission-log row, report, and base OOF/test pair.
