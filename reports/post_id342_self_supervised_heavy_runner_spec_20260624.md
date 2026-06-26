# Post-ID342 self-supervised heavy runner spec

Created: `2026-06-24 09:17:07 UTC+09:00`

This is report-only. It creates no runner notebook, raw cache, embedding, feature matrix, model, prediction, submission artifact, Kaggle run, or DACON submission.

## Decision

- Runner spec ready: `True`
- Proposed notebook dir: `notebooks/kaggle_self_supervised_subject_state_runner`
- Proposed Kaggle slug: `softkleenex/dacon-etri-self-supervised-subject-state-post-id342`
- Create runner files now: `False`
- Push Kaggle now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Required phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`
- Next action: Spec is ready. Do not create or push the runner until the exact heavy approval phrase is provided.

## Runner Files After Approval

| path | create now | after exact approval | purpose |
|---|---:|---:|---|
| `notebooks/kaggle_self_supervised_subject_state_runner/kernel-metadata.json` | `False` | `True` | Kaggle kernel metadata with GPU/TPU disabled unless separately justified. |
| `notebooks/kaggle_self_supervised_subject_state_runner/Self_Supervised_Subject_State_Post_ID342.ipynb` | `False` | `True` | No-submit notebook that stages data, fits label-free embeddings, and writes review reports only. |
| `scripts/run_post_id342_self_supervised_subject_state_no_submit.py` | `False` | `True` | Reusable runner script invoked by the Kaggle notebook; should not produce submission CSVs. |
| `reports/post_id342_self_supervised_remote_*` | `False` | `True` | Remote report-only outputs for embedding fit, fold-internal downstream smoke, and stop decisions. |

## Validation Gates

| gate | criterion | stop if fail |
|---|---|---:|
| `embedding_purity` | Embedding fit uses no target labels, no OOF predictions, and no public-score feedback. | `True` |
| `fold_internal_downstream` | Q3/S3 downstream smoke must beat ID342 on stable near-anchor lenses. | `True` |
| `failed_frontier_resistance` | The gain must not appear only when ID287/ID329-like failed frontier geometry is included. | `True` |
| `public_gap_budget` | Projected public proxy must be materially below the 0.596 band before artifact review. | `True` |
| `artifact_separation` | No submission CSV, base prediction artifact, notebook history artifact, or log row is created by this runner. | `True` |

## Approved Command Sequence

| step | command | allowed now | after exact approval | why |
|---:|---|---:|---:|---|
| 1 | `make post-id342-self-supervised-heavy-contract` | `True` | `True` | Refresh the contract and verify approval boundary. |
| 2 | `create dedicated runner files under notebooks/kaggle_self_supervised_subject_state_runner/` | `False` | `True` | The upstream rebuild runner is not a heavy embedding runner. |
| 3 | `ALLOW_KAGGLE_REMOTE=true make kaggle-push-self-supervised-subject-state-runner` | `False` | `True` | Push no-submit runner only after exact approval and local contracts pass. |
| 4 | `ALLOW_KAGGLE_REMOTE=true make kaggle-wait-and-pull-self-supervised-subject-state-output` | `False` | `True` | Pull reports only; do not create submissions. |
| 5 | `make post-id342-self-supervised-remote-review` | `False` | `True` | Review remote reports before any future artifact contract. |

## Research Sources

| source | url | runner implication |
|---|---|---|
| ETRI Human Understanding AI Paper Challenge 2024 dataset design | https://arxiv.org/html/2403.16509v1 | Use strict participant/date-aware folds and treat validation/test cohort shift as first-class risk. |
| MoCA: Multi-modal Cross-masked Autoencoder for Time Series in Digital Health | https://arxiv.org/html/2506.02260v3 | Prefer cross-modal masking over single-channel reconstruction so missing sensor streams become part of the objective. |
| CroSSL: Cross-modal Self-Supervised Learning for Time-series Signals | https://arxiv.org/pdf/2307.16847 | Keep modality-specific encoders and cross-modal latent masking separate from downstream target labels. |
| Personalization of Stress Mobile Sensing using Self-Supervised Learning | https://arxiv.org/abs/2308.02731 | Evaluate subject-state transfer and personalization explicitly for Q3 instead of relying on pooled local CV. |
| PixleepFlow | https://arxiv.org/html/2502.17469v2 | Represent synchronized lifelog channels as day/evening/sleep-transition state grids rather than isolated scalar proxies. |

## Forbidden Actions Now

```json
{
  "create_runner_notebook_now": false,
  "create_submission_artifact": false,
  "fit_embedding_now": false,
  "modify_submission_log": false,
  "push_kaggle_now": false,
  "read_full_raw_streams_now": false,
  "run_kaggle_now": false,
  "submit_now": false,
  "train_downstream_model_now": false,
  "write_feature_matrix_now": false,
  "write_oof_or_test_predictions_now": false
}
```
