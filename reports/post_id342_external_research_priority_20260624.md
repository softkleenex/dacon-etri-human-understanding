# Post-ID342 external research priority

Created: `2026-06-24 10:16:26 UTC+09:00`

This report updates research priority only. It performs no raw payload read, embedding fit, feature creation, model training, Kaggle action, submission-log edit, or DACON submission.

## Decision

- Research priority ready: `True`
- Top priority route: `aim_cross_modal_masked_subject_state`
- Heavy compute allowed now: `False`
- Create runner files now: `False`
- Run Kaggle now: `False`
- DACON submit now: `False`
- Required phrase: `Self-supervised subject-state heavy compute no-submit 실행 승인`
- Next action: Keep execution closed. If exact heavy approval is later provided, prioritize a missingness-aware cross-modal masked subject-state runner over image-only, alpha, temperature, or donor routes.

## Score State

```json
{
  "best_public_anchor_id": 342,
  "best_public_score": 0.5960832519,
  "closed_route_count": 11,
  "gap_to_public_055": 0.046083251899999955,
  "route_count": 12,
  "target_public_score": 0.55
}
```

## Recommended Heavy Route

```json
{
  "dacon_candidate_threshold_public_proxy": 0.55,
  "name": "missingness_aware_cross_modal_masked_subject_state",
  "priority_components": [
    "AIM-style inherited plus artificial missingness masking",
    "CroSSL-style modality-specific encoders and latent masking",
    "TimeMAE-style semantic window tokens",
    "Q3/S3 fold-internal frozen-embedding downstream heads"
  ],
  "review_threshold_public_proxy": 0.58
}
```

## Research Implications

| finding | evidence | action |
|---|---|---|
| The large public gap is representation-sized, not calibration-sized. | ID342 is still +0.0460832519 from Public 0.55 after 11/12 local routes closed. | Do not spend DACON on alpha, temperature, target-swap, or image-only variants. |
| Missingness must be modeled, not erased. | ETRI/DACON expose multi-device lifelog streams with varied sampling and natural gaps. | Use inherited missingness tokens plus artificial masks as a core SSL objective. |
| Q3 and S3 need joint subject-state context. | Prior Q3/S3-only handcrafted, sequence-summary, SOL, prior, and PCA gates failed. | Pretrain a shared day/evening/sleep-transition encoder and test Q3/S3 with frozen embeddings. |
| Image-style layouts are useful as priors but not sufficient. | PixleepFlow supports synchronized sensor images, but local ID342 image/CNN smokes did not pass. | Use channel layout for masking and explanation only inside stronger cross-modal SSL. |

## Method Candidates

| priority | route | fit | why | risk |
|---:|---|---|---|---|
| 1 | `aim_cross_modal_masked_subject_state` | highest | Matches the 12-stream incomplete lifelog setting and directly targets latent subject state for Q3/S3. | Needs heavy no-submit Kaggle execution and strict fold-internal downstream review. |
| 2 | `time_mae_semantic_window_subject_state` | high | The dataset is label-scarce and long-horizon; semantic sub-series windows are more relevant than point loss. | Can collapse into another generic sequence summary unless cross-modal missingness is retained. |
| 3 | `crossl_latent_modality_dropout` | high | The raw streams have natural missing modalities and different sampling frequencies. | May be too small to close a 0.046 public gap unless paired with AIM-style inherited masking. |
| 4 | `tnc_subject_temporal_neighborhood` | medium | Q3/S3 are state-transition targets, so local temporal smoothness is plausible. | High leakage risk if neighborhoods are not fold- and date-safe; not a standalone first route. |
| 5 | `ts2vec_hierarchical_context` | medium | Hierarchical timestamp/subsequence embeddings are broadly useful for irregular lifelog windows. | Less directly matched to missing modality and cross-device behavior than AIM/CroSSL. |
| 6 | `pixleepflow_image_layout_only` | low_as_standalone | Image conversion is domain-relevant, but local sequence-image and tiny CNN gates did not open Q3/S3. | Do not launch another image-only or random-conv route from current evidence. |

## External Sources

| source | type | URL | implication |
|---|---|---|---|
| Understanding Human Daily Experience Through Continuous Sensing: ETRI Lifelog Dataset 2024 | dataset_paper | https://arxiv.org/html/2508.03698v1 | The remaining route should model day-state and sleep-transition context jointly. |
| The 4th ETRI Human Understanding AI Paper Challenge data page | competition_page | https://dacon.io/en/competitions/official/236468/data | A future runner should preserve all 12 modalities and their missingness patterns. |
| PixleepFlow: A Pixel-Based Lifelog Framework for Predicting Sleep Quality and Stress Level | etri_competition_method | https://arxiv.org/html/2502.17469v2 | Image layout is useful as a representation prior, but local ID342 smokes say it is not enough alone. |
| Multi-modal Time Series Analysis: A Tutorial and Survey | survey | https://arxiv.org/html/2503.13709v1 | The next design should use cross-modal alignment/fusion with explicit robustness checks. |
| CroSSL: Cross-modal Self-Supervised Learning for Time-series through Latent Masking | method_paper | https://arxiv.org/abs/2307.16847 | This is a strong template for phone/watch/sleep-context modality dropout and fusion. |
| LSM-2: Learning from Incomplete Wearable Sensor Data | method_paper | https://arxiv.org/html/2506.05321v1 | Missingness should be a first-class token, not a preprocessing nuisance to fill away. |
| TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders | method_paper | https://arxiv.org/html/2303.00320v4 | Use 30-minute to multi-hour windows as semantic units instead of minute-level reconstruction only. |
| Unsupervised Representation Learning for Time Series with Temporal Neighborhood Coding | method_paper | https://arxiv.org/abs/2106.00750 | Subject-day neighborhoods are plausible, but validation must prevent date/subject leakage. |
| TS2Vec: Towards Universal Representation of Time Series | method_paper | https://arxiv.org/abs/2106.10466 | Use as a secondary objective only after missingness-aware cross-modal masking is fixed. |

## Prerequisites

```json
{
  "domain_contract_ready": true,
  "goal_not_achieved": true,
  "heavy_execution_closed_now": true,
  "id342_anchor_current": true,
  "input_reports_parse": true,
  "only_heavy_route_remaining": true,
  "raw_payload_not_read": true,
  "raw_schema_contract_ready": true,
  "success_thresholds_ready": true
}
```

## Forbidden Actions Now

```json
{
  "create_predictions": false,
  "create_submission_artifact": false,
  "fit_embedding": false,
  "modify_submission_log": false,
  "pull_kaggle_output": false,
  "push_kaggle": false,
  "read_raw_parquet_payloads": false,
  "run_kaggle": false,
  "submit_dacon": false,
  "train_model": false,
  "write_feature_matrix": false,
  "write_token_cache": false
}
```
