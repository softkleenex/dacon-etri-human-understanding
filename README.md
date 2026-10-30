# DACON ETRI Human Understanding AI Paper Challenge

This repository contains the codebase and workspace for the **ETRI Human Understanding AI Paper Challenge**. The competition focuses on predicting user sleep quality, fatigue, stress, and objective sleep metrics using multimodal lifelog data collected from smartphones and smartwatches.

## 🎯 Overview

- **Objective:** Predict 7 targets (Q1~Q3: Sleep Quality, Fatigue, Stress; S1~S4: Sleep Metrics) using multimodal lifelog data.
- **Task Type:** Multi-task Binary Classification (0 or 1 for each metric) optimized for **Average Log-Loss**.
- **Data Source:** Smartphone ('m' prefix) and Smartwatch ('w' prefix) sensor/usage logs over 700 days, with 450 days of labels.
- **Status:** Closed. The leaderboard ended 2026-06-26; the paper was accepted to an ICTC 2026 Special Track but was **not published** — author registration was not completed, and the EDAS record now reads `Withdrawn`. See [Final outcome](#-final-outcome-closed) below and [ICTC/DACON status](docs/ICTC_DACON_STATUS_20260916.md).

## 📌 Final outcome (closed)

| Stage | Result |
| --- | --- |
| Leaderboard | Best submission **ID342** — local CV `0.534000`, public `0.5960832519`, rank 220. The `0.55` target was never reached. |
| Paper | *Anchor-Constrained Feature-Similarity Ensembling for Multimodal Lifelog State Prediction* — **accepted** to the ICTC 2026 Special Track *iWETRIA* (paper #373, EDAS `1571325954`, IEEE PDF eXpress `2026332958`). Three reviews, split recommendations. Camera-ready and IEEE copyright were filed. |
| EDAS record today | **`Withdrawn`.** Acceptance is not retained when author registration is not completed. |
| Reproducibility check | **Not satisfied.** DACON required raw data → preprocessing → features → training → inference → `submission.csv` in one path. `reproduce_submission.py` reassembles *stored* OOF/test matrices; `replay_lineage.py` does start from raw data but drifts `+0.000845` log loss because the June scikit-learn/numpy versions were never pinned. No re-submission was sent by the 9/18 13:00 deadline. |
| Author registration | **Not completed** (ICTC requires one author at Regular rate). |
| Publication | Not in proceedings, not on IEEE Xplore. |
| DACON comprehensive evaluation | Excluded — it required leaderboard + paper acceptance + code/model report + author registration. **No award.** |

The operational lesson is recorded in full in the blog note linked at the bottom: reproducibility is decided
when experiment #1 is built, not when the submission is packaged. A 50-step ensemble lineage in which each
step consumes the previous step's *stored predictions* buys score and forfeits reproducibility.

## 🧠 Core Architecture

This project uses a multi-track modeling stack, but candidate promotion is gated by validation and public-gap risk rather than local score alone:
1. **Machine Learning Pipeline (`CatBoost` + `Optuna`)**: Uses GroupKFold cross-validation and Bayesian optimization (via Optuna) to find the best tree-based hyperparameters while avoiding data leakage.
2. **Deep Learning Pipeline (`PyTorch Lightning`)**: Implements a `MultiTaskLSTM` with a **Temporal Attention Layer** to capture longitudinal patterns across days, outputting to 7 task-specific heads.
3. **Advanced Feature Engineering**: Incorporates Cross-Sensor interactions (e.g., HR vs Activity), Ambient Sound parsing, App Usage Bag-of-Words, and Time-Series Rolling/Lag windows.

## 🗂️ Directory Structure

- `artifacts/`: `base_predictions/` OOF and test matrices that later experiments blend. Immutable once registered. Git-ignored.
- `conf/`: Hydra/YAML experiment configuration.
- `data/`: Raw and processed dataset storage. Git-ignored.
- `docs/`: Project documentation, strategy notes, and planning.
- `docs/competition/`: Official competition rules, metric descriptions, and the data dictionary.
- `models/`: Checkpoints for PyTorch models and CatBoost models.
- `notebooks/history/`: One stub notebook per experiment (349 files) — the first link in the artifact contract.
- `notebooks/kaggle_runner/`: Jupyter notebook runner pushed to Kaggle for cloud GPU training.
- `reports/`: Paired `<name>_<date>.json` + `.md` outputs of the report-only analysis scripts (1,662 files).
- `scripts/`: Executable automation (`pre_flight_check.py`, `ensemble.py`, `normalize_submission_log.py`, audit utilities).
- `src/`: Reusable source code (PyTorch DataLoaders, Feature Engineering, Lightning Trainers, Utilities).
- `submissions/`: Output CSV predictions, `submission_log.csv`, and JSON metadata files (349 log rows, ids 1-350).
- `tests/`: pytest suite, including the frozen ID341-era contract snapshots.

## 🚀 Setup & Installation

1. **Create Virtual Environment & Install Dependencies:**
   ```bash
   make install
   ```
   *Note: This automatically installs all pip dependencies and the local Dacon API package from backup.*

2. **Environment Variables:**
   Ensure `DACON_TOKEN` and `DACON_TEAMNAME` are exported in your shell for API submissions.

## ☁️ Kaggle Cloud Workflow & Automation

We leverage Kaggle's free compute for heavy model training and tuning. Cloud runs must remain staging-only until a human explicitly approves a DACON submission.

> The submission phase closed on 2026-06-26. The steps below are kept as a record of how the workflow ran; `make dacon-submit` has no remaining purpose and its approval gate still blocks it by default.

1. **Push Source Code to Kaggle Dataset:**
   Zips the `src` and `conf` folders and uploads them. **Pre-flight checks** run automatically to ensure valid configurations and paths.
   ```bash
   make kaggle-push-code
   ```

2. **Push and Run Notebook:**
   Pushes the execution notebook to Kaggle and starts the GPU training.
   ```bash
   make kaggle-push-notebook
   ```

3. **Pull, normalize, and submit deliberately:**
   Pull Kaggle outputs, refresh submission metadata, build a safe ensemble, then submit only after reviewing the generated file.
   ```bash
   make kaggle-wait-and-pull
   make normalize-submissions
   make ensemble
   make docs-audit
   make audit
   ALLOW_DACON_SUBMIT=true make dacon-submit SUB_PATH=submissions/submission_ensemble_latest.csv SUB_MEMO="reviewed safe ensemble"
   ```

Do not run auto-submission scripts or `make dacon-submit` without an explicit `SUB_PATH`, `SUB_MEMO`, user approval, and `ALLOW_DACON_SUBMIT=true`.

## 🤝 Ensembling & Validation

- **Safety Checks:** All predictions pass through a strict `clip_probabilities` function (in `src/utils/validation.py`) to prevent infinite Log-Loss penalties.
- **Auto Ensembling:**
  Easily combine the top-K performing models (extracted automatically by score in the filename).
  ```bash
  make ensemble
  ```

## 📖 Key Documentation
- [Evidence](docs/EVIDENCE.md) — primary documents for the paper's acceptance, and an explicit statement of what they do *not* prove
- [ICTC/DACON Final Status](docs/ICTC_DACON_STATUS_20260916.md) — paper review, reproducibility finding, and why there is no award
- [ID341 Snapshot Contract Tests](docs/ID341_SNAPSHOT_CONTRACT_TESTS_20261028.md) — why 27 tests are skipped rather than rewritten
- [Root Document Roles](docs/ROOT_DOCUMENTS.md)
- [Final Handoff](docs/FINAL_HANDOFF_20260626.md)
- [Workflow Review and Session Handoff](docs/WORKFLOW_REVIEW.md)
- [Master Plan](docs/MASTER_PLAN.md)
- [Public 0.55 Plan](docs/PUBLIC_055_PLAN.md)
- [Data Dictionary](docs/competition/DATA_DICTIONARY.md)
- [Submission Log](submissions/submission_log.csv)
- [Kaggle Environment & Troubleshooting](docs/infrastructure/cloud_env/guide.md)
- [Pipeline Automation Guide](docs/infrastructure/automation/guide.md)

<!-- BLOG-URL:START -->

## Blog

- Blog note: [ETRI 휴먼 이해 대회 회고](https://softkleenex.github.io/posts/dacon-etri-human-understanding/)

<!-- BLOG-URL:END -->
