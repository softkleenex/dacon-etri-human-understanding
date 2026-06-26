# DACON ETRI Human Understanding AI Paper Challenge

This repository contains the codebase and workspace for the **ETRI Human Understanding AI Paper Challenge**. The competition focuses on predicting user sleep quality, fatigue, stress, and objective sleep metrics using multimodal lifelog data collected from smartphones and smartwatches.

## 🎯 Overview

- **Objective:** Predict 7 targets (Q1~Q3: Sleep Quality, Fatigue, Stress; S1~S4: Sleep Metrics) using multimodal lifelog data.
- **Task Type:** Multi-task Binary Classification (0 or 1 for each metric) optimized for **Average Log-Loss**.
- **Data Source:** Smartphone ('m' prefix) and Smartwatch ('w' prefix) sensor/usage logs over 700 days, with 450 days of labels.
- **Current status:** Public `0.55` is not yet proven. See [Workflow Review and Session Handoff](docs/WORKFLOW_REVIEW.md) before running new experiments.

## 🧠 Core Architecture

This project uses a multi-track modeling stack, but candidate promotion is gated by validation and public-gap risk rather than local score alone:
1. **Machine Learning Pipeline (`CatBoost` + `Optuna`)**: Uses GroupKFold cross-validation and Bayesian optimization (via Optuna) to find the best tree-based hyperparameters while avoiding data leakage.
2. **Deep Learning Pipeline (`PyTorch Lightning`)**: Implements a `MultiTaskLSTM` with a **Temporal Attention Layer** to capture longitudinal patterns across days, outputting to 7 task-specific heads.
3. **Advanced Feature Engineering**: Incorporates Cross-Sensor interactions (e.g., HR vs Activity), Ambient Sound parsing, App Usage Bag-of-Words, and Time-Series Rolling/Lag windows.

## 🗂️ Directory Structure

- `data/`: Raw and processed dataset storage.
- `docs/`: Project documentation, strategy notes, and planning.
- `models/`: Checkpoints for PyTorch models and CatBoost models.
- `notebooks/kaggle_runner/`: Jupyter notebook runner pushed to Kaggle for cloud GPU training.
- `docs/competition/`: Official competition rules, metric descriptions, and the data dictionary.
- `scripts/`: Executable automation (`pre_flight_check.py`, `ensemble.py`, `normalize_submission_log.py`, audit utilities).
- `src/`: Reusable source code (PyTorch DataLoaders, Feature Engineering, Lightning Trainers, Utilities).
- `submissions/`: Output CSV predictions, `submission_log.csv`, and JSON metadata files.

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
- [Root Document Roles](docs/ROOT_DOCUMENTS.md)
- [Final Handoff](docs/FINAL_HANDOFF_20260626.md)
- [Workflow Review and Session Handoff](docs/WORKFLOW_REVIEW.md)
- [Master Plan](docs/MASTER_PLAN.md)
- [Public 0.55 Plan](docs/PUBLIC_055_PLAN.md)
- [Data Dictionary](docs/competition/DATA_DICTIONARY.md)
- [Submission Log](submissions/submission_log.csv)
- [Kaggle Environment & Troubleshooting](docs/infrastructure/cloud_env/guide.md)
- [Pipeline Automation Guide](docs/infrastructure/automation/guide.md)
