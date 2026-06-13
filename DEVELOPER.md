# 🦊 DEVELOPER.md - Developer Guidelines

## 🎯 Project Overview
- **Competition:** DACON ETRI Human Understanding AI Paper Challenge (softkleenex team)
- **Objective:** Predict 7 targets (Q1~Q3, S1~S4) using multi-task binary classification.
- **Metric:** Average Log-Loss (Target: < 0.56)

---

## 🛠️ Build & Environment Commands
- **Install Dependencies:** `make install` (Installs virtual env and local Dacon API package)
- **Ruff Linter:** `make lint`
- **Ruff Formatter:** `make format`
- **Run Tests:** `make test`
- **Local Notebook Generation:** `.venv/bin/python notebooks/generate_notebooks.py` (Generates main.ipynb inside notebooks/kaggle_runner/)

---

## ☁️ Kaggle Operations (Strict Sequential Constraint)
> [!IMPORTANT]
> **RULE:** Run only ONE Kaggle kernel at a time. Sequence: Push -> Wait -> Pull -> Next.
- **Push packaged code to Kaggle Dataset:** `make kaggle-push-code`
- **Push and launch notebook kernel:** `make kaggle-push-only KAGGLE_SLUG="softkleenex/dacon-etri-cat-tuning-v10"`
- **Monitor and pull output in background:** `.venv/bin/python -u scripts/kaggle_manager.py --slug <slug> --out_dir <dir>`

---

## 🚀 Kaggle TPU Deployment Guardrails & Rules
> [!CAUTION]
> **Kaggle Sandbox Environment Guardrails (Never Repeat Mismatches!):**
> 1. **Duality Slug Alignment Rule:**
>    - The ID in `dataset-metadata.json` (`softkleenex/dacon-etri-human-base`) MUST exactly match the item under `dataset_sources` in `kernel-metadata.json`.
>    - Any mismatch causes Kaggle to reject dataset mounting, leading to fatal `ModuleNotFoundError`.
> 2. **Environment-Agnostic Dynamic Mapping Rule:**
>    - Kaggle sandboxes can mount datasets *either* as a zipped `.source.zip` or as an uncompressed direct directory tree.
>    - Every deployed Notebook's **Cell 1** must utilize the **Dynamic Mapper** to automatically scan `/kaggle/input` for the folder containing `src` and `conf`, then copy them to `./` via `shutil.copytree(..., dirs_exist_ok=True)`. Never assume a static zip extraction.
> 3. **GPU Quota TPU Bypass Rule:**
>    - If Kaggle GPU weekly quota is exhausted, push kernels using the TPU accelerator by ensuring `enable_gpu: false` and `enable_tpu: true` in `kernel-metadata.json`. (Automated via `make kaggle-push-only` sed replacements).

---

## 🩺 Strict ML/DL Code Guidelines (The Leak-Free Vaccine)
1. **No Look-Ahead Bias (100% Leak-Free):**
   - Chronological sorting must occur *before* feature calculations.
   - Forward filling (`ffill()`) must be done chronologically per subject, never globally or leaking future indexes.
2. **Feature Alignment Symmetry:**
   - Always merge metadata features (`base_oof`, `user_embeddings`, `hf_chronos_embeddings`) symmetrically into both `df_train` and `df_test`.
   - Never let test set merge with missing features (defaulting to 0/mean), which collapses tree predictions.
3. **Validation Strategy:**
   - Use `make_subject_timeseries_folds` to split chronologically *within* each subject, preventing identity leakages.
4. **Target-Specific Architecture & Chain Order:**
    - **Classifier Chain Order Rule:** Always follow easiest-to-hardest sequential biological causality order: `S3` -> `S2` -> `S1` -> `S4` -> `Q1` -> `Q2` -> `Q3`. Never put highly subjective targets like `Q1` first as they infect downstream clean targets with cumulative errors.
    - **Causal Classifier Chain Propagation Constraint:** Objective sleep target predictions (`S1~S4`) are allowed to propagate downstream inside the chain to both S-targets and subjective Q-targets. However, subjective prediction priors (`Q1~Q3`) must be completely blocked from propagating to downstream targets to eliminate noisy cumulative error cascades.
5. **Model Normalization & Stacking Calibration:**
   - Constrain tree depth limits to `2 ~ 4` for tiny data ($N=451$) to eliminate over-parameterization.
   - **Personalized Subject-wise Calibration:** Adjust model predictions with historical subject-wise target means using a blend factor `alpha` regularized strictly within `[0.5, 1.0]` to prevent overfitting on subjects with sparse data points.
   - **Multi-Target Joint Adjustment:** Implement optimized S4 joint post-processing by shifting S4 predictions proportionally to the product of deep (`S3`) and light (`S2`) sleep probabilities to respect physiological bounds.

---

## 🔒 Absolute Path & Execution Safety
1. **NbConvert Execution CWD Isolation:**
   - When running the notebook standalone via `nbconvert`, the execution context changes to `notebooks/core/`.
   - All relative file reads/writes (e.g. `submissions/`, `data/`) will fail or create orphan directories.
   - **Rule:** Always prepend output paths with the absolute `PROJECT_ROOT` path to guarantee directory resolution is cwd-agnostic.
2. **Kaggle Directory Copy Tree Robustness:**
   - When unzipping `source.zip` on Kaggle, the root parent folder `data` may not be yielded as an entry in `dirs` during `os.walk`.
   - **Rule:** Use direct existence checks `src_data = src_root / "data"; if src_data.exists():` instead of checking `if "data" in dirs:` to ensure raw parquet assets are unzipped and mapped perfectly.

---

## 🔬 SOTA-v31 Breakthrough: 100% Leak-Free Optuna, 2-Stage MI Selector & Circadian Decay Interpolation
> [!IMPORTANT]
> **Historic Breakthrough (Single-Model Base CV: 0.599263 - Outer-loop Leak-Free SOTA 🏆):**
> 1. **2-Stage Adaptive Feature Selector (MI + GBDT FI Filter):**
>    - Feature space inflation (300+ features) threatens GBDTs with overfitting on highly subjective cognitive targets like `Q3` (Depression).
>    - **Technical Standard**: Inside each chronological fold loop of the GBDT CV, dynamically calculate **Mutual Information (MI)** classif values to discard the bottom 20% noise features. Then, train a lightweight GBDT on the remaining space to yield Feature Importance ranking, strictly slicing the top-N features (e.g. Q3: 80, Q1/Q2: 120, S-targets: 120-180). This eliminates the curse of dimensionality strictly leak-free.
> 2. **Chronological Circadian Decay Interpolation:**
>    - Simple linear or global median imputation damages time-series rhythmicity. 
>    - **Technical Standard**: Impute missing sensor gaps by chronologically searching the nearest past/future hours. Apply an exponential decay function $e^{-\lambda \cdot \Delta t}$ relative to time distance, and pass a Sigmoid soft-fallback threshold to decay smoothly toward the user's historical median if the gap exceeds 7 days. This preserves circadian cycles without look-ahead bias.
> 3. **Late-Night Dopamine & Physiological Resonance Features:**
>    - **`dopamine_hrv_resonance`**: Captures late-night (23:00~04:00) social media screen transitions multiplied by night rolling HRV fluctuations to directly model autonomic disturbance.
>    - **`sleep_latency_activity_coherence`**: Ratio between chronological sleep latency proxy and diurnal activity stability metrics (circadian IS/IV) to model daytime-nighttime coherence.
> 4. **Circadian Cosinor & Frequency Domain Analysis (Frequency Domain LF/HF Ratio):**
>    - Cosinor analysis fits 24h sine/cosine curves to obtain MESOR, acrophase, and amplitude. LF/HF ratio calculations from raw heart rate RR intervals capture sympathetic-parasympathetic balance.


---

## 🤖 Multi-Agent Orchestration & Subagent Guardrails (Antigravity-Only Tooling)
1. **The `invoke_subagent` Execution Boundary:**
   - The `invoke_subagent` tool is a **system-private capability** exclusive to Antigravity. It cannot be executed directly by human users via the terminal or Python scripts.
   - It is used to spin up autonomous child agents in the background with inherited or shared workspaces to delegate massive workloads (e.g., hyperparameter sweeps or genetic feature evolution).
2. **Strict Registration over File-Path Manipulations:**
   - **Never** perform path-based read/write operations on the protected agent directories (like `/Users/isangjae/.gemini/antigravity-cli/agents/`) to bypass security boundaries, as this violates sandbox policies.
   - **Rule:** Always define custom subagent specifications via `define_subagent` in the active session context, then trigger them strictly using `invoke_subagent` by passing their registered `TypeName` values (e.g., `kaggle-ensemble-expert` or `kaggle-evolver`).
3. **Subagent Session Persistence:**
   - Since custom definitions are volatile and reset upon session compaction/resumption, Antigravity must proactively re-register custom agents in `define_subagent` prior to calling `invoke_subagent` inside a fresh session.

