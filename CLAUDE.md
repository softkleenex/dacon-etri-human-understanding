# CLAUDE.md

## 1. Build and Test Commands
Common commands for developing and auditing the codebase:

- **Virtual Environment Setup**: `.venv/bin/pip install -e .` or `make install`
- **Linter (Ruff)**: `make lint` or `.venv/bin/ruff check .`
- **Formatter (Ruff)**: `make format` or `.venv/bin/ruff check --fix . && .venv/bin/ruff format .`
- **Unit Tests (Pytest)**: `make test` or `PYTHONPATH=. .venv/bin/pytest tests/ -v`
- **Run SOTA Baseline Training**: `.venv/bin/python notebooks/core/SOTA_v31_Ultimate_100Percent_LeakFree_Pipeline.py`
- **Run Stacking Stager**: `.venv/bin/python scripts/run_nvidia_stacking_ensemble.py`
- **Submit Prediction to DACON**: `.venv/bin/python scripts/submit_calibrated.py --file submissions/<filename>`

## 2. Core Coding and Generalization Guidelines
Follow these guidelines to ensure SOTA generalization and avoid leakage:

- **100% Strictly Leak-Free Validation**:
  - NEVER fit standard scalers, target encoders, or imputers outside the CV fold loops.
  - All feature scaling, filling, and dynamic feature selection must be fitted solely on fold train indices (`tr_idx`) and transformed on valid/test.
- **Target-Specific Feature Selection**:
  - Use `mutual_info_classif` and GBDT Feature Importance inside the fold loops to dynamically reduce dimensions.
  - S-targets (physiological) are restricted to Top 120-150 features.
  - Q-targets (cognitive) are restricted to Top 80-120 features to prevent late-night overfitting.
- **Advanced Circadian Feature Engineering**:
  - **Circadian Cosinor Analysis**: Least-squares fitting of 24h sine/cosine vectors to retrieve MESOR, acrophase, and amplitude.
  - **HRV Frequency Domain**: LF/HF ratio calculations from heart rate RR intervals using FFT.
  - **Circadian Decay Interpolation**: Missing sensor data is chronologically decay-imputed with sigmoid soft-fallbacks to subject median for windows >7 days.
- **Individual Model Pre-Calibration Prohibition**:
  - NEVER apply individual model pre-calibration (Logistic Regression beta calibration) on small, unbalanced per-subject segments (~30 samples) as L2 regularization shrinks coefficients to near-zero, wiping out within-subject variation over time (dates) on the test set. 
  - Bypass individual pre-calibration and feed well-calibrated tree predictions directly to stacking. Use Step 3 (Personalized Baseline Calibration) with regularized blend factor $\alpha \in [0.5, 1.0]$ for robust, dynamic subject-level calibration.
- **Code Integrity**:
  - Keep comments and docstrings intact.
  - Match existing PEP8 indentation and quote styles (standard double quotes for strings, managed by Ruff).
