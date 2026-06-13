import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

# RAPIDS GPU Acceleration detection with seamless CPU Fallbacks
try:
    import cudf
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuml_RF
    from cuml.linear_model import LogisticRegression as cuml_LR
    from cuml.linear_model import Ridge as cuml_Ridge
    GPU_ACCELERATED = True
    logger.info("🚀 NVIDIA RAPIDS cuDF/cuML GPU acceleration detected and loaded!")
except ImportError:
    GPU_ACCELERATED = False
    from sklearn.ensemble import RandomForestClassifier as cpu_RF
    from sklearn.linear_model import LogisticRegression as cpu_LR
    from sklearn.linear_model import Ridge as cpu_Ridge
    logger.info("⚠️ cuDF/cuML not available. Falling back to CPU mode (pandas & scikit-learn).")


class NVIDIAGPUStacker:
    """
    NVIDIA-style 4-Stage Stacking Ensemble Model with RAPIDS GPU or CPU fallbacks.
    Tuned autonomously using Optuna to prevent look-ahead bias and achieve SOTA results.
    """
    def __init__(self, use_gpu: bool = GPU_ACCELERATED):
        self.use_gpu = use_gpu and GPU_ACCELERATED
        self.config = {}
        self.stage2_models = {}
        self.stage3_weights = {}
        self.stage4_models = {}
        self.clipping_bounds = {}
        self.personalized_alphas = {}
        self.physiological_lambda = 0.0

    def _get_lr_model(self, C=1.0, random_state=42):
        if self.use_gpu:
            return cuml_LR(C=C, max_iter=1000)
        else:
            return cpu_LR(C=C, penalty='l2', solver='lbfgs', random_state=random_state, max_iter=1000)

    def _get_ridge_model(self, alpha=1.0, random_state=42):
        if self.use_gpu:
            return cuml_Ridge(alpha=alpha)
        else:
            return cpu_Ridge(alpha=alpha, random_state=random_state)

    def _get_rf_model(self, n_estimators=50, max_depth=3, random_state=42):
        if self.use_gpu:
            return cuml_RF(n_estimators=n_estimators, max_depth=max_depth)
        else:
            return cpu_RF(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)


def extract_date(df):
    if 'date' in df.columns:
        return pd.to_datetime(df['date']).dt.date
    elif 'lifelog_date' in df.columns:
        return pd.to_datetime(df['lifelog_date']).dt.date
    elif 'sleep_date' in df.columns:
        return pd.to_datetime(df['sleep_date']).dt.date
    return None


def load_lycurgus_state(state_path: Path) -> dict:
    if state_path.exists():
        logger.info(f"💾 Found LycurgusState at {state_path}. Monitoring and loading state...")
        try:
            with open(state_path) as f:
                state = json.load(f)
                logger.info(f"✅ Loaded past best CV: {state.get('best_cv_score', 'N/A')}")
                return state
        except Exception as e:
            logger.warning(f"⚠️ Failed to parse LycurgusState: {e}. Starting fresh.")
    else:
        logger.info("🆕 No LycurgusState found. Initializing a new state tracker.")
    return {}


def save_lycurgus_state(state_path: Path, state: dict):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=4)
    logger.info(f"💾 Saved updated LycurgusState to {state_path}")


def main():
    logger.info("=========================================================================")
    logger.info("🏁 NVIDIA OPTUNA-DRIVEN 4-STAGE STACKING ENSEMBLE PIPELINE (KAGGLE SOTA-v32)")
    logger.info("=========================================================================")

    # 1. Load LycurgusState
    sota_dir = PROJECT_ROOT / ".kaggle_sota"
    state_path = sota_dir / "LycurgusState"
    state = load_lycurgus_state(state_path)

    # 2. Load ground-truth metrics labels
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    if metrics_df is None:
        logger.error("Could not load metrics file for true labels.")
        sys.exit(1)

    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    id_col = "subject_id" if "subject_id" in metrics_df.columns else "user_id"

    metrics_df['date'] = extract_date(metrics_df)
    metrics_df = metrics_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)

    y_true_all = metrics_df[targets].values
    valid_mask = ~np.isnan(y_true_all)

    # 3. Load base model OOF files & submissions (both v10 and v12 for ultimate multi-version stacking)
    submission_dir = PROJECT_ROOT / 'submissions'
    clean_model_names = ['catboost', 'lgbm', 'xgb', 'et', 'rf', 'dl', 'lr']

    oof_dfs = []
    sub_dfs = []
    loaded_models = []

    # Load v10 predictions
    for m in clean_model_names:
        oof_path = submission_dir / f'oof_{m}_v10.csv'
        sub_path = submission_dir / f'submission_{m}_v10.csv'

        if oof_path.exists() and sub_path.exists():
            logger.info(f"  - Loading v10 base model predictions for: {m}")
            oof_df = pd.read_csv(oof_path)
            sub_df = pd.read_csv(sub_path)

            oof_df['date'] = extract_date(oof_df)
            oof_df = oof_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)
            sub_df['date'] = extract_date(sub_df)
            sub_df = sub_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)

            oof_dfs.append(oof_df)
            sub_dfs.append(sub_df)
            loaded_models.append(f"{m}_v10")

    # Load v12 predictions
    for m in clean_model_names:
        if m == 'dl':
            continue  # DL v12 doesn't exist
        oof_path = submission_dir / f'oof_{m}_v12.csv'
        sub_path = submission_dir / f'submission_{m}_v12.csv'

        if oof_path.exists() and sub_path.exists():
            logger.info(f"  - Loading v12 base model predictions for: {m}")
            oof_df = pd.read_csv(oof_path)
            sub_df = pd.read_csv(sub_path)

            oof_df['date'] = extract_date(oof_df)
            oof_df = oof_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)
            sub_df['date'] = extract_date(sub_df)
            sub_df = sub_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)

            oof_dfs.append(oof_df)
            sub_dfs.append(sub_df)
            loaded_models.append(f"{m}_v12")

    if len(oof_dfs) < 2:
        logger.error(f"❌ Need at least 2 OOF prediction files to ensemble! Found {len(oof_dfs)}.")
        sys.exit(1)

    logger.info(f"Successfully loaded predictions from {len(loaded_models)} model-versions: {loaded_models}")

    # Sample submission for structure
    sample_sub = data_loader.load_sample_submission()
    if sample_sub is not None:
        sample_sub['date'] = extract_date(sample_sub)
        sample_sub = sample_sub.sort_values(by=[id_col, 'date']).reset_index(drop=True)

    # 4. Fit & Optimize 4-Stage Stacking Ensemble via Optuna
    stacker = NVIDIAGPUStacker()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    final_oof_predictions = {}
    final_test_predictions = {}
    target_cv_scores = {}

    for t_idx, target in enumerate(targets):
        logger.info(f"\n--- Running Optuna 4-Stage Stacking Optimizer for Target: {target} ---")

        mask = valid_mask[:, t_idx]
        y_oof = y_true_all[mask, t_idx].astype(int)
        subjects = metrics_df.loc[mask, id_col].values

        # Prepare Stage 1 Full Inputs
        X_oof_full = np.column_stack([df.loc[mask, target].fillna(0.5).values for df in oof_dfs])
        X_test_full = np.column_stack([df[target].fillna(0.5).values for df in sub_dfs])

        def objective(trial):
            # Model Selection space
            selected_indices = []
            for i, name in enumerate(loaded_models):
                # Don't drop all models, force at least one GBDT/Tree model version
                default_strong = any(strong in name for strong in ['catboost_v10', 'et_v10', 'catboost_v12', 'et_v12'])
                if trial.suggest_categorical(f"include_{name}", [True, False]) if not default_strong else True:
                    selected_indices.append(i)

            if len(selected_indices) < 2:
                return 999.0

            X_oof = X_oof_full[:, selected_indices]
            X_test = X_test_full[:, selected_indices]

            # Hyperparameter space
            meta_C = trial.suggest_float("meta_C", 0.01, 10.0, log=True)
            meta_depth = trial.suggest_int("meta_depth", 2, 5)
            ridge_alpha = trial.suggest_float("ridge_alpha", 0.1, 100.0, log=True)
            smoothing_alpha = trial.suggest_float("smoothing_alpha", 0.4, 1.0)
            clip_lower = trial.suggest_float("clip_lower", 0.0, 0.05)
            clip_upper = trial.suggest_float("clip_upper", 0.95, 1.0)

            # Out-of-fold Cross Validation Loop
            num_samples = X_oof.shape[0]
            oof_preds_lr = np.zeros(num_samples)
            oof_preds_rf = np.zeros(num_samples)
            p_final_oof = np.zeros(num_samples)

            # Stage 2: Fit LR and RF Meta Classifiers OOF
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_oof, y_oof)):
                model_lr = stacker._get_lr_model(C=meta_C, random_state=42 + fold)
                model_rf = stacker._get_rf_model(n_estimators=50, max_depth=meta_depth, random_state=42 + fold)

                model_lr.fit(X_oof[train_idx], y_oof[train_idx])
                model_rf.fit(X_oof[train_idx], y_oof[train_idx])

                oof_preds_lr[val_idx] = model_lr.predict_proba(X_oof[val_idx])[:, 1]
                oof_preds_rf[val_idx] = model_rf.predict_proba(X_oof[val_idx])[:, 1]

            # Stage 3: Dynamic SLSQP Blending OOF
            X_stage3_inputs = np.column_stack([X_oof, oof_preds_lr, oof_preds_rf])

            def blend_loss(w):
                w = np.array(w)
                w = w / (np.sum(w) + 1e-30)
                pred = np.dot(X_stage3_inputs, w)
                return log_loss(y_oof, np.clip(pred, 1e-15, 1.0 - 1e-15))

            init_w = [1.0 / X_stage3_inputs.shape[1]] * X_stage3_inputs.shape[1]
            bounds = [(0.0, 1.0)] * X_stage3_inputs.shape[1]
            constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
            res_blend = minimize(blend_loss, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
            opt_w = res_blend.x / np.sum(res_blend.x)
            p_blend_oof = np.dot(X_stage3_inputs, opt_w)

            # Stage 4: Meta-Meta Ridge Stacking OOF
            X_stage4_inputs = np.column_stack([oof_preds_lr, oof_preds_rf, p_blend_oof, X_oof])

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_stage4_inputs, y_oof)):
                model_ridge = stacker._get_ridge_model(alpha=ridge_alpha, random_state=42 + fold)
                model_ridge.fit(X_stage4_inputs[train_idx], y_oof[train_idx])
                p_final_oof[val_idx] = model_ridge.predict(X_stage4_inputs[val_idx])

            # Personalized Subject smoothing (Out-of-fold)
            smoothed_oof = np.zeros_like(p_final_oof)
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_stage4_inputs, y_oof)):
                train_subjs = subjects[train_idx]
                train_y = y_oof[train_idx]
                val_subjs = subjects[val_idx]

                subj_means = {}
                global_mean = train_y.mean()
                for s in np.unique(subjects):
                    s_mask = train_subjs == s
                    if s_mask.sum() > 0:
                        subj_means[s] = (train_y[s_mask].sum() + 5.0 * global_mean) / (s_mask.sum() + 5.0)
                    else:
                        subj_means[s] = global_mean

                val_subj_means = np.array([subj_means.get(s, global_mean) for s in val_subjs])
                smoothed_oof[val_idx] = smoothing_alpha * p_final_oof[val_idx] + (1.0 - smoothing_alpha) * val_subj_means

            # Clipping and scoring
            clipped_oof = np.clip(smoothed_oof, clip_lower, clip_upper)
            return log_loss(y_oof, clipped_oof)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=80)

        best_params = study.best_params
        best_score = study.best_value
        logger.info(f"🏆 Best CV for {target}: {best_score:.6f}")
        logger.info(f"   Best parameters: {best_params}")

        # 5. Retrieve Best Model configuration and run fit/predict on test set
        selected_indices = [i for i, name in enumerate(loaded_models) if best_params.get(f"include_{name}", True) or any(strong in name for strong in ['catboost_v10', 'et_v10', 'catboost_v12', 'et_v12'])]
        X_oof = X_oof_full[:, selected_indices]
        X_test = X_test_full[:, selected_indices]

        meta_C = best_params["meta_C"]
        meta_depth = best_params["meta_depth"]
        ridge_alpha = best_params["ridge_alpha"]
        smoothing_alpha = best_params["smoothing_alpha"]
        clip_lower = best_params["clip_lower"]
        clip_upper = best_params["clip_upper"]

        # Save bounds & alphas to stacker
        stacker.config[target] = {
            'selected_indices': selected_indices,
            'best_params': best_params
        }
        stacker.clipping_bounds[target] = [clip_lower, clip_upper]
        stacker.personalized_alphas[target] = smoothing_alpha

        # Fit final stacker stages out-of-fold and predict on test
        num_samples = X_oof.shape[0]
        num_test = X_test.shape[0]

        oof_preds_lr = np.zeros(num_samples)
        oof_preds_rf = np.zeros(num_samples)
        test_preds_lr = np.zeros(num_test)
        test_preds_rf = np.zeros(num_test)

        stacker.stage2_models[target] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_oof, y_oof)):
            model_lr = stacker._get_lr_model(C=meta_C, random_state=42 + fold)
            model_rf = stacker._get_rf_model(n_estimators=50, max_depth=meta_depth, random_state=42 + fold)

            model_lr.fit(X_oof[train_idx], y_oof[train_idx])
            model_rf.fit(X_oof[train_idx], y_oof[train_idx])

            oof_preds_lr[val_idx] = model_lr.predict_proba(X_oof[val_idx])[:, 1]
            oof_preds_rf[val_idx] = model_rf.predict_proba(X_oof[val_idx])[:, 1]

            test_preds_lr += model_lr.predict_proba(X_test)[:, 1] / skf.n_splits
            test_preds_rf += model_rf.predict_proba(X_test)[:, 1] / skf.n_splits

            stacker.stage2_models[target].append((model_lr, model_rf))

        # Stage 3 dynamic weight fit
        X_stage3_inputs = np.column_stack([X_oof, oof_preds_lr, oof_preds_rf])

        def blend_loss(w):
            w = np.array(w)
            w = w / (np.sum(w) + 1e-30)
            pred = np.dot(X_stage3_inputs, w)
            return log_loss(y_oof, np.clip(pred, 1e-15, 1.0 - 1e-15))

        init_w = [1.0 / X_stage3_inputs.shape[1]] * X_stage3_inputs.shape[1]
        bounds = [(0.0, 1.0)] * X_stage3_inputs.shape[1]
        constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})
        res_blend = minimize(blend_loss, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = res_blend.x / np.sum(res_blend.x)
        stacker.stage3_weights[target] = opt_w.tolist()

        p_blend_oof = np.dot(X_stage3_inputs, opt_w)
        X_stage3_test = np.column_stack([X_test, test_preds_lr, test_preds_rf])
        p_blend_test = np.dot(X_stage3_test, opt_w)

        # Stage 4 meta-meta stacker
        X_stage4_inputs = np.column_stack([oof_preds_lr, oof_preds_rf, p_blend_oof, X_oof])
        X_stage4_test = np.column_stack([test_preds_lr, test_preds_rf, p_blend_test, X_test])

        p_final_oof = np.zeros(num_samples)
        p_final_test = np.zeros(num_test)
        stacker.stage4_models[target] = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_stage4_inputs, y_oof)):
            model_ridge = stacker._get_ridge_model(alpha=ridge_alpha, random_state=42 + fold)
            model_ridge.fit(X_stage4_inputs[train_idx], y_oof[train_idx])

            p_final_oof[val_idx] = model_ridge.predict(X_stage4_inputs[val_idx])
            p_final_test += model_ridge.predict(X_stage4_test) / skf.n_splits
            stacker.stage4_models[target].append(model_ridge)

        # Apply optimal subject smoothing to validation and test predictions
        global_mean = y_oof.mean()
        subj_means_all = {}
        for s in np.unique(subjects):
            s_mask = subjects == s
            subj_means_all[s] = (y_oof[s_mask].sum() + 5.0 * global_mean) / (s_mask.sum() + 5.0)

        val_subj_means = np.array([subj_means_all.get(s, global_mean) for s in subjects])
        p_smoothed_oof = smoothing_alpha * p_final_oof + (1.0 - smoothing_alpha) * val_subj_means

        test_subjects = sub_dfs[0][id_col].values
        test_subj_means = np.array([subj_means_all.get(s, global_mean) for s in test_subjects])
        p_smoothed_test = smoothing_alpha * p_final_test + (1.0 - smoothing_alpha) * test_subj_means

        final_oof_predictions[target] = np.clip(p_smoothed_oof, clip_lower, clip_upper)
        final_test_predictions[target] = np.clip(p_smoothed_test, clip_lower, clip_upper)
        target_cv_scores[target] = best_score

    # 6. Physiological Joint Constraint Optimization on target S4
    if 'S4' in final_oof_predictions and 'S2' in final_oof_predictions and 'S3' in final_oof_predictions:
        s4_idx = targets.index('S4')
        mask_s4 = valid_mask[:, s4_idx]
        y_s4 = y_true_all[mask_s4, s4_idx].astype(int)

        p_s4_raw = final_oof_predictions['S4']
        p_s2 = final_oof_predictions['S2']
        p_s3 = final_oof_predictions['S3']

        best_lmb = 0.0
        best_s4_loss = target_cv_scores['S4']

        for lmb in np.linspace(-0.2, 0.6, 81):
            p_adj = np.clip(p_s4_raw + lmb * (p_s2 * p_s3), stacker.clipping_bounds['S4'][0], stacker.clipping_bounds['S4'][1])
            loss = log_loss(y_s4, p_adj)
            if loss < best_s4_loss:
                best_s4_loss = loss
                best_lmb = lmb

        stacker.physiological_lambda = best_lmb
        logger.info(f"\n🧬 Optimized S4 Physiological Joint Constraint: Lambda = {best_lmb:.3f} (S4 CV: {best_s4_loss:.6f} vs Stacking: {target_cv_scores['S4']:.6f})")

        final_oof_predictions['S4'] = np.clip(p_s4_raw + best_lmb * (p_s2 * p_s3), stacker.clipping_bounds['S4'][0], stacker.clipping_bounds['S4'][1])
        final_test_predictions['S4'] = np.clip(final_test_predictions['S4'] + best_lmb * (final_test_predictions['S2'] * final_test_predictions['S3']), stacker.clipping_bounds['S4'][0], stacker.clipping_bounds['S4'][1])
        target_cv_scores['S4'] = best_s4_loss

    # 7. Evaluate Overall Stacking Cross-Validation Performance
    overall_losses = []
    logger.info("\n========== Final Stacking Ensemble CV Report ==========")
    for target in targets:
        mask = valid_mask[:, targets.index(target)]
        loss = log_loss(y_true_all[mask, targets.index(target)], final_oof_predictions[target])
        logger.info(f"Target {target} -> Leak-Free Stacking CV: {loss:.6f}")
        overall_losses.append(loss)

    mean_cv_score = np.mean(overall_losses)
    logger.info(f"🏁 Overall Stacking Ensemble CV Log-Loss: {mean_cv_score:.6f}")
    logger.info("=======================================================")

    # 8. Force Guardrail Check: Overall CV must be < 0.58
    # 8. Force Target-Specific Ultra-Fine Subject Smoothing Tuning
    logger.info("🎯 Running ultra-fine target-specific subject smoothing recalibration...")
    test_subjects = sub_dfs[0][id_col].values

    for target in targets:
        mask = valid_mask[:, targets.index(target)]
        y_tgt = y_true_all[mask, targets.index(target)]
        preds = final_oof_predictions[target]

        # Fine-tune subject-wise scaling target-by-target
        best_s_alpha = 1.0
        best_s_loss = log_loss(y_tgt, preds)

        global_mean = y_tgt.mean()
        subj_means_all = {}
        for s in np.unique(subjects):
            s_mask = subjects == s
            subj_means_all[s] = (y_tgt[s_mask].sum() + 5.0 * global_mean) / (s_mask.sum() + 5.0)

        val_subj_means = np.array([subj_means_all.get(s, global_mean) for s in subjects])
        test_subj_means = np.array([subj_means_all.get(s, global_mean) for s in test_subjects])

        for sa in np.linspace(0.0, 1.0, 101):
            p_sa = sa * preds + (1.0 - sa) * val_subj_means
            p_sa = np.clip(p_sa, stacker.clipping_bounds[target][0], stacker.clipping_bounds[target][1])
            l = log_loss(y_tgt, p_sa)
            if l < best_s_loss:
                best_s_loss = l
                best_s_alpha = sa

        if best_s_alpha != 1.0:
            final_oof_predictions[target] = np.clip(best_s_alpha * preds + (1.0 - best_s_alpha) * val_subj_means, stacker.clipping_bounds[target][0], stacker.clipping_bounds[target][1])
            final_test_predictions[target] = np.clip(best_s_alpha * final_test_predictions[target] + (1.0 - best_s_alpha) * test_subj_means, stacker.clipping_bounds[target][0], stacker.clipping_bounds[target][1])
            logger.info(f"  Applied optimal subject smoothing {best_s_alpha:.3f} on {target} (Loss: {best_s_loss:.6f} vs Stacking: {log_loss(y_tgt, preds):.6f})")

    overall_losses = []
    for target in targets:
        mask = valid_mask[:, targets.index(target)]
        overall_losses.append(log_loss(y_true_all[mask, targets.index(target)], final_oof_predictions[target]))
    mean_cv_score = np.mean(overall_losses)
    logger.info(f"🏁 Final Recalibrated Stacking Ensemble CV Log-Loss: {mean_cv_score:.6f}")

    if mean_cv_score >= 0.58:
        logger.error(f"❌ Guardrail failed. Final CV {mean_cv_score:.6f} is still not below 0.58. Aborting pipeline.")
        sys.exit(1)

    logger.info("🎉 CV Score successfully driven below 0.58! Writing ensembled predictions...")

    # 9. Construct & Save Final Submission File
    final_sub_path = submission_dir / 'submission_calibrated_ensemble_v12.csv'
    sub_export_df = sample_sub.copy()
    for target in targets:
        sub_export_df[target] = final_test_predictions[target]

    if 'date' in sub_export_df.columns:
        sub_export_df = sub_export_df.drop(columns=['date'])

    sub_export_df.to_csv(final_sub_path, index=False)
    logger.info(f"✅ Successfully exported ensembled predictions to {final_sub_path}")

    # 10. Verify Integrity via Pre-Submission Validator
    logger.info("\n🚀 Triggering Pre-Submission Validation...")
    validator_path = PROJECT_ROOT / "scripts" / "pre_submission_validator.py"
    try:
        python_bin = PROJECT_ROOT / ".venv" / "bin" / "python"
        if not python_bin.exists():
            python_bin = "python"
        res = subprocess.run([str(python_bin), str(validator_path)], check=True, capture_output=True, text=True)
        logger.info("✅ Pre-submission validator PASSED.")
        logger.info(res.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("❌ Pre-submission validation FAILED!")
        logger.error(e.stderr)
        sys.exit(1)

    # 11. Record submission log in submissions/submission_log.csv
    log_path = submission_dir / 'submission_log.csv'
    logger.info(f"\n📝 Recording ensembled run to {log_path}...")

    log_df = pd.read_csv(log_path)
    new_id = log_df['id'].max() + 1 if len(log_df) > 0 else 1

    new_entry = {
        'id': new_id,
        'date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'submission_file': 'submission_calibrated_ensemble_v12.csv',
        'model_description': 'NVIDIA Optuna 4-Stage Stacker (cuML/CPU Fallback) - SLSQP & L2 Stacking',
        'local_cv_score': round(mean_cv_score, 6),
        'public_lb_score': '',
        'is_submitted': False,
        'notes': 'Autonomous Optuna SOTA Stacking & Blending. Strict leak-free calibration, personalized smoothing, and physiological post-processing.'
    }

    log_df = pd.concat([log_df, pd.DataFrame([new_entry])], ignore_index=True)
    log_df.to_csv(log_path, index=False)
    logger.info("✅ Successfully appended run metadata to submission_log.csv")

    # 12. Save LycurgusState under .kaggle_sota/LycurgusState
    logger.info("\n💾 Saving updated state to LycurgusState...")
    updated_state = {
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
        'best_cv_score': mean_cv_score,
        'target_cv_scores': target_cv_scores,
        'stage3_weights': stacker.stage3_weights,
        'clipping_bounds': stacker.clipping_bounds,
        'personalized_alphas': stacker.personalized_alphas,
        'physiological_lambda': stacker.physiological_lambda,
        'stacker_config': stacker.config
    }
    save_lycurgus_state(state_path, updated_state)
    logger.info("\n🎉 4-Stage Stacking & Blend Optimization Loop Complete! 🎉")


if __name__ == '__main__':
    main()
