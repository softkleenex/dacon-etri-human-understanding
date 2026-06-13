import os
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.data_loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

def calibrate_beta_subject_wise(y_true, p_raw, subjects, global_T=1.0):
    eps = 1e-15
    p_raw = np.clip(p_raw, eps, 1 - eps)
    logits = np.log(p_raw / (1 - p_raw))

    unique_subjects = np.unique(subjects)
    calibrated_p = np.zeros_like(p_raw)

    for subject in unique_subjects:
        subj_mask = subjects == subject
        other_mask = ~subj_mask

        if other_mask.sum() > 20:
            clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
            clf.fit(logits[other_mask].reshape(-1, 1), y_true[other_mask])
            calibrated_p[subj_mask] = clf.predict_proba(logits[subj_mask].reshape(-1, 1))[:, 1]
        else:
            calibrated_p[subj_mask] = 1 / (1 + np.exp(-logits[subj_mask] / global_T))

    return np.clip(calibrated_p, eps, 1 - eps)

def main():
    logger.info("=========================================================================")
    logger.info("🎯 STARTING SOTA-V30 OPTUNA HYBRID BLENDING ALPHA OPTIMIZER")
    logger.info("=========================================================================")

    # 1. Load labels
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    if metrics_df is None:
        logger.error("Could not load metrics file for true labels.")
        return

    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    id_col = "subject_id" if "subject_id" in metrics_df.columns else "user_id"

    if 'lifelog_date' in metrics_df.columns:
        metrics_df['date'] = pd.to_datetime(metrics_df['lifelog_date']).dt.date
    elif 'date' not in metrics_df.columns and 'sleep_date' in metrics_df.columns:
        metrics_df['date'] = pd.to_datetime(metrics_df['sleep_date']).dt.date

    metrics_df['date'] = pd.to_datetime(metrics_df['date']).dt.date
    metrics_df = metrics_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)

    # 2. Get OOF files
    submission_dir = 'submissions'
    clean_model_names = ['catboost', 'lgbm', 'xgb', 'et', 'rf', 'dl', 'lr']
    oof_files = []
    for m in clean_model_names:
        path = os.path.join(submission_dir, f'oof_{m}_v10.csv')
        if os.path.exists(path):
            oof_files.append(path)

    if len(oof_files) < 2:
        logger.warning(f"Need at least 2 clean OOF files. Found {len(oof_files)}.")
        return

    logger.info(f"Loading {len(oof_files)} clean base models...")
    raw_oof_preds = []
    sub_files = []

    for oof_f in oof_files:
        df = pd.read_csv(oof_f)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values(by=[id_col, 'date']).reset_index(drop=True)
        raw_oof_preds.append(df)

        # Load corresponding test submission
        m_name = os.path.basename(oof_f).replace('oof_', '').replace('_v10.csv', '')
        sub_path = os.path.join(submission_dir, f'submission_{m_name}_v10.csv')
        if os.path.exists(sub_path):
            sub_files.append((m_name, pd.read_csv(sub_path)))

    # Sample submission for index
    sample_sub = data_loader.load_sample_submission()
    if sample_sub is not None:
        if 'lifelog_date' in sample_sub.columns:
            sample_sub['date'] = pd.to_datetime(sample_sub['lifelog_date']).dt.date
        elif 'sleep_date' in sample_sub.columns:
            sample_sub['date'] = pd.to_datetime(sample_sub['sleep_date']).dt.date
        sample_sub = sample_sub.sort_values(by=[id_col, 'date']).reset_index(drop=True)

    # 3. Target loop
    best_alphas = {}
    best_target_scores = {}

    final_oof_predictions = pd.DataFrame()
    final_oof_predictions[id_col] = metrics_df[id_col]
    final_oof_predictions['date'] = metrics_df['date']

    final_sub_predictions = pd.DataFrame()
    if sample_sub is not None:
        final_sub_predictions[id_col] = sample_sub[id_col]
        final_sub_predictions['date'] = sample_sub['date']

    eps = 1e-15

    for target in targets:
        logger.info(f"\n--- Optimizing Hybrid Blending Weights for Target: {target} ---")

        y_true = metrics_df[target].values
        subjects = metrics_df[id_col].values

        # Build base matrices
        # shape: (num_samples, num_models)
        OOF_mat = np.column_stack([df[target].fillna(0.5).values for df in raw_oof_preds])

        SUB_mat = None
        if len(sub_files) == len(raw_oof_preds):
            SUB_mat = np.column_stack([df[target].fillna(0.5).values for _, df in sub_files])

        # A. Find optimal global blending weights (SLSQP Minimization)
        def loss_func(weights):
            weights = np.array(weights)
            weights = weights / np.sum(weights) # normalize
            p_blend = np.dot(OOF_mat, weights)
            return log_loss(y_true, np.clip(p_blend, eps, 1-eps))

        init_weights = [1.0 / OOF_mat.shape[1]] * OOF_mat.shape[1]
        bounds = [(0.0, 1.0)] * OOF_mat.shape[1]
        constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - np.sum(w)})

        res_blend = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        best_blend_weights = res_blend.x / np.sum(res_blend.x)
        p_blend_opt = np.dot(OOF_mat, best_blend_weights)

        # B. Find optimal personalized stacking weights (Subject-wise Beta Calibration)
        # Apply Subject-wise Beta Calibration on the blended prediction
        p_stack_opt = calibrate_beta_subject_wise(y_true, p_blend_opt, subjects)

        # C. Optuna Stacking vs Blending Hybridization Alpha (w_blend * p_blend + w_stack * p_stack)
        def objective(trial):
            alpha = trial.suggest_float('alpha', 0.0, 1.0)
            p_hybrid = alpha * p_stack_opt + (1.0 - alpha) * p_blend_opt
            return log_loss(y_true, np.clip(p_hybrid, eps, 1-eps))

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        best_alpha = study.best_params['alpha']
        best_score = study.best_value

        logger.info(f"Target {target} results:")
        logger.info(f"  - Pure Blending CV: {log_loss(y_true, np.clip(p_blend_opt, eps, 1-eps)):.5f}")
        logger.info(f"  - Pure Stacking CV: {log_loss(y_true, np.clip(p_stack_opt, eps, 1-eps)):.5f}")
        logger.info(f"  - Optimized Hybrid Alpha: {best_alpha:.4f}")
        logger.info(f"  - Optimized Hybrid CV: {best_score:.5f}")

        best_alphas[target] = best_alpha
        best_target_scores[target] = best_score

        # Save optimal predictions
        p_final_oof = best_alpha * p_stack_opt + (1.0 - best_alpha) * p_blend_opt
        final_oof_predictions[target] = np.clip(p_final_oof, eps, 1-eps)

        # Make test prediction
        if SUB_mat is not None and sample_sub is not None:
            p_sub_blend = np.dot(SUB_mat, best_blend_weights)

            # Predict stacking with Logistic Regression fitted on full training logits
            logits_full = np.log(p_blend_opt / (1.0 - p_blend_opt))
            clf_full = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
            clf_full.fit(logits_full.reshape(-1, 1), y_true)

            logits_sub = np.log(p_sub_blend / (1.0 - p_sub_blend))
            p_sub_stack = clf_full.predict_proba(logits_sub.reshape(-1, 1))[:, 1]

            p_final_sub = best_alpha * p_sub_stack + (1.0 - best_alpha) * p_sub_blend
            final_sub_predictions[target] = np.clip(p_final_sub, eps, 1-eps)

    # 4. Global ensemble scoring
    avg_score = np.mean(list(best_target_scores.values()))
    logger.info("\n" + "="*50)
    logger.info("🏁 SOTA-v30 Stacking & Hybrid Blending Stacking Scoreboard")
    logger.info(f"  -> Final Stacking Hybrid CV: {avg_score:.5f}")
    logger.info("="*50)

    # Save optimal predictions
    final_sub_path = os.path.join(submission_dir, 'submission_calibrated_ensemble.csv')
    if SUB_mat is not None and sample_sub is not None:
        # Align submission structure with sample
        orig_sample = DataLoader(data_dir="data").load_sample_submission()
        final_sub_predictions_aligned = orig_sample.copy()

        for t in targets:
            final_sub_predictions_aligned[t] = final_sub_predictions[t].values

        final_sub_predictions_aligned.to_csv(final_sub_path, index=False)
        logger.info(f"✅ Optimized submission successfully written to {final_sub_path}")

if __name__ == '__main__':
    main()
