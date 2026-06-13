import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

# Suppress warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path.cwd()))
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

        if other_mask.sum() > 20: # Ensure enough samples to fit
            # Train Logistic Regression strictly on OTHER subjects (OOF)
            # Use high regularization (C=0.1) to prevent overfitting
            clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
            clf.fit(logits[other_mask].reshape(-1, 1), y_true[other_mask])

            # Predict for current subject
            calibrated_p[subj_mask] = clf.predict_proba(logits[subj_mask].reshape(-1, 1))[:, 1]
        else:
            # Fallback to temperature scaling
            calibrated_p[subj_mask] = 1 / (1 + np.exp(-logits[subj_mask] / global_T))

    return np.clip(calibrated_p, eps, 1 - eps)

def calibrate_beta_global(y_true, p_raw):
    eps = 1e-15
    p_raw = np.clip(p_raw, eps, 1 - eps)
    logits = np.log(p_raw / (1 - p_raw))

    clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
    clf.fit(logits.reshape(-1, 1), y_true)

    return clf

def run_meta_stacking_v28(submission_dir='submissions'):
    logger.info("Initializing Ultimate SOTA-v28 Stacking & Beta Calibration Pipeline...")

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

    # 2. Get OOF files (added 'lr')
    clean_model_names = ['catboost', 'lgbm', 'xgb', 'et', 'rf', 'dl', 'lr']
    oof_files = []
    for m in clean_model_names:
        path = os.path.join(submission_dir, f'oof_{m}_v10.csv')
        if os.path.exists(path):
            oof_files.append(path)

    if len(oof_files) < 2:
        logger.warning(f"Need at least 2 clean OOF files. Found {len(oof_files)}.")
        return

    logger.info(f"Loading {len(oof_files)} clean base models: {[os.path.basename(f) for f in oof_files]}...")
    raw_oof_preds = []
    sub_files = []

    for oof_f in oof_files:
        df = pd.read_csv(oof_f)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values(by=[id_col, 'date']).reset_index(drop=True)

        if not (df[id_col] == metrics_df[id_col]).all() or not (df['date'] == metrics_df['date']).all():
            logger.error(f"Mismatch in {oof_f}")
            continue

        sub_name = os.path.basename(oof_f).replace('oof_', 'submission_')
        sub_path = os.path.join(submission_dir, sub_name)
        if not os.path.exists(sub_path):
            continue

        raw_oof_preds.append(df[targets].values)
        sub_files.append(sub_path)

    num_models = len(raw_oof_preds)
    y_true = metrics_df[targets].values
    valid_mask = ~np.isnan(y_true)

    sub_dfs = [pd.read_csv(f) for f in sub_files]

    # Pre-Calibrate OOF and Test using out-of-fold subject-wise Beta Calibration
    calibrated_oof_preds = [p.copy() for p in raw_oof_preds]
    calibrated_sub_dfs = [df.copy() for df in sub_dfs]

    logger.info("Applying out-of-fold Subject-Wise Beta Calibration...")
    for m_idx in range(num_models):
        for t_idx, target in enumerate(targets):
            mask = valid_mask[:, t_idx]
            raw_p_oof = raw_oof_preds[m_idx][:, t_idx]
            pred_mask = ~np.isnan(raw_p_oof)
            combined_mask = mask & pred_mask

            if combined_mask.sum() == 0:
                continue

            y_calib = y_true[combined_mask, t_idx]
            p_calib_raw = raw_p_oof[combined_mask]
            subjects = metrics_df.loc[combined_mask, id_col].values

            # Subject-wise beta calibration out-of-fold
            calib_p_oof = raw_p_oof.copy()
            calib_p_oof[combined_mask] = calibrate_beta_subject_wise(y_calib, p_calib_raw, subjects)
            calibrated_oof_preds[m_idx][:, t_idx] = calib_p_oof

            # Subject-Specific Calibration for Test Set:
            # Fit Logistic Regression of y_calib on logits of p_calib_raw for each subject individually.
            # Then predict for that subject's test set!
            raw_p_test = sub_dfs[m_idx][target].values
            eps = 1e-15
            raw_p_test = np.clip(raw_p_test, eps, 1 - eps)
            logits_test = np.log(raw_p_test / (1 - raw_p_test))

            p_calib_clipped = np.clip(p_calib_raw, eps, 1 - eps)
            logits_train = np.log(p_calib_clipped / (1 - p_calib_clipped))

            calibrated_test_p = np.zeros_like(raw_p_test)
            test_subjects = sub_dfs[m_idx][id_col].values

            for subject in np.unique(subjects):
                subj_train_mask = subjects == subject
                subj_test_mask = test_subjects == subject

                # Ensure we have enough train samples for this subject
                if subj_train_mask.sum() > 5:
                    clf_subj = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
                    clf_subj.fit(logits_train[subj_train_mask].reshape(-1, 1), y_calib[subj_train_mask])

                    if subj_test_mask.sum() > 0:
                        calibrated_test_p[subj_test_mask] = clf_subj.predict_proba(logits_test[subj_test_mask].reshape(-1, 1))[:, 1]
                else:
                    # Fallback to global calibration
                    clf_global = calibrate_beta_global(y_calib, p_calib_raw)
                    if subj_test_mask.sum() > 0:
                        calibrated_test_p[subj_test_mask] = clf_global.predict_proba(logits_test[subj_test_mask].reshape(-1, 1))[:, 1]

            calibrated_sub_dfs[m_idx][target] = calibrated_test_p

    # Output DataFrames for Meta-Stacking
    sample_sub_df = data_loader.load_sample_submission()
    stack_sub_df = sample_sub_df.copy()
    for t in targets:
        stack_sub_df[t] = 0.5

    blended_oof_overall = []
    stacked_oof_overall = []

    final_oof_dict = {}
    final_test_dict = {}
    oof_masks = {}

    logger.info("\n========== Running Stacking vs Blending Target Sweep ==========")

    for t_idx, target in enumerate(targets):
        mask = valid_mask[:, t_idx]
        if mask.sum() == 0:
            continue

        pred_mask = np.ones_like(mask)
        for p in calibrated_oof_preds:
            pred_mask = pred_mask & ~np.isnan(p[:, t_idx])

        combined_mask = mask & pred_mask
        if combined_mask.sum() == 0:
            continue

        oof_masks[target] = combined_mask

        # Prepare inputs: columns are base model calibrated predictions
        X_oof = np.column_stack([p[combined_mask, t_idx] for p in calibrated_oof_preds])
        y_oof = y_true[combined_mask, t_idx]

        # Prepare Test inputs
        X_test = np.column_stack([df[target].values for df in calibrated_sub_dfs])

        # 1. SLSQP Logit Blending baseline
        def blend_objective(weights):
            w = weights / (np.sum(weights) + 1e-15)
            eps = 1e-15
            X_oof_clipped = np.clip(X_oof, eps, 1 - eps)
            logits = np.log(X_oof_clipped / (1.0 - X_oof_clipped))
            blended_logits = np.dot(logits, w)
            p_blended = 1.0 / (1.0 + np.exp(-blended_logits))
            p_blended = np.clip(p_blended, eps, 1 - eps)
            return log_loss(y_oof, p_blended)

        initial_weights = [1.0 / num_models] * num_models
        bounds = [(0.0, 1.0)] * num_models
        constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
        res = minimize(blend_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        opt_w = res.x / np.sum(res.x)
        blend_score = res.fun

        X_oof_clipped = np.clip(X_oof, 1e-15, 1 - 1e-15)
        logits_oof = np.log(X_oof_clipped / (1.0 - X_oof_clipped))
        blended_logits_oof = np.dot(logits_oof, opt_w)
        base_blended = 1.0 / (1.0 + np.exp(-blended_logits_oof))
        base_blended = np.clip(base_blended, 1e-15, 1 - 1e-15)

        # 2. Logistic Regression Meta-Learner Stacking (L2 regularization)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_oof = np.zeros(len(y_oof))
        meta_test_preds = np.zeros(len(X_test))

        best_c = 1.0
        best_meta_loss = 999.0

        for c_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
            fold_scores = []
            for train_idx, val_idx in skf.split(X_oof, y_oof):
                meta_clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
                meta_clf.fit(X_oof[train_idx], y_oof[train_idx])
                fold_p = meta_clf.predict_proba(X_oof[val_idx])[:, 1]
                fold_scores.append(log_loss(y_oof[val_idx], fold_p))
            mean_c_loss = np.mean(fold_scores)
            if mean_c_loss < best_meta_loss:
                best_meta_loss = mean_c_loss
                best_c = c_val

        # Generate OOF with best_c
        for train_idx, val_idx in skf.split(X_oof, y_oof):
            meta_clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
            meta_clf.fit(X_oof[train_idx], y_oof[train_idx])
            meta_oof[val_idx] = meta_clf.predict_proba(X_oof[val_idx])[:, 1]
            meta_test_preds += meta_clf.predict_proba(X_test)[:, 1] / 5.0

        stack_score = log_loss(y_oof, meta_oof)

        # Select best base prediction path (Stacking vs Blending)
        if stack_score < blend_score:
            logger.info(f"Target {target} -> Stacking chosen. (Stack CV: {stack_score:.5f} vs Blend CV: {blend_score:.5f})")
            base_oof_preds = meta_oof
            base_test_preds = np.clip(meta_test_preds, 1e-15, 1 - 1e-15)
        else:
            logger.info(f"Target {target} -> Blending chosen. (Blend CV: {blend_score:.5f} vs Stack CV: {stack_score:.5f})")
            base_oof_preds = base_blended
            X_test_clipped = np.clip(X_test, 1e-15, 1 - 1e-15)
            logits_test = np.log(X_test_clipped / (1.0 - X_test_clipped))
            blended_logits_test = np.dot(logits_test, opt_w)
            base_test_preds = 1.0 / (1.0 + np.exp(-blended_logits_test))
            base_test_preds = np.clip(base_test_preds, 1e-15, 1 - 1e-15)

        # 3. Post-Processing: Out-of-fold Personalized Baseline Calibration
        subjects = metrics_df.loc[combined_mask, id_col].values
        test_subjects = sub_dfs[0][id_col].values

        calibrated_oof = np.zeros_like(base_oof_preds)

        # We search alpha strictly out-of-fold
        for train_idx, val_idx in skf.split(X_oof, y_oof.astype(int)):
            train_subjects = subjects[train_idx]
            train_y = y_oof[train_idx]

            # Subject means strictly on train fold
            subject_means = {}
            global_mean = train_y.mean()

            for s in np.unique(subjects):
                s_mask = train_subjects == s
                if s_mask.sum() > 0:
                    subject_means[s] = (train_y[s_mask].sum() + 5.0 * global_mean) / (s_mask.sum() + 5.0)
                else:
                    subject_means[s] = global_mean

            # Find best alpha
            best_alpha = 1.0
            best_fold_loss = 999.0
            train_preds = base_oof_preds[train_idx]
            train_subj_means = np.array([subject_means[s] for s in train_subjects])

            for alpha in np.linspace(0.5, 1.0, 51):
                p_blend = alpha * train_preds + (1 - alpha) * train_subj_means
                loss = log_loss(train_y, np.clip(p_blend, 0.005, 0.995))
                if loss < best_fold_loss:
                    best_fold_loss = loss
                    best_alpha = alpha

            val_preds = base_oof_preds[val_idx]
            val_subjects = subjects[val_idx]
            val_subj_means = np.array([subject_means[s] for s in val_subjects])

            calibrated_oof[val_idx] = np.clip(best_alpha * val_preds + (1 - best_alpha) * val_subj_means, 0.005, 0.995)

        # Overall best alpha for Test set projection
        global_mean = y_oof.mean()
        subject_means_all = {}
        for s in np.unique(subjects):
            s_mask = subjects == s
            if s_mask.sum() > 0:
                subject_means_all[s] = (y_oof[s_mask].sum() + 5.0 * global_mean) / (s_mask.sum() + 5.0)
            else:
                subject_means_all[s] = global_mean

        best_alpha_global = 1.0
        best_global_loss = 999.0
        subj_means_all = np.array([subject_means_all[s] for s in subjects])

        for alpha in np.linspace(0.5, 1.0, 51):
            p_blend = alpha * base_oof_preds + (1 - alpha) * subj_means_all
            loss = log_loss(y_oof, np.clip(p_blend, 0.005, 0.995))
            if loss < best_global_loss:
                best_global_loss = loss
                best_alpha_global = alpha

        # Project test set using best global alpha
        test_subj_means = np.array([subject_means_all.get(s, global_mean) for s in test_subjects])
        final_test_preds = best_alpha_global * base_test_preds + (1 - best_alpha_global) * test_subj_means
        final_test_preds_clipped = np.clip(final_test_preds, 0.005, 0.995)

        # Save to temporary storage for post-processing
        final_oof_dict[target] = np.clip(calibrated_oof, 0.005, 0.995)
        final_test_dict[target] = final_test_preds_clipped
        blended_oof_overall.append(min(blend_score, stack_score))

    # 4. SOTA-v29: Physiological Multi-Target Joint Post-Processing on S4 (Wake After Sleep Onset)
    # Target S2 = Sleep Latency (1=high latency/poor sleep)
    # Target S3 = Deep Sleep duration (1=low duration/poor sleep)
    # Target S4 = Awakening count (1=high WASO/poor sleep)
    # Biological constraint: If latency is high (S2=1) AND duration is low (S3=1), the probability of having high S4 must increase.
    if 'S4' in final_oof_dict and 'S2' in final_oof_dict and 'S3' in final_oof_dict:
        s4_idx = targets.index('S4')
        mask_s4 = oof_masks['S4']
        y_s4 = y_true[mask_s4, s4_idx]

        p_s2_oof = final_oof_dict['S2']
        p_s3_oof = final_oof_dict['S3']
        p_s4_oof_raw = final_oof_dict['S4']

        best_lambda = 0.0
        best_s4_loss = log_loss(y_s4, p_s4_oof_raw)

        # Grid search for optimal biological parameter lambda to minimize OOF Log-Loss
        for lmb in np.linspace(0.0, 0.6, 61):
            p_s4_adj = np.clip(p_s4_oof_raw + lmb * (p_s2_oof * p_s3_oof), 0.005, 0.995)
            loss = log_loss(y_s4, p_s4_adj)
            if loss < best_s4_loss:
                best_s4_loss = loss
                best_lambda = lmb

        logger.info(f"🧬 Optimized S4 Physiological Constraint: Lambda = {best_lambda:.3f} (S4 CV: {best_s4_loss:.5f} vs Raw: {log_loss(y_s4, p_s4_oof_raw):.5f})")

        # Apply optimal multiplier
        final_oof_dict['S4'] = np.clip(p_s4_oof_raw + best_lambda * (p_s2_oof * p_s3_oof), 0.005, 0.995)
        final_test_dict['S4'] = np.clip(final_test_dict['S4'] + best_lambda * (final_test_dict['S2'] * final_test_dict['S3']), 0.005, 0.995)

    # 5. Populate and Evaluate Overall Metrics
    for t_idx, target in enumerate(targets):
        mask = oof_masks[target]
        y_oof = y_true[mask, t_idx]
        final_target_loss = log_loss(y_oof, final_oof_dict[target])
        logger.info(f"Target {target} -> Final Personalized CV: {final_target_loss:.5f}")

        stack_sub_df[target] = final_test_dict[target]
        stacked_oof_overall.append(final_target_loss)

    final_blend_cv = np.mean(blended_oof_overall)
    final_stack_cv = np.mean(stacked_oof_overall)

    logger.info("\n==========================================")
    logger.info("🏁 Ultimate SOTA-v29 Blending & Post-Processing Completed!")
    logger.info(f"  - Clean Base Ensemble CV: {final_blend_cv:.5f}")
    logger.info(f"  - Personalized Calibrated CV: {final_stack_cv:.5f}")
    logger.info("==========================================")

    hybrid_path = os.path.join(submission_dir, 'submission_calibrated_ensemble.csv')
    stack_sub_df.to_csv(hybrid_path, index=False)
    logger.info(f"✅ Best Hybrid Blending/Stacking output written to {hybrid_path}")

if __name__ == "__main__":
    run_meta_stacking_v28()
