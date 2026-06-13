import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import DataLoader
from src.data.make_dataset import build_training_dataset
from src.features.build_features import engineer_features
from src.features.transformers import FoldWisePCA, LeakFreeSubjectTargetEncoder, LeakFreeSubjectZScaler
from src.utils.logger import get_logger
from src.utils.validation import make_subject_hole_folds

logger = get_logger(__name__)

def add_temporal_target_features(df, y, train_idx, target_col, id_col, half_life=3.0):
    df_copy = df.copy()
    df_copy['original_index'] = df_copy.index

    # Isolate training labels
    y_train = pd.Series(np.nan, index=df_copy.index)
    y_train.loc[train_idx] = y.loc[train_idx]
    df_copy['y_train_temp'] = y_train

    df_copy['date_temp'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values(by=[id_col, 'date_temp']).reset_index(drop=True)

    # Shift to prevent target self-leakage on day of prediction
    df_copy['date_with_target'] = df_copy['date_temp'].where(df_copy['y_train_temp'].notna())

    # Previous Target Source (Past)
    df_copy['y_train_prev_source'] = df_copy.groupby(id_col)['y_train_temp'].shift(1)
    df_copy['date_prev_source'] = df_copy.groupby(id_col)['date_with_target'].shift(1)

    df_copy['y_prev'] = df_copy.groupby(id_col)['y_train_prev_source'].ffill()
    df_copy['date_prev'] = df_copy.groupby(id_col)['date_prev_source'].ffill()
    df_copy['delta_prev'] = (df_copy['date_temp'] - pd.to_datetime(df_copy['date_prev'])).dt.days

    # Next Target Source (Future) - Strictly Leak-Free because validation/test set indices are NaNs in y_train_temp!
    df_copy['y_train_next_source'] = df_copy.groupby(id_col)['y_train_temp'].shift(-1)
    df_copy['date_next_source'] = df_copy.groupby(id_col)['date_with_target'].shift(-1)

    df_copy['y_next'] = df_copy.groupby(id_col)['y_train_next_source'].bfill()
    df_copy['date_next'] = df_copy.groupby(id_col)['date_next_source'].bfill()
    df_copy['delta_next'] = (pd.to_datetime(df_copy['date_next']) - df_copy['date_temp']).dt.days

    # Calculate subject-specific mean baseline (Laplace smoothed)
    subject_means = df_copy.groupby(id_col)['y_train_temp'].transform('mean').fillna(0.5)

    # Apply Bidirectional Exponential Decay Chronological Weighted Interpolation
    decay_lambda = np.log(2.0) / half_life
    w_prev = np.exp(-decay_lambda * df_copy['delta_prev'].fillna(999.0))
    w_next = np.exp(-decay_lambda * df_copy['delta_next'].fillna(999.0))

    prior_weight = 0.2
    denom = w_prev + w_next + prior_weight
    df_copy['y_interp'] = (w_prev * df_copy['y_prev'].fillna(0.5) +
                           w_next * df_copy['y_next'].fillna(0.5) +
                           prior_weight * subject_means) / denom

    # Fallback to subject means for any NaNs
    df_copy['y_prev'] = df_copy['y_prev'].fillna(subject_means)
    df_copy['y_next'] = df_copy['y_next'].fillna(subject_means)
    df_copy['y_interp'] = df_copy['y_interp'].fillna(subject_means)
    df_copy['delta_prev'] = df_copy['delta_prev'].fillna(999.0)
    df_copy['delta_next'] = df_copy['delta_next'].fillna(999.0)

    df_copy = df_copy.sort_values(by='original_index').reset_index(drop=True)

    features = pd.DataFrame(index=df.index)
    features[f'{target_col}_prev'] = df_copy['y_prev'].values
    features[f'{target_col}_next'] = df_copy['y_next'].values
    features[f'{target_col}_interp'] = df_copy['y_interp'].values
    features[f'{target_col}_log_delta_prev'] = np.log1p(df_copy['delta_prev']).values
    features[f'{target_col}_log_delta_next'] = np.log1p(df_copy['delta_next']).values

    return features

def apply_robust_directives(df_train_fold, df_test_fold, tr_idx, target, id_col, numeric_cols, y_train_fold):
    """
    Applies 100% leak-free robust directives:
    1. For S2, S3, S4: Generate pairwise interactions (plus, minus, mul) for top 5 correlated candidate features.
       Skip division to prevent extreme value inflation.
    """
    df_train_fold = df_train_fold.copy()
    df_test_fold = df_test_fold.copy()

    special_cols = []

    if target in ["S2", "S3", "S4"]:
        # Find candidate columns to compute correlation
        candidate_cols = [c for c in df_train_fold.columns if c.endswith("_subject_zscore")]
        corrs = {}
        for col in candidate_cols:
            c = df_train_fold.iloc[tr_idx][col].corr(y_train_fold.iloc[tr_idx])
            if pd.notna(c):
                corrs[col] = abs(c)
        top_5 = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_cols = [col for col, _ in top_5]

        for i in range(len(top_5_cols)):
            for j in range(i + 1, len(top_5_cols)):
                c0, c1 = top_5_cols[i], top_5_cols[j]

                # Add
                name_add = f"{c0}_plus_{c1}"
                df_train_fold[name_add] = df_train_fold[c0] + df_train_fold[c1]
                df_test_fold[name_add] = df_test_fold[c0] + df_test_fold[c1]
                special_cols.append(name_add)

                # Subtract
                name_sub = f"{c0}_minus_{c1}"
                df_train_fold[name_sub] = df_train_fold[c0] - df_train_fold[c1]
                df_test_fold[name_sub] = df_test_fold[c0] - df_test_fold[c1]
                special_cols.append(name_sub)

                # Multiply
                name_mul = f"{c0}_mul_{c1}"
                df_train_fold[name_mul] = df_train_fold[c0] * df_train_fold[c1]
                df_test_fold[name_mul] = df_test_fold[c0] * df_test_fold[c1]
                special_cols.append(name_mul)

    return df_train_fold, df_test_fold, special_cols

def run_ultimate_pipeline():
    logger.info("Initializing Ultimate SOTA-v28 Sequential Classifier Chain Pipeline (6-Model Blender)...")

    # 1. Load Data
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    sample_sub_df = data_loader.load_sample_submission()

    processed_train_path = Path("data/processed/df_train_raw.parquet")
    processed_test_path = Path("data/processed/df_test_raw.parquet")

    if processed_train_path.exists() and processed_test_path.exists():
        logger.info("Found cached raw datasets in data/processed/. Loading directly for massive speedup...")
        df_train_raw = pd.read_parquet(processed_train_path)
        df_test_raw = pd.read_parquet(processed_test_path)
    else:
        logger.info("No cached datasets found. Rebuilding from raw sensor parquets...")
        sensor_data_dict = data_loader.load_sensor_data()
        df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)
        df_test_raw = build_training_dataset(sample_sub_df, sensor_data_dict)

        # Save to cache
        processed_train_path.parent.mkdir(parents=True, exist_ok=True)
        df_train_raw.to_parquet(processed_train_path, index=False)
        df_test_raw.to_parquet(processed_test_path, index=False)
        logger.info("Cached raw datasets to data/processed/ for future SOTA runs.")

    df_train_raw['is_test'] = 0
    df_test_raw['is_test'] = 1
    df_all_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # 2. Engineer Features
    df_all = engineer_features(df_all_raw)

    df_train = df_all[df_all['is_test'] == 0].drop(columns=['is_test']).copy()
    df_test = df_all[df_all['is_test'] == 1].drop(columns=['is_test']).copy()

    id_col = "subject_id" if "subject_id" in df_train.columns else "user_id"
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 3. Anchor Stacking: Incorporate Deep Learning Predictions (LSTM/Attention)
    lstm_oof_path = 'submissions/oof_dl_v10.csv'
    lstm_sub_path = 'submissions/submission_dl_v10.csv'

    base_feature_cols = []
    if os.path.exists(lstm_oof_path) and os.path.exists(lstm_sub_path):
        logger.info("Deep Learning sequence models found. Merging as anchor stack features...")
        base_oof = pd.read_csv(lstm_oof_path)
        base_oof['date'] = pd.to_datetime(base_oof['date']).dt.date
        df_train['date'] = pd.to_datetime(df_train['date']).dt.date
        df_train = pd.merge(df_train, base_oof, on=[id_col, 'date'], how='left', suffixes=('', '_base'))

        base_sub = pd.read_csv(lstm_sub_path)
        if 'date' not in base_sub.columns:
            if 'lifelog_date' in base_sub.columns:
                base_sub['date'] = base_sub['lifelog_date']
            elif 'sleep_date' in base_sub.columns:
                base_sub['date'] = base_sub['sleep_date']

        base_sub['date'] = pd.to_datetime(base_sub['date']).dt.date
        df_test['date'] = pd.to_datetime(df_test['date']).dt.date
        df_test = pd.merge(df_test, base_sub, on=[id_col, 'date'], how='left', suffixes=('', '_base'))

        eps = 1e-6
        for t in targets:
            base_col = f"{t}_base"
            if base_col in df_train.columns:
                prob = df_train[base_col].fillna(0.5).clip(eps, 1-eps)
                df_train[f'base_{t}_prob'] = prob
                df_train[f'base_{t}_logit'] = np.log(prob / (1 - prob))
            if base_col in df_test.columns:
                prob_t = df_test[base_col].fillna(0.5).clip(eps, 1-eps)
                df_test[f'base_{t}_prob'] = prob_t
                df_test[f'base_{t}_logit'] = np.log(prob_t / (1 - prob_t))
            base_feature_cols.extend([f'base_{t}_prob', f'base_{t}_logit'])

    # Setup CV
    n_splits = 5
    # Sorting train dataset chronologically per subject is critical for temporal mapping
    df_train = df_train.sort_values(by=[id_col, 'date']).reset_index(drop=True)
    df_test = df_test.sort_values(by=[id_col, 'date']).reset_index(drop=True)

    cv_folds = make_subject_hole_folds(df_train, n_folds=n_splits)

    drop_cols = targets + [id_col, "date", "timestamp", "sleep_date", "lifelog_date"]
    for t in targets:
        drop_cols.extend([f"{t}_zscore", f"{t}_is_missing"])
        if f"{t}_base" in df_train.columns:
            drop_cols.append(f"{t}_base")

    feature_cols = [c for c in df_train.columns if c not in drop_cols and c not in base_feature_cols]
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # 6 Prediction arrays
    oof_preds_cat = np.zeros((len(df_train), len(targets)))
    oof_preds_lgb = np.zeros((len(df_train), len(targets)))
    oof_preds_xgb = np.zeros((len(df_train), len(targets)))
    oof_preds_et = np.zeros((len(df_train), len(targets)))
    oof_preds_rf = np.zeros((len(df_train), len(targets)))
    oof_preds_lr = np.zeros((len(df_train), len(targets)))

    test_preds_cat = np.zeros((len(df_test), len(targets)))
    test_preds_lgb = np.zeros((len(df_test), len(targets)))
    test_preds_xgb = np.zeros((len(df_test), len(targets)))
    test_preds_et = np.zeros((len(df_test), len(targets)))
    test_preds_rf = np.zeros((len(df_test), len(targets)))
    test_preds_lr = np.zeros((len(df_test), len(targets)))

    # Define Easiest-to-Hardest Sequential Classifier Chain order (leak-free biological causality)
    chain_order = ["S3", "S2", "S1", "S4", "Q1", "Q2", "Q3"]

    # ==========================================
    # RUN SEQUENTIAL CLASSIFIER CHAIN
    # ==========================================
    logger.info("========== Starting 6-Model Sequential Classifier Chain Training ==========")
    for chain_step, target in enumerate(chain_order):
        t_global_idx = targets.index(target)
        logger.info(f"\n--- [Chain Step {chain_step+1}/7] Target: {target} ---")

        # 1. Inject previously predicted targets from the chain as features (fully out-of-fold!)
        prev_targets = chain_order[:chain_step]
        chain_feature_cols = []
        for prev_t in prev_targets:
            prev_global_idx = targets.index(prev_t)

            # 6-Model Blender OOF & Test
            blend_prev_oof = (1/6.0) * (oof_preds_cat[:, prev_global_idx] +
                                        oof_preds_lgb[:, prev_global_idx] +
                                        oof_preds_xgb[:, prev_global_idx] +
                                        oof_preds_et[:, prev_global_idx] +
                                        oof_preds_rf[:, prev_global_idx] +
                                        oof_preds_lr[:, prev_global_idx])

            blend_prev_test = (1/6.0) * (test_preds_cat[:, prev_global_idx] +
                                         test_preds_lgb[:, prev_global_idx] +
                                         test_preds_xgb[:, prev_global_idx] +
                                         test_preds_et[:, prev_global_idx] +
                                         test_preds_rf[:, prev_global_idx] +
                                         test_preds_lr[:, prev_global_idx])

            df_train[f'meta_prob_{prev_t}'] = blend_prev_oof
            df_test[f'meta_prob_{prev_t}'] = blend_prev_test

            eps = 1e-6
            df_train[f'meta_logit_{prev_t}'] = np.log(df_train[f'meta_prob_{prev_t}'].clip(eps, 1-eps) / (1 - df_train[f'meta_prob_{prev_t}'].clip(eps, 1-eps)))
            df_test[f'meta_logit_{prev_t}'] = np.log(df_test[f'meta_prob_{prev_t}'].clip(eps, 1-eps) / (1 - df_test[f'meta_prob_{prev_t}'].clip(eps, 1-eps)))

            chain_feature_cols.extend([f'meta_prob_{prev_t}', f'meta_logit_{prev_t}'])

        y = df_train[target]
        has_label = y.notna()

        valid_indices = np.where(has_label)[0]
        valid_idx_set = set(valid_indices)

        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            tr_idx = np.array([i for i in train_idx if i in valid_idx_set])
            va_idx = np.array([i for i in val_idx if i in valid_idx_set])

            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue

            df_train_fold = df_train.copy()
            y_train_fold = y.copy()

            # 0% Leak-Free Exponential Chronological Interpolation
            temp_features = add_temporal_target_features(df_train_fold, y_train_fold, tr_idx, target, id_col)

            temp_test = add_temporal_target_features(
                pd.concat([df_train_fold, df_test], ignore_index=True),
                pd.concat([y_train_fold, pd.Series(index=df_test.index)], ignore_index=True),
                np.arange(len(df_train_fold)),
                target,
                id_col
            )
            temp_features_test = temp_test.iloc[len(df_train_fold):].reset_index(drop=True)

            df_train_fold = pd.concat([df_train_fold, temp_features], axis=1)
            df_test_fold = pd.concat([df_test, temp_features_test], axis=1)

            # Subject scaling
            z_scaler = LeakFreeSubjectZScaler(id_col=id_col, numeric_cols=numeric_cols)
            z_scaler.fit(df_train_fold.iloc[tr_idx])
            df_train_fold = z_scaler.transform(df_train_fold)
            df_test_fold = z_scaler.transform(df_test_fold)

            # Laplace Target Encoding
            t_encoder = LeakFreeSubjectTargetEncoder(id_col=id_col)
            t_encoder.fit(df_train_fold.iloc[tr_idx], y_train_fold.iloc[tr_idx])
            df_train_fold = t_encoder.transform(df_train_fold)
            df_test_fold = t_encoder.transform(df_test_fold)

            # Apply Robust Directives (Pairwise Interactions for S2-S4)
            df_train_fold, df_test_fold, special_cols = apply_robust_directives(
                df_train_fold, df_test_fold, tr_idx, target, id_col, numeric_cols, y_train_fold
            )

            # PCA (for S2, S3, S4)
            pca_cols = []
            if target in ["S2", "S3", "S4"]:
                pca_transformer = FoldWisePCA(n_components=5, numeric_cols=numeric_cols)
                pca_transformer.fit(df_train_fold.iloc[tr_idx])
                df_train_fold = pca_transformer.transform(df_train_fold)
                df_test_fold = pca_transformer.transform(df_test_fold)
                pca_cols = [f"pca_comp_{idx}" for idx in range(5)]

            # Feature Correlation Selection
            stability_scores = {}
            if target in ["S1", "S2", "S3", "S4"]:
                candidate_cols = [c for c in df_train_fold.columns if c not in drop_cols and c not in base_feature_cols and not c.endswith("_freq") and "_plus_" not in c and "_minus_" not in c and "_mul_" not in c] + special_cols
            else:
                candidate_cols = [c for c in df_train_fold.columns if c.endswith("_subject_zscore")] + special_cols

            for feat in candidate_cols:
                corr = df_train_fold.iloc[tr_idx][feat].corr(y_train_fold.iloc[tr_idx])
                if pd.notna(corr):
                    stability_scores[feat] = abs(corr)

            selected_candidates = [k for k, v in stability_scores.items() if v >= 0.15]
            if len(selected_candidates) < 30:
                sorted_feats = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
                selected_candidates = [f for f, v in sorted_feats[:80] if v >= 0.05]
            else:
                # Cap maximum candidates to prevent GBDT overfitting
                sorted_feats = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
                selected_candidates = [f for f, v in sorted_feats[:80] if v >= 0.10]

            temporal_cols = [f'{target}_prev', f'{target}_next', f'{target}_interp', f'{target}_log_delta_prev', f'{target}_log_delta_next']
            active_cols = selected_candidates + base_feature_cols + chain_feature_cols + ["subject_target_encoded"] + pca_cols + temporal_cols + special_cols

            # Remove duplicates preserving order
            seen = set()
            active_cols = [x for x in active_cols if not (x in seen or seen.add(x))]

            X_tr = df_train_fold.iloc[tr_idx][active_cols].fillna(0.5)
            y_tr = y_train_fold.iloc[tr_idx]
            X_va = df_train_fold.iloc[va_idx][active_cols].fillna(0.5)
            X_te = df_test_fold[active_cols].fillna(0.5)

            # 1. CatBoost (Highly regularized, low depth)
            cat_model = CatBoostClassifier(iterations=250, depth=3, learning_rate=0.02, l2_leaf_reg=20.0, random_state=42, verbose=0, thread_count=2)
            cat_model.fit(X_tr, y_tr)
            oof_preds_cat[va_idx, t_global_idx] = cat_model.predict_proba(X_va)[:, 1]
            test_preds_cat[:, t_global_idx] += cat_model.predict_proba(X_te)[:, 1] / n_splits

            # 2. LightGBM (Highly regularized, low depth)
            lgb_model = lgb.LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, min_child_samples=10, reg_alpha=5.0, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=2)
            lgb_model.fit(X_tr, y_tr)
            oof_preds_lgb[va_idx, t_global_idx] = lgb_model.predict_proba(X_va)[:, 1]
            test_preds_lgb[:, t_global_idx] += lgb_model.predict_proba(X_te)[:, 1] / n_splits

            # 3. XGBoost (Highly regularized, low depth)
            xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, reg_alpha=5.0, reg_lambda=5.0, random_state=42, eval_metric="logloss", n_jobs=2)
            xgb_model.fit(X_tr, y_tr)
            oof_preds_xgb[va_idx, t_global_idx] = xgb_model.predict_proba(X_va)[:, 1]
            test_preds_xgb[:, t_global_idx] += xgb_model.predict_proba(X_te)[:, 1] / n_splits

            # 4. ExtraTrees (More robust bagging parameters)
            et_model = ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2)
            et_model.fit(X_tr, y_tr)
            oof_preds_et[va_idx, t_global_idx] = et_model.predict_proba(X_va)[:, 1]
            test_preds_et[:, t_global_idx] += et_model.predict_proba(X_te)[:, 1] / n_splits

            # 5. RandomForest (More robust bagging parameters)
            rf_model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2)
            rf_model.fit(X_tr, y_tr)
            oof_preds_rf[va_idx, t_global_idx] = rf_model.predict_proba(X_va)[:, 1]
            test_preds_rf[:, t_global_idx] += rf_model.predict_proba(X_te)[:, 1] / n_splits

            # 6. LogisticRegression (Robust Linear Model)
            lr_model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
            lr_model.fit(X_tr, y_tr)
            oof_preds_lr[va_idx, t_global_idx] = lr_model.predict_proba(X_va)[:, 1]
            test_preds_lr[:, t_global_idx] += lr_model.predict_proba(X_te)[:, 1] / n_splits

        logger.info(f"Target {target} CV -> CatBoost: {log_loss(y[has_label], oof_preds_cat[has_label, t_global_idx]):.4f} | LightGBM: {log_loss(y[has_label], oof_preds_lgb[has_label, t_global_idx]):.4f} | LogisticRegression: {log_loss(y[has_label], oof_preds_lr[has_label, t_global_idx]):.4f} | ExtraTrees: {log_loss(y[has_label], oof_preds_et[has_label, t_global_idx]):.4f}")

    # Save OOFs and Submissions
    logger.info("Saving corrected OOF predictions...")

    oof_template = pd.DataFrame({'date': df_train['date'], 'subject_id': df_train[id_col]})
    sub_template = pd.DataFrame({'date': df_test['date'], 'subject_id': df_test[id_col]})

    cat_oof = oof_template.copy()
    cat_sub = sub_template.copy()
    cat_oof[targets] = oof_preds_cat
    cat_sub[targets] = test_preds_cat
    cat_oof.to_csv("submissions/oof_catboost_v10.csv", index=False)
    cat_sub.to_csv("submissions/submission_catboost_v10.csv", index=False)

    lgb_oof = oof_template.copy()
    lgb_sub = sub_template.copy()
    lgb_oof[targets] = oof_preds_lgb
    lgb_sub[targets] = test_preds_lgb
    lgb_oof.to_csv("submissions/oof_lgbm_v10.csv", index=False)
    lgb_sub.to_csv("submissions/submission_lgbm_v10.csv", index=False)

    xgb_oof = oof_template.copy()
    xgb_sub = sub_template.copy()
    xgb_oof[targets] = oof_preds_xgb
    xgb_sub[targets] = test_preds_xgb
    xgb_oof.to_csv("submissions/oof_xgb_v10.csv", index=False)
    xgb_sub.to_csv("submissions/submission_xgb_v10.csv", index=False)

    et_oof = oof_template.copy()
    et_sub = sub_template.copy()
    et_oof[targets] = oof_preds_et
    et_sub[targets] = test_preds_et
    et_oof.to_csv("submissions/oof_et_v10.csv", index=False)
    et_sub.to_csv("submissions/submission_et_v10.csv", index=False)

    rf_oof = oof_template.copy()
    rf_sub = sub_template.copy()
    rf_oof[targets] = oof_preds_rf
    rf_sub[targets] = test_preds_rf
    rf_oof.to_csv("submissions/oof_rf_v10.csv", index=False)
    rf_sub.to_csv("submissions/submission_rf_v10.csv", index=False)

    lr_oof = oof_template.copy()
    lr_sub = sub_template.copy()
    lr_oof[targets] = oof_preds_lr
    lr_sub[targets] = test_preds_lr
    lr_oof.to_csv("submissions/oof_lr_v10.csv", index=False)
    lr_sub.to_csv("submissions/submission_lr_v10.csv", index=False)

    logger.info("Ultimate SOTA-v28 temporal sequential neighborhood chain model successfully trained!")

if __name__ == "__main__":
    run_ultimate_pipeline()
