import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

    y_train = pd.Series(np.nan, index=df_copy.index)
    y_train.loc[train_idx] = y.loc[train_idx]
    df_copy['y_train_temp'] = y_train

    df_copy['date_temp'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values(by=[id_col, 'date_temp']).reset_index(drop=True)

    df_copy['date_with_target'] = df_copy['date_temp'].where(df_copy['y_train_temp'].notna())
    df_copy['y_train_prev_source'] = df_copy.groupby(id_col)['y_train_temp'].shift(1)
    df_copy['date_prev_source'] = df_copy.groupby(id_col)['date_with_target'].shift(1)

    # STRICT LEAK-FREE TARGET FLOW: Future bfill() completely removed to prevent Look-Ahead target leakage!
    df_copy['y_prev'] = df_copy.groupby(id_col)['y_train_prev_source'].ffill()
    df_copy['date_prev'] = df_copy.groupby(id_col)['date_prev_source'].ffill()

    df_copy['delta_prev'] = (df_copy['date_temp'] - pd.to_datetime(df_copy['date_prev'])).dt.days

    # Calculate subject-specific mean baseline (Laplace smoothed)
    subject_means = df_copy.groupby(id_col)['y_train_temp'].transform('mean').fillna(0.5)

    # Apply Exponential Decay Chronological Weighted Interpolation (Forward-only)
    decay_lambda = np.log(2.0) / half_life
    w_prev = np.exp(-decay_lambda * df_copy['delta_prev'].fillna(999.0))

    prior_weight = 0.2
    denom = w_prev + prior_weight
    df_copy['y_interp'] = (w_prev * df_copy['y_prev'].fillna(0.5) + prior_weight * subject_means) / denom

    # Next/Future targets are forced to subject_means to maintain schema compatibility without any look-ahead leak!
    df_copy['y_prev'] = df_copy['y_prev'].fillna(subject_means)
    df_copy['y_next'] = subject_means
    df_copy['y_interp'] = df_copy['y_interp'].fillna(subject_means)
    df_copy['delta_next'] = 999.0  # Force infinity/no next info

    df_copy = df_copy.sort_values(by='original_index').reset_index(drop=True)

    features = pd.DataFrame(index=df.index)
    features[f'{target_col}_prev'] = df_copy['y_prev'].values
    features[f'{target_col}_next'] = df_copy['y_next'].values
    features[f'{target_col}_interp'] = df_copy['y_interp'].values
    features[f'{target_col}_log_delta_prev'] = np.log1p(df_copy['delta_prev'].fillna(999.0)).values
    features[f'{target_col}_log_delta_next'] = np.log1p(df_copy['delta_next']).values

    return features

def apply_robust_directives(df_train_fold, tr_idx, target, id_col, numeric_cols, y_train_fold):
    df_train_fold = df_train_fold.copy()
    special_cols = []

    if target in ["S2", "S3", "S4"]:
        candidate_cols = [c for c in df_train_fold.columns if c.endswith("_subject_zscore")]
        corrs = {}

        # Vectorized Pearson correlation to prevent Pandas loop bottleneck in S2/S3/S4
        X_tr = df_train_fold.iloc[tr_idx][candidate_cols].fillna(0.0).values
        y_tr = y_train_fold.iloc[tr_idx].fillna(0.5).values

        X_centered = X_tr - X_tr.mean(axis=0)
        y_centered = y_tr - y_tr.mean()

        numerator = np.sum(X_centered * y_centered[:, np.newaxis], axis=0)
        denom_X = np.sqrt(np.sum(X_centered ** 2, axis=0))
        denom_y = np.sqrt(np.sum(y_centered ** 2))

        corrs_array = numerator / (denom_X * denom_y + 1e-8)

        for idx, col in enumerate(candidate_cols):
            c = corrs_array[idx]
            if pd.notna(c):
                corrs[col] = abs(c)

        top_5 = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_cols = [col for col, _ in top_5]

        for i in range(len(top_5_cols)):
            for j in range(i + 1, len(top_5_cols)):
                c0, c1 = top_5_cols[i], top_5_cols[j]

                name_add = f"{c0}_plus_{c1}"
                df_train_fold[name_add] = df_train_fold[c0] + df_train_fold[c1]
                special_cols.append(name_add)

                name_sub = f"{c0}_minus_{c1}"
                df_train_fold[name_sub] = df_train_fold[c0] - df_train_fold[c1]
                special_cols.append(name_sub)

                name_mul = f"{c0}_mul_{c1}"
                df_train_fold[name_mul] = df_train_fold[c0] * df_train_fold[c1]
                special_cols.append(name_mul)

    return df_train_fold, special_cols

def main():
    logger.info("=========================================================================")
    logger.info("🎯 STARTING SOTA-V30 CLASSIFIER CHAIN ORDER OPTIMIZER")
    logger.info("=========================================================================")

    # 1. Load data
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()

    processed_train_path = Path("data/processed/df_train_raw.parquet")
    if processed_train_path.exists():
        logger.info("Loading cached raw training dataset...")
        df_train_raw = pd.read_parquet(processed_train_path)
    else:
        logger.info("No cached training dataset. Rebuilding...")
        sensor_data_dict = data_loader.load_sensor_data()
        df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)

    df_train_raw['is_test'] = 0

    # 2. Engineer features
    df_train = engineer_features(df_train_raw)

    id_col = "subject_id" if "subject_id" in df_train.columns else "user_id"
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 3. Load DL anchor features
    lstm_oof_path = 'submissions/oof_dl_v10.csv'
    base_feature_cols = []
    if os.path.exists(lstm_oof_path):
        logger.info("DL sequence model OOF found. Merging as anchor stack features...")
        base_oof = pd.read_csv(lstm_oof_path)
        base_oof['date'] = pd.to_datetime(base_oof['date']).dt.date
        df_train['date'] = pd.to_datetime(df_train['date']).dt.date
        df_train = pd.merge(df_train, base_oof, on=[id_col, 'date'], how='left', suffixes=('', '_base'))

        eps = 1e-6
        for t in targets:
            base_col = f"{t}_base"
            if base_col in df_train.columns:
                prob = df_train[base_col].fillna(0.5).clip(eps, 1-eps)
                df_train[f'base_{t}_prob'] = prob
                df_train[f'base_{t}_logit'] = np.log(prob / (1 - prob))
                base_feature_cols.extend([f'base_{t}_prob', f'base_{t}_logit'])

    # 4. Setup CV
    n_splits = 5
    df_train = df_train.sort_values(by=[id_col, 'date']).reset_index(drop=True)
    cv_folds = make_subject_hole_folds(df_train, n_folds=n_splits)

    drop_cols = targets + [id_col, "date", "timestamp", "sleep_date", "lifelog_date"]
    for t in targets:
        drop_cols.extend([f"{t}_zscore", f"{t}_is_missing"])
        if f"{t}_base" in df_train.columns:
            drop_cols.append(f"{t}_base")

    feature_cols = [c for c in df_train.columns if c not in drop_cols and c not in base_feature_cols]
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    # 5. Pre-compute fold-wise target-independent features (Z-scaling & PCA) to bypass redundant computations!
    logger.info("========== Stage 1: Pre-computing Fold-wise Z-scaling and PCA ==========")
    fold_cached_dfs = {}
    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        logger.info(f"Caching Fold {fold+1}/{n_splits}...")
        df_train_fold = df_train.copy()

        # Subject scaling (strictly fit on train fold indices)
        z_scaler = LeakFreeSubjectZScaler(id_col=id_col, numeric_cols=numeric_cols)
        z_scaler.fit(df_train_fold.iloc[train_idx])
        df_train_fold = z_scaler.transform(df_train_fold)

        # PCA (strictly fit on train fold indices)
        pca_transformer = FoldWisePCA(n_components=5, numeric_cols=numeric_cols)
        pca_transformer.fit(df_train_fold.iloc[train_idx])
        df_train_fold = pca_transformer.transform(df_train_fold)

        fold_cached_dfs[fold] = df_train_fold
    logger.info("✅ Caching complete!")

    # 5b. Pre-compute target-specific features (Target Encoding, Temporal, Selection)
    logger.info("========== Stage 2: Pre-computing Target-Specific Features & Selection ==========")
    precomputed = {t: {} for t in targets}

    for target in targets:
        logger.info(f"Processing target: {target}...")
        y = df_train[target]
        has_label = y.notna()
        valid_indices = np.where(has_label)[0]
        valid_idx_set = set(valid_indices)

        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            tr_idx = np.array([i for i in train_idx if i in valid_idx_set])
            va_idx = np.array([i for i in val_idx if i in valid_idx_set])

            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue

            # Copy pre-scaled/PCA'd base df for this fold
            df_train_fold = fold_cached_dfs[fold].copy()
            y_train_fold = y.copy()

            # Temporal interpolation (Target-dependent)
            temp_features = add_temporal_target_features(df_train_fold, y_train_fold, tr_idx, target, id_col)
            df_train_fold = pd.concat([df_train_fold, temp_features], axis=1)

            # Laplace Target Encoding (Target-dependent)
            t_encoder = LeakFreeSubjectTargetEncoder(id_col=id_col)
            t_encoder.fit(df_train_fold.iloc[tr_idx], y_train_fold.iloc[tr_idx])
            df_train_fold = t_encoder.transform(df_train_fold)

            # Pairwise Interactions (Robust Directives)
            df_train_fold, special_cols = apply_robust_directives(
                df_train_fold, tr_idx, target, id_col, numeric_cols, y_train_fold
            )

            pca_cols = [f"pca_comp_{idx}" for idx in range(5)]

            # Feature Correlation Selection
            stability_scores = {}
            # Deep memory saving: Use Zscore features (which retain 100% of signals) and interaction features for all targets.
            # This avoids duplicating raw features matrix, preventing Mac OOM silent SIGKILL.
            candidate_cols = [c for c in df_train_fold.columns if c.endswith("_subject_zscore")] + special_cols

            # Vectorized numpy correlation calculation
            X_tr_corr = df_train_fold.iloc[tr_idx][candidate_cols].fillna(0.0).values
            y_tr_corr = y_train_fold.iloc[tr_idx].fillna(0.5).values

            X_centered = X_tr_corr - X_tr_corr.mean(axis=0)
            y_centered = y_tr_corr - y_tr_corr.mean()

            numerator = np.sum(X_centered * y_centered[:, np.newaxis], axis=0)
            denom_X = np.sqrt(np.sum(X_centered ** 2, axis=0))
            denom_y = np.sqrt(np.sum(y_centered ** 2))

            corrs = numerator / (denom_X * denom_y + 1e-8)

            for idx, feat in enumerate(candidate_cols):
                corr = corrs[idx]
                if pd.notna(corr):
                    stability_scores[feat] = abs(corr)

            selected_candidates = [k for k, v in stability_scores.items() if v >= 0.15]
            if len(selected_candidates) < 30:
                sorted_feats = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
                selected_candidates = [f for f, v in sorted_feats[:80] if v >= 0.05]
            else:
                sorted_feats = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
                selected_candidates = [f for f, v in sorted_feats[:80] if v >= 0.10]

            temporal_cols = [f'{target}_prev', f'{target}_next', f'{target}_interp', f'{target}_log_delta_prev', f'{target}_log_delta_next']
            active_cols = selected_candidates + base_feature_cols + ["subject_target_encoded"] + pca_cols + temporal_cols + special_cols

            seen = set()
            active_cols = [x for x in active_cols if not (x in seen or seen.add(x))]

            X_tr_base = df_train_fold.iloc[tr_idx][active_cols].fillna(0.5)
            y_tr = y_train_fold.iloc[tr_idx]
            X_va_base = df_train_fold.iloc[va_idx][active_cols].fillna(0.5)
            y_va = y_train_fold.iloc[va_idx]

            precomputed[target][fold] = {
                'X_tr_base': X_tr_base,
                'y_tr': y_tr,
                'X_va_base': X_va_base,
                'y_va': y_va,
                'tr_idx': tr_idx,
                'va_idx': va_idx
            }

    logger.info("✅ Pre-computation complete!")

    # 6. Evaluation function using precomputed base features
    def evaluate_chain_order(order):
        oof_preds = np.zeros((len(df_train), len(targets)))

        for chain_step, target in enumerate(order):
            t_global_idx = targets.index(target)
            prev_targets = order[:chain_step]

            # Target evaluation parameters
            y_tgt = df_train[target]
            has_label = y_tgt.notna()

            for fold in range(n_splits):
                if fold not in precomputed[target]:
                    continue

                fold_data = precomputed[target][fold]
                X_tr = fold_data['X_tr_base'].copy()
                X_va = fold_data['X_va_base'].copy()

                tr_idx = fold_data['tr_idx']
                va_idx = fold_data['va_idx']

                # Append preceding chain step OOF predictions
                for prev_t in prev_targets:
                    prev_global_idx = targets.index(prev_t)

                    blend_prev_oof_tr = oof_preds[tr_idx, prev_global_idx]
                    blend_prev_oof_va = oof_preds[va_idx, prev_global_idx]

                    X_tr[f'meta_prob_{prev_t}'] = blend_prev_oof_tr
                    X_va[f'meta_prob_{prev_t}'] = blend_prev_oof_va

                    eps = 1e-6
                    X_tr[f'meta_logit_{prev_t}'] = np.log(X_tr[f'meta_prob_{prev_t}'].clip(eps, 1-eps) / (1 - X_tr[f'meta_prob_{prev_t}'].clip(eps, 1-eps)))
                    X_va[f'meta_logit_{prev_t}'] = np.log(X_va[f'meta_prob_{prev_t}'].clip(eps, 1-eps) / (1 - X_va[f'meta_prob_{prev_t}'].clip(eps, 1-eps)))

                # Fit highly regularized fast LightGBM proxy model
                clf = lgb.LGBMClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.03,
                    reg_alpha=2.0,
                    reg_lambda=2.0,
                    random_state=42,
                    verbose=-1,
                    n_jobs=1
                )
                clf.fit(X_tr, fold_data['y_tr'])

                oof_preds[va_idx, t_global_idx] = clf.predict_proba(X_va)[:, 1]

        # Calculate Log-Loss over all targets
        scores = []
        for t_idx, target in enumerate(targets):
            y_tgt = df_train[target]
            has_label = y_tgt.notna()
            preds = oof_preds[:, targets.index(target)]
            loss = log_loss(y_tgt[has_label], np.clip(preds[has_label], 1e-15, 1-1e-15))
            scores.append(loss)

        return np.mean(scores), scores

    # 7. Sweep Search utilizing Optuna
    logger.info("\n========== Running Permutation Sweep via Optuna ==========")

    baseline_order = ["S3", "S2", "S1", "Q3", "Q2", "S4", "Q1"]
    base_score, target_base_scores = evaluate_chain_order(baseline_order)
    logger.info(f"Baseline Order: {baseline_order}")
    logger.info(f"Baseline Proxy CV Score: {base_score:.6f}")
    for t, s in zip(targets, target_base_scores):
        logger.info(f"  - {t}: {s:.5f}")

    best_overall_score = base_score
    best_overall_order = list(baseline_order)
    best_overall_target_scores = list(target_base_scores)

    def optuna_objective(trial):
        nonlocal best_overall_score, best_overall_order, best_overall_target_scores

        # Propose a target permutation
        available = list(targets)
        proposed_order = []
        for i in range(7):
            idx = trial.suggest_int(f"p_{i}", 0, len(available) - 1)
            proposed_order.append(available.pop(idx))

        score, target_scores = evaluate_chain_order(proposed_order)

        if score < best_overall_score:
            best_overall_score = score
            best_overall_order = list(proposed_order)
            best_overall_target_scores = list(target_scores)

        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(optuna_objective, n_trials=300)

    logger.info("\n" + "="*60)
    logger.info("🏁 CHAIN ORDER OPTIMIZATION COMPLETED!")
    logger.info("="*60)
    logger.info(f"Optimal Classifier Chain Order: {best_overall_order}")
    logger.info(f"Optimal Proxy CV Log-loss Score: {best_overall_score:.6f} (vs Baseline: {base_score:.6f})")
    logger.info("Target-wise Optimized Proxy CV Scores:")
    for t_idx, target in enumerate(targets):
        idx = best_overall_order.index(target)
        orig_idx = baseline_order.index(target)
        logger.info(f"  - {target}: {best_overall_target_scores[targets.index(target)]:.5f} (Baseline: {target_base_scores[targets.index(target)]:.5f})")
    logger.info("="*60)

    # Save the optimal order as a python config file or metadata
    import json
    metadata_path = Path("conf/optimal_chain_metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump({
            "optimal_chain_order": best_overall_order,
            "best_proxy_cv": best_overall_score,
            "baseline_proxy_cv": base_score
        }, f, indent=4)
    logger.info(f"✅ Saved optimal chain metadata to {metadata_path}")

if __name__ == '__main__':
    main()
