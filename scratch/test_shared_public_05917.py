import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path("/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding")
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date"]
EPS = 1e-6

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject_cat"] = out["subject_id"].astype(str)
    for prefix, col in [("sleep", "sleep_date"), ("lifelog", "lifelog_date")]:
        dt = pd.to_datetime(out[col])
        out[f"{prefix}_month"] = dt.dt.month.astype("float")
        out[f"{prefix}_day"] = dt.dt.day.astype("float")
        out[f"{prefix}_dayofweek"] = dt.dt.dayofweek.astype("float")
        out[f"{prefix}_weekofyear"] = dt.dt.isocalendar().week.astype("float")
        out[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float")
        out[f"{prefix}_dayofyear"] = dt.dt.dayofyear.astype("float")
        
        # 순환 특성
        out[f"{prefix}_month_sin"] = np.sin(2 * np.pi * out[f"{prefix}_month"] / 12)
        out[f"{prefix}_month_cos"] = np.cos(2 * np.pi * out[f"{prefix}_month"] / 12)
        out[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * out[f"{prefix}_dayofweek"] / 7)
        out[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * out[f"{prefix}_dayofweek"] / 7)
        out[f"{prefix}_doy_sin"] = np.sin(2 * np.pi * out[f"{prefix}_dayofyear"] / 366)
        out[f"{prefix}_doy_cos"] = np.cos(2 * np.pi * out[f"{prefix}_dayofyear"] / 366)
        
    out["date_gap_days"] = (pd.to_datetime(out["sleep_date"]) - pd.to_datetime(out["lifelog_date"])).dt.days.astype("float")
    out["subject_num"] = out["subject_id"].str.extract(r"(\d+)").astype("float")
    return out

def make_subject_hole_folds(train: pd.DataFrame, n_folds: int = 5) -> list:
    all_indices = train.index.to_numpy()
    result = []
    block_count = max(n_folds * 2, 4)
    by_subject = {}
    
    # Ensure datetime sorting
    sorted_train = train.copy()
    sorted_train["sleep_date"] = pd.to_datetime(sorted_train["sleep_date"])
    
    for subject_id, group in sorted_train.sort_values(["subject_id", "sleep_date"]).groupby("subject_id", sort=False):
        indices = group.index.to_numpy()
        chunks = [chunk for chunk in np.array_split(indices, block_count) if len(chunk)]
        by_subject[str(subject_id)] = chunks
        
    for fold_id in range(n_folds):
        valid_parts = []
        for chunks in by_subject.values():
            for hole_id in (fold_id, fold_id + n_folds):
                if hole_id < len(chunks):
                    valid_parts.append(chunks[hole_id])
        if not valid_parts:
            continue
        valid_idx = np.concatenate(valid_parts)
        train_idx = np.setdiff1d(all_indices, valid_idx, assume_unique=False)
        if len(train_idx) and len(valid_idx):
            result.append((train_idx, valid_idx))
    return result

def apply_stability_filter(feature_df: pd.DataFrame, train_df: pd.DataFrame, target_cols: list, min_stability: float = 0.1, max_features_per_target: int = 250) -> dict:
    exclude_cols = set(ID_COLS + target_cols + ['subject_cat', 'date', 'lifelog_date', 'sleep_date'])
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
    
    selected_features = {}
    for target in target_cols:
        stability_scores = {}
        for feat in feature_cols:
            try:
                corr = feature_df[feat].corr(train_df[target])
                if pd.notna(corr):
                    stability_scores[feat] = abs(corr)
            except:
                pass
        filtered = {k: v for k, v in stability_scores.items() if v >= min_stability}
        sorted_feats = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_k = min(max_features_per_target, len(sorted_feats))
        selected_features[target] = [f for f, _ in sorted_feats[:top_k]]
    return selected_features

def train_lgbm_model(train_df: pd.DataFrame, features: pd.DataFrame, selected_features: dict, base_oof_path: str = None, n_folds: int = 5, seed: int = 42) -> tuple:
    cv_folds = make_subject_hole_folds(train_df, n_folds)
    model_features = features.copy()
    oof_preds = pd.DataFrame(index=train_df.index)
    models = {}
    fold_metrics = []
    base_feature_cols = []
    
    if base_oof_path and os.path.exists(base_oof_path):
        base_oof = pd.read_csv(base_oof_path)
        base_oof["date"] = pd.to_datetime(base_oof["date"]).dt.date
        
        train_df_copy = train_df.copy()
        train_df_copy["date"] = pd.to_datetime(train_df_copy["date"]).dt.date
            
        for target in TARGET_COLS:
            merged = train_df_copy[['subject_id', 'date']].merge(
                base_oof[['subject_id', 'date', target]], on=['subject_id', 'date'], how='left'
            )
            prob = merged[target].fillna(0.5).clip(EPS, 1-EPS)
            model_features[f'base_{target}_prob'] = prob.to_numpy()
            model_features[f'base_{target}_logit'] = np.log(prob / (1 - prob)).to_numpy()
        base_feature_cols = [f'base_{t}_prob' for t in TARGET_COLS] + [f'base_{t}_logit' for t in TARGET_COLS]
        
    for target in TARGET_COLS:
        target_features = selected_features.get(target, [])
        feature_cols = target_features + base_feature_cols
        X = model_features[feature_cols].fillna(0)
        y = train_df[target].values.astype(int)
        
        target_models = []
        target_oof = np.zeros(len(train_df))
        
        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
            dvalid = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=dtrain)
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 15,
                'learning_rate': 0.05,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'min_child_samples': 15,
                'seed': seed + fold_idx,
                'verbose': -1
            }
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=300,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(30, verbose=False)]
            )
            target_oof[va_idx] = model.predict(X.iloc[va_idx])
            target_models.append(model)
            
        oof_score = log_loss(y, target_oof.clip(EPS, 1-EPS))
        oof_preds[target] = target_oof
        models[target] = target_models
        fold_metrics.append({'target': target, 'oof_logloss': oof_score})
        
    avg_logloss = np.mean([m['oof_logloss'] for m in fold_metrics])
    return oof_preds, models, {'avg_logloss': avg_logloss, 'per_target': fold_metrics}

def train_xgb_model(train_df: pd.DataFrame, features: pd.DataFrame, selected_features: dict, base_oof_path: str = None, n_folds: int = 5, seed: int = 42) -> tuple:
    cv_folds = make_subject_hole_folds(train_df, n_folds)
    model_features = features.copy()
    oof_preds = pd.DataFrame(index=train_df.index)
    models = {}
    fold_metrics = []
    base_feature_cols = []
    
    if base_oof_path and os.path.exists(base_oof_path):
        base_oof = pd.read_csv(base_oof_path)
        base_oof["date"] = pd.to_datetime(base_oof["date"]).dt.date
        
        train_df_copy = train_df.copy()
        train_df_copy["date"] = pd.to_datetime(train_df_copy["date"]).dt.date
            
        for target in TARGET_COLS:
            merged = train_df_copy[['subject_id', 'date']].merge(
                base_oof[['subject_id', 'date', target]], on=['subject_id', 'date'], how='left'
            )
            prob = merged[target].fillna(0.5).clip(EPS, 1-EPS)
            model_features[f'base_{target}_prob'] = prob.to_numpy()
            model_features[f'base_{target}_logit'] = np.log(prob / (1 - prob)).to_numpy()
        base_feature_cols = [f'base_{t}_prob' for t in TARGET_COLS] + [f'base_{t}_logit' for t in TARGET_COLS]
        
    for target in TARGET_COLS:
        target_features = selected_features.get(target, [])
        feature_cols = target_features + base_feature_cols
        X = model_features[feature_cols].fillna(0)
        y = train_df[target].values.astype(int)
        
        target_models = []
        target_oof = np.zeros(len(train_df))
        
        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx])
            dvalid = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx])
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'max_depth': 4,
                'learning_rate': 0.05,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 5,
                'seed': seed + fold_idx,
                'verbosity': 0
            }
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=300,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=30,
                verbose_eval=False
            )
            target_oof[va_idx] = model.predict(dvalid)
            target_models.append(model)
            
        oof_score = log_loss(y, target_oof.clip(EPS, 1-EPS))
        oof_preds[target] = target_oof
        models[target] = target_models
        fold_metrics.append({'target': target, 'oof_logloss': oof_score})
        
    avg_logloss = np.mean([m['oof_logloss'] for m in fold_metrics])
    return oof_preds, models, {'avg_logloss': avg_logloss, 'per_target': fold_metrics}

def train_catboost_model(train_df: pd.DataFrame, features: pd.DataFrame, selected_features: dict, base_oof_path: str = None, n_folds: int = 5, seed: int = 42) -> tuple:
    cv_folds = make_subject_hole_folds(train_df, n_folds)
    model_features = features.copy()
    oof_preds = pd.DataFrame(index=train_df.index)
    models = {}
    fold_metrics = []
    base_feature_cols = []
    
    if base_oof_path and os.path.exists(base_oof_path):
        base_oof = pd.read_csv(base_oof_path)
        base_oof["date"] = pd.to_datetime(base_oof["date"]).dt.date
        
        train_df_copy = train_df.copy()
        train_df_copy["date"] = pd.to_datetime(train_df_copy["date"]).dt.date
            
        for target in TARGET_COLS:
            merged = train_df_copy[['subject_id', 'date']].merge(
                base_oof[['subject_id', 'date', target]], on=['subject_id', 'date'], how='left'
            )
            prob = merged[target].fillna(0.5).clip(EPS, 1-EPS)
            model_features[f'base_{target}_prob'] = prob.to_numpy()
            model_features[f'base_{target}_logit'] = np.log(prob / (1 - prob)).to_numpy()
        base_feature_cols = [f'base_{t}_prob' for t in TARGET_COLS] + [f'base_{t}_logit' for t in TARGET_COLS]
        
    for target in TARGET_COLS:
        target_features = selected_features.get(target, [])
        feature_cols = target_features + base_feature_cols
        X = model_features[feature_cols].fillna(0)
        y = train_df[target].values.astype(int)
        
        target_models = []
        target_oof = np.zeros(len(train_df))
        
        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            model = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=4,
                l2_leaf_reg=3,
                random_seed=seed + fold_idx,
                verbose=False,
                early_stopping_rounds=30
            )
            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=(X.iloc[va_idx], y[va_idx]),
                verbose=False
            )
            target_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            target_models.append(model)
            
        oof_score = log_loss(y, target_oof.clip(EPS, 1-EPS))
        oof_preds[target] = target_oof
        models[target] = target_models
        fold_metrics.append({'target': target, 'oof_logloss': oof_score})
        
    avg_logloss = np.mean([m['oof_logloss'] for m in fold_metrics])
    return oof_preds, models, {'avg_logloss': avg_logloss, 'per_target': fold_metrics}

def find_target_specific_weights(oof_lgbm: pd.DataFrame, oof_xgb: pd.DataFrame, oof_cat: pd.DataFrame, train_df: pd.DataFrame, n_steps: int = 11) -> dict:
    target_weights = {}
    for target in TARGET_COLS:
        best_w = None
        best_loss = float('inf')
        y_true = train_df[target].values.astype(int)
        
        for w1_pct in range(n_steps):
            for w2_pct in range(n_steps - w1_pct):
                w3_pct = n_steps - 1 - w1_pct - w2_pct
                w1 = w1_pct / (n_steps - 1)
                w2 = w2_pct / (n_steps - 1)
                w3 = w3_pct / (n_steps - 1)
                
                blended = (
                    w1 * oof_lgbm[target].values + 
                    w2 * oof_xgb[target].values + 
                    w3 * oof_cat[target].values
                )
                loss = log_loss(y_true, blended.clip(EPS, 1-EPS))
                if loss < best_loss:
                    best_loss = loss
                    best_w = {'lgbm': w1, 'xgb': w2, 'cat': w3, 'loss': best_loss}
        target_weights[target] = best_w
    return target_weights

if __name__ == '__main__':
    print("🚀 Starting Public 0.5917 Stacking pipeline replication on local data...")
    
    # Load processed datasets (includes subject_id, TARGETS, and 3400+ features)
    df_processed = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "df_train_processed.csv")
    print(f"📊 Processed feature dataset shape: {df_processed.shape}")
    
    # Parse dates
    df_processed["sleep_date"] = pd.to_datetime(df_processed["sleep_date"])
    df_processed["lifelog_date"] = pd.to_datetime(df_processed["lifelog_date"])
    
    # 1. Add Calendar features
    df_feats = add_calendar_features(df_processed)
    print("✅ Completed calendar feature addition.")
    
    # 2. Extract selected features per target using stability filter
    print("⏳ Running stability feature filter (Pearson corr per target)...")
    selected_features = apply_stability_filter(df_feats, df_processed, TARGET_COLS, min_stability=0.08, max_features_per_target=200)
    
    for t in TARGET_COLS:
        print(f"  - Target {t}: Selected {len(selected_features[t])} stable features.")
        
    # Test Run 1: Bypassing Anchor Model (No base_oof_path)
    print("\n=======================================================")
    print("🤖 Run 1: Training base models WITHOUT V152/SOTA Anchor OOF")
    print("=======================================================")
    
    oof_lgb, _, metrics_lgb = train_lgbm_model(df_processed, df_feats, selected_features, base_oof_path=None)
    print(f"  - LightGBM OOF Mean LogLoss: {metrics_lgb['avg_logloss']:.6f}")
    
    oof_xgb, _, metrics_xgb = train_xgb_model(df_processed, df_feats, selected_features, base_oof_path=None)
    print(f"  - XGBoost OOF Mean LogLoss: {metrics_xgb['avg_logloss']:.6f}")
    
    oof_cat, _, metrics_cat = train_catboost_model(df_processed, df_feats, selected_features, base_oof_path=None)
    print(f"  - CatBoost OOF Mean LogLoss: {metrics_cat['avg_logloss']:.6f}")
    
    # Find optimal blend weights per target
    target_weights = find_target_specific_weights(oof_lgb, oof_xgb, oof_cat, df_processed)
    blend_scores = []
    print("\n--- Target-Specific Blend weights (Without Anchor) ---")
    for t in TARGET_COLS:
        w = target_weights[t]
        blend_scores.append(w['loss'])
        print(f"  {t} -> LGB: {w['lgbm']:.2f} | XGB: {w['xgb']:.2f} | Cat: {w['cat']:.2f} (LogLoss: {w['loss']:.6f})")
    print(f"🔥 Final Blended CV LogLoss (No Anchor): {np.mean(blend_scores):.6f}")
    
    # Test Run 2: Incorporating SOTA v12 ExtraTrees as Anchor Model
    anchor_oof = SUBMISSIONS_DIR / "oof_et_v12.csv"
    if anchor_oof.exists():
        print("\n=======================================================")
        print(f"🤖 Run 2: Training base models WITH Anchor OOF: {anchor_oof.name}")
        print("=======================================================")
        
        oof_lgb_a, _, metrics_lgb_a = train_lgbm_model(df_processed, df_feats, selected_features, base_oof_path=anchor_oof)
        print(f"  - LightGBM (with Anchor) OOF Mean LogLoss: {metrics_lgb_a['avg_logloss']:.6f}")
        
        oof_xgb_a, _, metrics_xgb_a = train_xgb_model(df_processed, df_feats, selected_features, base_oof_path=anchor_oof)
        print(f"  - XGBoost (with Anchor) OOF Mean LogLoss: {metrics_xgb_a['avg_logloss']:.6f}")
        
        oof_cat_a, _, metrics_cat_a = train_catboost_model(df_processed, df_feats, selected_features, base_oof_path=anchor_oof)
        print(f"  - CatBoost (with Anchor) OOF Mean LogLoss: {metrics_cat_a['avg_logloss']:.6f}")
        
        target_weights_a = find_target_specific_weights(oof_lgb_a, oof_xgb_a, oof_cat_a, df_processed)
        blend_scores_a = []
        print("\n--- Target-Specific Blend weights (With Anchor) ---")
        for t in TARGET_COLS:
            w = target_weights_a[t]
            blend_scores_a.append(w['loss'])
            print(f"  {t} -> LGB: {w['lgbm']:.2f} | XGB: {w['xgb']:.2f} | Cat: {w['cat']:.2f} (LogLoss: {w['loss']:.6f})")
        print(f"🔥 Final Blended CV LogLoss (With Anchor): {np.mean(blend_scores_a):.6f}")
