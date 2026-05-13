import os
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier

sys.path.append(str(Path.cwd()))
from src.data.data_loader import DataLoader
from src.data.make_dataset import build_training_dataset
from src.features.build_features import engineer_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

def train_simple_models():
    logger.info("Starting Extreme Regularization Baseline Training...")
    
    # 1. Load Data
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    sensor_data_dict = data_loader.load_sensor_data()
    df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)
    
    # 2. Engineer Features
    df_train = engineer_features(df_train_raw)
    
    id_col = "subject_id" if "subject_id" in df_train.columns else "user_id"
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    
    # Calculate Naive Baseline
    naive_losses = []
    for t in targets:
        y_true = df_train[t].dropna()
        y_pred = np.full(len(y_true), y_true.mean())
        naive_losses.append(log_loss(y_true, y_pred))
    naive_loss = np.mean(naive_losses)
    logger.info(f"Naive Log-Loss (predicting means): {naive_loss:.4f}")
    
    drop_cols = targets + [id_col, "date", "timestamp", "sleep_date", "lifelog_date"]
    for t in targets:
        drop_cols.extend([f"{t}_zscore", f"{t}_is_missing"])
        
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # 3. Aggressive Feature Pruning
    fi_path = 'reports/feature_importance_Q3.csv'
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path)
        # Use top 60 features
        top_features = fi_df['Feature'].head(60).tolist()
        # Also keep handcrafted
        hand_crafted = [c for c in df_train.columns if 'app_' in c or 'snd_' in c]
        numeric_cols = [c for c in numeric_cols if c in top_features or c in hand_crafted]
        logger.info(f"Aggressively pruned to {len(numeric_cols)} top features.")
        
    # =========================================================
    # ANCHOR STACKING (Base OOF Injection)
    # =========================================================
    # We will use the best LSTM model's OOF predictions as features
    lstm_oof_path = 'submissions/oof_local_transformer_cv_score0.6207.csv'
    lstm_sub_path = 'submissions/submission_local_transformer_cv_score0.6207.csv'
    
    base_feature_cols = []
    if os.path.exists(lstm_oof_path):
        logger.info(f"Applying Anchor Stacking using {lstm_oof_path}...")
        base_oof = pd.read_csv(lstm_oof_path)
        base_oof['date'] = pd.to_datetime(base_oof['date']).dt.date
        df_train['date'] = pd.to_datetime(df_train['date']).dt.date
        
        # Merge OOF into train
        df_train = pd.merge(df_train, base_oof, on=[id_col, 'date'], how='left', suffixes=('', '_base'))
        
        eps = 1e-6
        for t in targets:
            base_col = f"{t}_base"
            if base_col in df_train.columns:
                prob = df_train[base_col].fillna(0.5).clip(eps, 1-eps)
                df_train[f'base_{t}_prob'] = prob
                df_train[f'base_{t}_logit'] = np.log(prob / (1 - prob))
                base_feature_cols.extend([f'base_{t}_prob', f'base_{t}_logit'])
                
        numeric_cols.extend(base_feature_cols)
        logger.info(f"Added {len(base_feature_cols)} anchor features.")

    # 4. K-Fold Training with Extreme Regularization
    from src.utils.validation import make_subject_timeseries_folds
    n_splits = 5
    cv_folds = make_subject_timeseries_folds(df_train, n_folds=n_splits)
    
    target_cv_scores = []
    
    # Add ID as categorical feature
    cat_features = [id_col]
    feature_cols = numeric_cols + cat_features
    
    for target in targets:
        y = df_train[target]
        has_prediction = np.zeros(len(df_train), dtype=bool)
        
        # We only evaluate on rows where target is not nan
        oof_preds = np.zeros(len(df_train))
        
        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            has_prediction[val_idx] = True
            
            # Safe Imputation INSIDE the fold loop
            fold_mean = df_train.iloc[train_idx][numeric_cols].mean()
            
            X_train = df_train.iloc[train_idx][feature_cols].copy()
            X_train[numeric_cols] = X_train[numeric_cols].fillna(fold_mean).fillna(0)
            y_train = y.iloc[train_idx]
            train_mask = y_train.notna()
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            
            X_val = df_train.iloc[val_idx][feature_cols].copy()
            X_val[numeric_cols] = X_val[numeric_cols].fillna(fold_mean).fillna(0)
            y_val = y.iloc[val_idx]
            val_mask = y_val.notna()
            
            # Model 2: CatBoost with heavy regularization and Categorical Features
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=10, # Heavy L2
                random_strength=5, # More randomness to fight overfitting
                bagging_temperature=0.8,
                cat_features=cat_features,
                verbose=0,
                random_state=42
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val[val_mask]), early_stopping_rounds=50)
            
            if val_mask.sum() > 0:
                preds = model.predict_proba(X_val)[:, 1]
                # Clip to prevent infinity
                preds = np.clip(preds, 1e-15, 1 - 1e-15)
                oof_preds[val_idx[val_mask]] = preds[val_mask]
                
        # Calculate score for this target
        valid_idx = y.notna() & has_prediction
        target_score = log_loss(y[valid_idx], oof_preds[valid_idx])
        logger.info(f"Target {target} CV Log-Loss: {target_score:.4f}")
        target_cv_scores.append(target_score)
        
        # Store for full OOF saving
        if 'oof_df' not in locals():
            oof_df = df_train[[id_col, "date"]].copy() if "date" in df_train.columns else df_train[[id_col]].copy()
        oof_df[target] = oof_preds
        
    final_score = np.mean(target_cv_scores)
    logger.info(f"Final Simple Model CV Score: {final_score:.4f} vs Naive: {naive_loss:.4f}")
    
    # Save OOF predictions
    os.makedirs('submissions', exist_ok=True)
    oof_path = f"submissions/oof_simple_catboost_score{final_score:.4f}.csv"
    oof_df.to_csv(oof_path, index=False)
    logger.info(f"OOF predictions saved to {oof_path}")
    
    # Train on full data for submission
    logger.info("Training on full data for submission...")
    final_sub = df_train[[id_col, "date"]].copy() if "date" in df_train.columns else df_train[[id_col]].copy()
    
    # Load sample submission
    data_loader = DataLoader(data_dir="data")
    sample_sub_df = data_loader.load_sample_submission()
    if sample_sub_df is not None:
        df_test_raw = build_training_dataset(sample_sub_df, sensor_data_dict)
        # We need to concatenate train and test for proper rolling features, 
        # but for this simple ML script, since we already did it for LSTM, 
        # let's just re-use the base OOF strategy or predict on what we have.
        # Actually, let's keep it simple and just save the OOF as the submission for now
        # WAIT, we MUST predict on the actual test set to blend!
        df_test_raw['is_test'] = 1
        df_train_raw['is_test'] = 0
        df_all_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
        df_all = engineer_features(df_all_raw)
        df_test = df_all[df_all['is_test'] == 1].drop(columns=['is_test']).copy()
        
        # Anchor stacking for test set
        if os.path.exists(lstm_sub_path):
            base_sub = pd.read_csv(lstm_sub_path)
            base_sub['date'] = pd.to_datetime(base_sub['date']).dt.date if 'date' in base_sub.columns else pd.to_datetime(base_sub['sleep_date']).dt.date
            df_test['date'] = pd.to_datetime(df_test['date']).dt.date
            df_test = pd.merge(df_test, base_sub, on=[id_col, 'date'], how='left', suffixes=('', '_base'))
            
            for t in targets:
                base_col = f"{t}_base"
                if base_col in df_test.columns:
                    prob = df_test[base_col].fillna(0.5).clip(eps, 1-eps)
                    df_test[f'base_{t}_prob'] = prob
                    df_test[f'base_{t}_logit'] = np.log(prob / (1 - prob))

        X_test = df_test[feature_cols].copy()
        # Impute using full training mean
        train_mean = df_train[numeric_cols].mean()
        X_test[numeric_cols] = X_test[numeric_cols].fillna(train_mean).fillna(0)
        
        submission = sample_sub_df.copy()
        for t in targets:
            submission[t] = 0.0
            
        for target in targets:
            y = df_train[target]
            valid_mask = y.notna()
            X_train_full = df_train[feature_cols][valid_mask].copy()
            y_train_full = y[valid_mask]
            
            # Re-impute just in case
            X_train_full[numeric_cols] = X_train_full[numeric_cols].fillna(train_mean).fillna(0)
            
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=10,
                random_strength=5,
                bagging_temperature=0.8,
                cat_features=cat_features,
                verbose=0,
                random_state=42
            )
            model.fit(X_train_full, y_train_full)
            
            preds = model.predict_proba(X_test)[:, 1]
            submission[target] = np.clip(preds, 1e-15, 1 - 1e-15)
            
        sub_path = f"submissions/submission_simple_catboost_score{final_score:.4f}.csv"
        submission.to_csv(sub_path, index=False)
        logger.info(f"Test submission saved to {sub_path}")

    if final_score < naive_loss:
        logger.info("🎉 SUCCESS! The regularized model beat the naive baseline.")
    else:
        logger.warning("❌ FAIL! The model is still overfitting and worse than random guessing.")

if __name__ == "__main__":
    train_simple_models()