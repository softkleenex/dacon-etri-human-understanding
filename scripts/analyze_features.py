import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupShuffleSplit

# Ensure src is in the path
sys.path.append(str(Path.cwd()))

from src.data.data_loader import DataLoader
from src.data.make_dataset import build_training_dataset
from src.features.build_features import engineer_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

def analyze_feature_importance():
    logger.info("Starting Feature Importance Analysis (Downtime Workflow)...")
    
    # 1. Load Data
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    sensor_data_dict = data_loader.load_sensor_data()
    
    if metrics_df is None:
        logger.error("Metrics file not found. Cannot run analysis.")
        return
        
    df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)
    
    # 2. Engineer Features
    df_train = engineer_features(df_train_raw)
    
    id_col = "subject_id" if "subject_id" in df_train.columns else "user_id"
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    
    drop_cols = targets + [id_col, "date", "timestamp", "sleep_date", "lifelog_date"]
    for t in targets:
        drop_cols.extend([f"{t}_zscore", f"{t}_is_missing"])
        
    feature_cols = [c for c in df_train.columns if c not in drop_cols]
    numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    # Simple global imputation for local analysis
    features_df = df_train[numeric_cols].copy()
    features_df = features_df.fillna(features_df.mean())
    features_df = features_df.fillna(0) # For columns entirely NaN
    
    # 3. Train a quick CatBoost model on a representative target (e.g., Q3 - Stress)
    target = "Q3"
    logger.info(f"Training a quick CatBoost model to analyze feature importance for target: {target}")
    
    X = features_df
    y = df_train[target].fillna(0) # Handle NaN targets if any
    groups = df_train[id_col]
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42
    )
    
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
    
    # 4. Extract Feature Importances
    importances = model.get_feature_importance()
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(f"\n{'='*50}")
    print(f"🌟 Top 20 Most Important Features for predicting {target}")
    print(f"{'='*50}")
    print(feature_importance_df.head(20).to_string(index=False))
    
    # Save to CSV for deeper analysis
    os.makedirs('reports', exist_ok=True)
    report_path = f'reports/feature_importance_{target}.csv'
    feature_importance_df.to_csv(report_path, index=False)
    print(f"\n✅ Full feature importance report saved to {report_path}")

if __name__ == "__main__":
    analyze_feature_importance()