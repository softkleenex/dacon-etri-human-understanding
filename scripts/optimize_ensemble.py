import os
import sys
import glob
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import log_loss

sys.path.append(str(Path.cwd()))
from src.data.data_loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

def optimize_ensemble_weights(submission_dir='submissions'):
    """
    Finds OOF files, loads them, and uses scipy.optimize to find the best blending weights.
    Then, it blends the corresponding test submission files using these optimal weights.
    """
    logger.info("Starting Ensemble Weight Optimization...")
    
    # 1. Load True Targets
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    if metrics_df is None:
        logger.error("Could not load metrics file for true labels.")
        return
        
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    id_col = "subject_id" if "subject_id" in metrics_df.columns else "user_id"
    
    # Sort for alignment
    if 'lifelog_date' in metrics_df.columns:
        metrics_df['date'] = pd.to_datetime(metrics_df['lifelog_date']).dt.date
    elif 'date' not in metrics_df.columns and 'sleep_date' in metrics_df.columns:
        metrics_df['date'] = pd.to_datetime(metrics_df['sleep_date']).dt.date
        
    metrics_df['date'] = pd.to_datetime(metrics_df['date']).dt.date
    metrics_df = metrics_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)
    
    # 2. Find OOF files
    oof_files = glob.glob(os.path.join(submission_dir, 'oof_*.csv'))
    if len(oof_files) < 2:
        logger.warning(f"Need at least 2 OOF files for optimization. Found {len(oof_files)}.")
        return
        
    logger.info(f"Found {len(oof_files)} OOF files to blend:")
    oof_preds = []
    sub_files = []
    
    for oof_f in oof_files:
        logger.info(f" - {oof_f}")
        df = pd.read_csv(oof_f)
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df.sort_values(by=[id_col, 'date']).reset_index(drop=True)
        
        # Verify alignment
        if not (df[id_col] == metrics_df[id_col]).all() or not (df['date'] == metrics_df['date']).all():
            logger.error(f"User ID or Date mismatch in {oof_f}")
            continue
            
        # Try to find corresponding submission file
        # format: oof_RUNNAME_scoreXXXX.csv -> submission_RUNNAME_scoreXXXX.csv
        sub_name = os.path.basename(oof_f).replace('oof_', 'submission_')
        sub_path = os.path.join(submission_dir, sub_name)
        if not os.path.exists(sub_path):
            logger.warning(f"No matching submission file found for {oof_f}. Expected {sub_path}")
            continue
            
        oof_preds.append(df[targets].values)
        sub_files.append(sub_path)
        
    num_models = len(oof_preds)
    if num_models < 2:
        logger.warning("Not enough matched OOF + Submission files.")
        return
        
    logger.info(f"Optimizing blending weights for {num_models} models...")
    
    # 3. Optimization Objective (Minimize Log-Loss)
    y_true = metrics_df[targets].values
    
    # We create a mask for valid targets
    valid_mask = ~np.isnan(y_true)
    
    def log_loss_objective(weights):
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        # Blend OOF predictions
        blended_preds = np.zeros_like(oof_preds[0])
        for w, p in zip(weights, oof_preds):
            blended_preds += w * p
            
        # Clip to prevent infinity
        blended_preds = np.clip(blended_preds, 1e-15, 1 - 1e-15)
        
        # Calculate mean log loss across valid targets
        total_loss = 0
        active = 0
        for i in range(len(targets)):
            mask = valid_mask[:, i]
            if mask.sum() > 0:
                total_loss += log_loss(y_true[mask, i], blended_preds[mask, i])
                active += 1
                
        return total_loss / active if active > 0 else 0.0

    # Initial guess: equal weights
    initial_weights = [1.0 / num_models] * num_models
    bounds = [(0.0, 1.0)] * num_models
    
    # Constrain weights to sum to 1
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    
    res = minimize(log_loss_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    optimal_weights = res.x / np.sum(res.x)
    optimal_score = res.fun
    
    logger.info(f"Optimization Successful! New Ensemble CV Score: {optimal_score:.4f}")
    for i, w in enumerate(optimal_weights):
        logger.info(f" - Weight {w:.4f} for {os.path.basename(sub_files[i])}")
        
    # 4. Create Final Optimal Submission
    logger.info("Applying weights to test submissions...")
    
    ensemble_sub_df = pd.read_csv(sub_files[0]).copy()
    for t in targets:
        ensemble_sub_df[t] = 0.0
        
    for w, sub_f in zip(optimal_weights, sub_files):
        df = pd.read_csv(sub_f)
        for t in targets:
            ensemble_sub_df[t] += w * df[t]
            
    # Clip
    for t in targets:
        ensemble_sub_df[t] = np.clip(ensemble_sub_df[t], 1e-15, 1 - 1e-15)
        
    output_path = os.path.join(submission_dir, 'submission_optimal_ensemble_latest.csv')
    ensemble_sub_df.to_csv(output_path, index=False)
    logger.info(f"✅ Optimally Blended Submission saved to {output_path}")
    
if __name__ == "__main__":
    optimize_ensemble_weights()