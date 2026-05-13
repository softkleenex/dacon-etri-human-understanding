import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import yaml
import optuna
from pathlib import Path
from omegaconf import OmegaConf

sys.path.append(str(Path.cwd()))
from src.models.dl_trainer import DLTrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def objective(trial):
    # Load base config robustly
    conf_path = Path('conf/config.yaml')
    if not conf_path.exists():
        # Auto-discover for Kaggle
        for p in Path('/kaggle/input').rglob('conf/config.yaml'):
            conf_path = p
            break
            
    with open(conf_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Override environment if running on Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        config['env'] = 'kaggle'
        
        # Auto-discover data_dir
        for p in Path('/kaggle/input').rglob('ch2026_metrics_train.csv'):
            config['paths']['kaggle']['data_dir'] = str(p.parent)
            break
        
    # Suggest hyperparameters for DL model
    config['model']['name'] = 'dl_transformer'
    config['model']['lr'] = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    config['model']['hidden_dim'] = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    config['model']['num_layers'] = trial.suggest_int("num_layers", 1, 3)
    config['model']['dropout'] = trial.suggest_float("dropout", 0.1, 0.5)
    config['model']['batch_size'] = trial.suggest_categorical("batch_size", [32, 64, 128])
    config['model']['seq_len'] = 14
    config['model']['epochs'] = 20 # Keep it relatively short for tuning
    
    # Fast CV for tuning (e.g., 3 folds instead of 5)
    config['training']['n_splits'] = 3
    config['training']['run_name'] = f'optuna_trial_{trial.number}'
    
    cfg = OmegaConf.create(config)
    
    try:
        trainer = DLTrainer(cfg)
        trainer.run()
        
        # Read the OOF logloss from the generated file or we need to extract it.
        # Since DLTrainer doesn't return the score directly, let's parse the CSV or assume we can intercept it.
        # Wait, the best way is to modify DLTrainer.run() to return the avg_score!
        # I will do that via a replace call next.
        
        # For now, let's read the latest oof file
        import glob
        import pandas as pd
        from sklearn.metrics import log_loss
        import numpy as np
        
        # The trainer saves oof_optuna_trial_X_scoreY.csv
        oof_files = glob.glob(f"submissions/oof_optuna_trial_{trial.number}_score*.csv")
        if oof_files:
            # Parse score from filename
            score_str = oof_files[0].split('_score')[-1].replace('.csv', '')
            return float(score_str)
        else:
            return 1.0 # Penalty for failure
            
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return 1.0

def main():
    logger.info("Starting Massive Kaggle GPU Optuna Tuning for CNN-Transformer...")
    
    # Create study
    os.makedirs('submissions', exist_ok=True)
    db_path = f"sqlite:///submissions/dl_optuna_study.db"
    study = optuna.create_study(direction='minimize', study_name='dl_transformer_tuning', storage=db_path, load_if_exists=True)
    
    # Run optimization
    study.optimize(objective, n_trials=30)
    
    logger.info("Tuning Complete!")
    logger.info(f"Best Score: {study.best_value}")
    logger.info(f"Best Params: {study.best_params}")

if __name__ == "__main__":
    main()
