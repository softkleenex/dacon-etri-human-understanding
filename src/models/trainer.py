import os
import copy
import joblib
import numpy as np
import pandas as pd
import wandb
import optuna
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, GroupKFold

from src.features.build_features import engineer_features
from src.models.model_builder import get_model
from src.utils.logger import get_logger
from src.utils.validation import clip_probabilities, validate_submission

logger = get_logger(__name__)


def evaluate_model(y_true, y_pred, y_prob=None) -> dict[str, float]:
    """Evaluates the model using Log-Loss, Macro F1 and AUC."""
    metrics = {}
    try:
        metrics["log_loss"] = log_loss(y_true, y_prob if y_prob is not None else y_pred)
    except Exception as e:
        logger.warning(f"Could not calculate Log-Loss: {e}")
    
    metrics["macro_f1"] = f1_score(y_true, (np.array(y_prob if y_prob is not None else y_pred) > 0.5).astype(int), average="macro")
    
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 2 and len(y_prob.shape) > 1 and y_prob.shape[1] > 2:
                metrics["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovo")
            elif len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
                metrics["auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics["auc"] = roc_auc_score(y_true, y_prob)
        except Exception as e:
            logger.warning(f"Could not calculate AUC: {e}")
    return metrics


class Trainer:
    """Orchestrates the training pipeline."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.env = cfg.get("env", "local")
        paths = cfg.get("paths", {}).get(self.env, {})
        self.data_dir = paths.get("data_dir", "data")
        self.model_dir = paths.get("model_dir", "models")
        self.output_dir = paths.get("output_dir", "submissions")
        self.model_type = cfg.model.get("name", "lightgbm")
        self.use_wandb = cfg.training.get("use_wandb", False)
        self.n_splits = cfg.training.get("n_splits", 5)
        self.tune_hyperparameters = cfg.training.get("tune_hyperparameters", False)
        self.n_trials = cfg.training.get("n_trials", 20)
        self.targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    def run(self):
        if self.use_wandb:
            wandb.init(
                project=self.cfg.training.get("wandb_project", "dacon-etri"),
                name=self.cfg.training.get("run_name", f"run_{self.model_type}"),
                config=dict(self.cfg),
            )
            logger.info("WandB initialized.")

        # 1. Load data
        logger.info(f"Loading data from {self.data_dir}...")

        from src.data.data_loader import DataLoader
        from src.data.make_dataset import build_training_dataset

        data_loader = DataLoader(data_dir=self.data_dir)
        metrics_df = data_loader.load_metrics()
        sensor_data_dict = data_loader.load_sensor_data()
        sample_sub_df = data_loader.load_sample_submission()

        if metrics_df is not None:
            df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)
        else:
            logger.warning("Metrics file not found. Using dummy data for testing the pipeline.")
            df_train_raw = pd.DataFrame(
                {
                    "subject_id": ["id1", "id1", "id2", "id2", "id3"],
                    "feature1": [0.5, 0.6, 0.1, np.nan, 0.2],
                    "Q1": [0, 1, 0, 0, 1],
                    "Q2": [1, 1, 0, 1, 0],
                    "Q3": [0, 0, 0, 1, 0],
                    "S1": [1, 0, 1, 0, 1],
                    "S2": [0, 1, 0, 0, 0],
                    "S3": [0, 0, 1, 1, 0],
                    "S4": [1, 1, 0, 0, 1],
                }
            )

        # 2. Process features
        logger.info("Processing features for training data...")
        df_train = engineer_features(df_train_raw)

        # Detect user/subject column
        id_col = "subject_id" if "subject_id" in df_train.columns else "user_id"
        drop_cols = self.targets + [id_col, "date", "timestamp", "sleep_date", "lifelog_date"]
        
        # Add leaked features like Q1_zscore, Q1_is_missing, etc.
        for t in self.targets:
            drop_cols.append(f"{t}_zscore")
            drop_cols.append(f"{t}_is_missing")

        # 3. Features preparation
        features_only = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns], errors="ignore")
        numeric_cols = features_only.select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Initializing and training model: {self.model_type} with {self.n_splits}-fold CV")
        
        if self.tune_hyperparameters:
            logger.info(f"Starting Optuna hyperparameter tuning for {self.n_trials} trials...")
            
            def objective(trial):
                # Suggest parameters based on model type
                params = {}
                if self.model_type == "catboost":
                    params = {
                        "iterations": trial.suggest_int("iterations", 500, 2000),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                        "depth": trial.suggest_int("depth", 4, 10),
                        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
                        "random_strength": trial.suggest_float("random_strength", 1e-9, 10, log=True),
                        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                        "border_count": trial.suggest_int("border_count", 32, 255),
                        "random_state": 42,
                        "thread_count": -1,
                        "verbose": 0,
                    }
                    if self.cfg.model.get("params", {}).get("task_type") == "GPU":
                        params["task_type"] = "GPU"
                elif self.model_type == "lightgbm":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
                        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
                        "max_depth": trial.suggest_int("max_depth", 3, 12),
                        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                        "random_state": 42,
                        "verbose": -1,
                    }
                else:
                    logger.warning("Optuna tuning is only implemented for CatBoost and LightGBM right now.")
                    return 1.0 # High loss
                    
                target_cv_scores = []
                for target in self.targets:
                    if target in df_train.columns:
                        y_train = df_train[target]
                        gkf = GroupKFold(n_splits=3) # Faster CV for tuning
                        oof_preds = np.zeros(len(y_train))
                        
                        groups_optuna = df_train[id_col]
                        
                        for fold, (train_idx, val_idx) in enumerate(gkf.split(features_only, y_train, groups=groups_optuna)):
                            X_fold_train = features_only.iloc[train_idx].copy()
                            y_fold_train = y_train.iloc[train_idx]
                            X_fold_val = features_only.iloc[val_idx].copy()
                            y_fold_val = y_train.iloc[val_idx]
                            
                            # Fold-specific Imputation
                            fold_mean = X_fold_train[numeric_cols].mean()
                            X_fold_train[numeric_cols] = X_fold_train[numeric_cols].fillna(fold_mean).fillna(0)
                            X_fold_val[numeric_cols] = X_fold_val[numeric_cols].fillna(fold_mean).fillna(0)
                            
                            model = get_model(self.model_type, params)
                            
                            model.fit(
                                X_fold_train, y_fold_train,
                                eval_set=[(X_fold_val, y_fold_val)]
                            )
                            
                            preds = model.predict_proba(X_fold_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_fold_val)
                            oof_preds[val_idx] = preds
                            
                        oof_score = log_loss(y_train, oof_preds)
                        target_cv_scores.append(oof_score)
                
                return np.mean(target_cv_scores)

            os.makedirs(self.output_dir, exist_ok=True)
            db_path = f"sqlite:///{os.path.join(self.output_dir, 'optuna_study.db')}"
            study = optuna.create_study(direction='minimize', study_name=f'{self.model_type}_tuning', storage=db_path, load_if_exists=True)
            study.optimize(objective, n_trials=self.n_trials)
            
            logger.info(f"Best Optuna Params: {study.best_params}")
            logger.info(f"Best Optuna Score: {study.best_value}")
            
            # Update cfg.model.params with best params
            current_params = OmegaConf.to_container(self.cfg.model.get("params", {}), resolve=True)
            current_params.update(study.best_params)
            self.cfg.model.params = current_params
            logger.info("Updated model parameters with Optuna best params.")

        models_dict = {t: [] for t in self.targets}
        cv_scores = {t: [] for t in self.targets}
        
        groups = df_train[id_col] if id_col in df_train.columns else None

        for target in self.targets:
            if target in df_train.columns:
                logger.info(f"--- Training model for target {target} ---")
                y_train = df_train[target]
                
                # Setup KFold
                gkf = GroupKFold(n_splits=self.n_splits)
                
                oof_preds = np.zeros(len(y_train))
                
                for fold, (train_idx, val_idx) in enumerate(gkf.split(features_only, y_train, groups=groups)):
                    X_fold_train = features_only.iloc[train_idx].copy()
                    y_fold_train = y_train.iloc[train_idx]
                    X_fold_val = features_only.iloc[val_idx].copy()
                    y_fold_val = y_train.iloc[val_idx]
                    
                    # Fold-specific Imputation
                    fold_mean = X_fold_train[numeric_cols].mean()
                    X_fold_train[numeric_cols] = X_fold_train[numeric_cols].fillna(fold_mean).fillna(0)
                    X_fold_val[numeric_cols] = X_fold_val[numeric_cols].fillna(fold_mean).fillna(0)
                    
                    model = get_model(self.model_type, self.cfg.model.get("params", {}))
                    
                    model.fit(
                        X_fold_train, y_fold_train,
                        eval_set=[(X_fold_val, y_fold_val)]
                    )
                    
                    preds = model.predict_proba(X_fold_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_fold_val)
                    oof_preds[val_idx] = preds
                    
                    models_dict[target].append(model)
                    
                    fold_score = log_loss(y_fold_val, preds)
                    logger.info(f"Fold {fold+1} log-loss: {fold_score:.4f}")
                
                oof_score = log_loss(y_train, oof_preds)
                cv_scores[target] = oof_score
                logger.info(f"OOF metrics for {target}: log-loss = {oof_score:.4f}")
                if self.use_wandb:
                    wandb.log({f"oof_logloss_{target}": oof_score})

        logger.info(f"Overall CV Score (Average Log-Loss): {np.mean(list(cv_scores.values())):.4f}")

        # 5. Save model
        logger.info(f"Saving models to {self.model_dir}...")
        os.makedirs(self.model_dir, exist_ok=True)
        for target, models in models_dict.items():
            for i, model in enumerate(models):
                joblib.dump(model, os.path.join(self.model_dir, f"{self.model_type}_{target}_fold{i}.pkl"))

        # 6. Predict for submission
        if sample_sub_df is not None:
            logger.info("Building test dataset for submission...")
            df_test_raw = build_training_dataset(sample_sub_df, sensor_data_dict)
            df_test = engineer_features(df_test_raw)
            
            X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns], errors="ignore")
            
            for col in features_only.columns:
                if col not in X_test.columns:
                    X_test[col] = np.nan
            X_test = X_test[features_only.columns]
            
            # Use training mean for test set imputation to prevent NameError and Leakage
            train_mean = features_only.mean()
            X_test = X_test.fillna(train_mean).fillna(0)
            
            submission = sample_sub_df.copy()
            
            for target in self.targets:
                if target in models_dict and len(models_dict[target]) > 0:
                    logger.info(f"Predicting for target {target}")
                    # Average predictions over folds
                    preds_sum = np.zeros(len(X_test))
                    for model in models_dict[target]:
                        preds_sum += model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                    
                    preds_avg = preds_sum / len(models_dict[target])
                    submission[target] = preds_avg

            # 4.5 Validation & Clipping
            submission = clip_probabilities(submission, self.targets)
            if not validate_submission(submission, sample_sub_df):
                logger.warning("Submission validation failed. Please check the logs. Proceeding anyway, but Dacon might reject it.")

            os.makedirs(self.output_dir, exist_ok=True)

            # Generate unique filename with run_name and score
            run_name = self.cfg.get("training", {}).get("run_name", "baseline")
            avg_score = np.mean(list(cv_scores.values()))
            sub_filename = f"submission_{run_name}_score{avg_score:.4f}.csv"
            sub_path = os.path.join(self.output_dir, sub_filename)
            
            submission.to_csv(sub_path, index=False)
            logger.info(f"Submission saved to {sub_path}")
            
            # Save metadata config alongside for reproducibility
            import json
            from omegaconf import OmegaConf
            meta_path = os.path.join(self.output_dir, f"submission_{run_name}_score{avg_score:.4f}_meta.json")
            with open(meta_path, 'w') as f:
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=4)
            
            # Also keep a copy as 'submission_latest.csv' for easy access by Makefile
            latest_path = os.path.join(self.output_dir, "submission_latest.csv")
            submission.to_csv(latest_path, index=False)

        if self.use_wandb:
            wandb.finish()

        logger.info("Training completed.")
