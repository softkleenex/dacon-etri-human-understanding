import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from sklearn.impute import KNNImputer
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import DataLoader
from src.data.dl_data_module import DaconDataModule
from src.data.make_dataset import build_training_dataset
from src.features.build_features import engineer_features
from src.utils.logger import get_logger
from src.utils.validation import clip_probabilities, validate_submission

logger = get_logger(__name__)


class DLTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.env = cfg.get("env", "local")
        paths = cfg.get("paths", {}).get(self.env, {})
        self.data_dir = paths.get("data_dir", "data")
        self.model_dir = paths.get("model_dir", "models")
        self.output_dir = paths.get("output_dir", "submissions")
        self.use_wandb = cfg.training.get("use_wandb", False)
        self.n_splits = cfg.training.get("n_splits", 5)
        self.targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    def prepare_data(self):
        import os

        cache_train_path = os.path.join(self.data_dir, "processed", "df_train_preloaded_cache.parquet")
        cache_test_path = os.path.join(self.data_dir, "processed", "df_test_preloaded_cache.parquet")

        if os.path.exists(cache_train_path) and os.path.exists(cache_test_path):
            logger.info("Found preloaded feature cache in processed/ directory. Loading from parquet cache...")
            try:
                df_train = pd.read_parquet(cache_train_path)
                df_test = pd.read_parquet(cache_test_path)
                return df_train, df_test
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-engineering features...")

        # 1. Load Data
        data_loader = DataLoader(data_dir=self.data_dir)
        metrics_df = data_loader.load_metrics()
        sensor_data_dict = data_loader.load_sensor_data()
        sample_sub_df = data_loader.load_sample_submission()

        if metrics_df is not None:
            df_train_raw = build_training_dataset(metrics_df, sensor_data_dict)
        else:
            raise ValueError("Metrics file not found. Cannot train.")

        # 2. Engineer Features
        if sample_sub_df is not None:
            df_test_raw = build_training_dataset(sample_sub_df, sensor_data_dict)
            df_train_raw["is_test"] = 0
            df_test_raw["is_test"] = 1
            df_all_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
            df_all = engineer_features(df_all_raw)

            # Inject HF Chronos Embeddings into df_all
            import os

            hf_emb_path = "data/processed/hf_chronos_embeddings.csv"
            if os.path.exists(hf_emb_path):
                logger.info(f"Applying HF Chronos Embeddings from {hf_emb_path}...")
                id_col_temp = "subject_id" if "subject_id" in df_all.columns else "user_id"
                hf_emb_df = pd.read_csv(hf_emb_path)
                hf_emb_df["date"] = pd.to_datetime(hf_emb_df["date"]).dt.date
                df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
                df_all = pd.merge(df_all, hf_emb_df, on=[id_col_temp, "date"], how="left")

            df_train = df_all[df_all["is_test"] == 0].drop(columns=["is_test"]).copy().reset_index(drop=True)
            df_test = df_all[df_all["is_test"] == 1].drop(columns=["is_test"]).copy().reset_index(drop=True)
        else:
            df_train = engineer_features(df_train_raw)

            # Inject HF Chronos Embeddings into df_train
            import os

            hf_emb_path = "data/processed/hf_chronos_embeddings.csv"
            if os.path.exists(hf_emb_path):
                logger.info(f"Applying HF Chronos Embeddings from {hf_emb_path}...")
                id_col_temp = "subject_id" if "subject_id" in df_train.columns else "user_id"
                hf_emb_df = pd.read_csv(hf_emb_path)
                hf_emb_df["date"] = pd.to_datetime(hf_emb_df["date"]).dt.date
                df_train["date"] = pd.to_datetime(df_train["date"]).dt.date
                df_train = pd.merge(df_train, hf_emb_df, on=[id_col_temp, "date"], how="left")

            df_test = None

        # Save to parquet cache
        try:
            os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
            df_train.to_parquet(cache_train_path, index=False)
            if df_test is not None:
                df_test.to_parquet(cache_test_path, index=False)
            logger.info("Successfully cached preloaded features to processed/ directory.")
        except Exception as e:
            logger.warning(f"Failed to cache preloaded features: {e}")

        return df_train, df_test
        return df_train, df_test

    def run(self, preloaded_train=None, preloaded_test=None):
        logger.info("Starting Deep Learning Pipeline...")

        if preloaded_train is not None:
            logger.info("Using preloaded engineered features. Skipping loading and feature engineering.")
            df_train = preloaded_train
            self.df_test_base = preloaded_test
        else:
            df_train, self.df_test_base = self.prepare_data()

        id_col = "subject_id" if "subject_id" in df_train.columns else "user_id"
        drop_cols = self.targets + [id_col, "date", "timestamp", "sleep_date", "lifelog_date"]
        for t in self.targets:
            drop_cols.extend([f"{t}_zscore", f"{t}_is_missing"])

        feature_cols = [c for c in df_train.columns if c not in drop_cols]
        numeric_cols = df_train[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        # 3. K-Fold Training
        model_name = self.cfg.model.get("name", "dl_mlp")
        use_sequence = model_name in ["dl_lstm", "dl_transformer"]
        seq_len = self.cfg.model.get("seq_len", 7)

        logger.info(f"Training {model_name} with {self.n_splits}-fold CV. Sequence modeling: {use_sequence}")

        from src.utils.validation import make_subject_timeseries_folds

        cv_folds = make_subject_timeseries_folds(df_train, n_folds=self.n_splits)

        oof_preds = {t: np.full(len(df_train), np.nan) for t in self.targets}
        has_prediction = np.zeros(len(df_train), dtype=bool)
        trained_models = []
        fold_processors = []  # Store imputers and scalers per fold

        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            logger.info(f"--- Fold {fold + 1}/{self.n_splits} ---")

            # Record that these indices received a validation prediction
            has_prediction[val_idx] = True

            # Prevent data leakage: Fit Imputer and Scaler ONLY on train_idx
            df_fold = df_train.copy()

            valid_cols = [c for c in numeric_cols if not df_train[c].isna().all()]

            imputer = KNNImputer(n_neighbors=5)
            scaler = StandardScaler()

            if len(valid_cols) > 0:
                logger.info("Applying KNN Imputation and StandardScaler on train fold...")
                # Fit on train
                train_data = df_fold.iloc[train_idx][valid_cols]
                imputer.fit(train_data)

                # Transform all (for sequence extraction continuity)
                df_fold[valid_cols] = imputer.transform(df_fold[valid_cols])

                # Scale
                scaler.fit(df_fold.iloc[train_idx][valid_cols])
                df_fold[valid_cols] = scaler.transform(df_fold[valid_cols])

            for c in numeric_cols:
                if c not in valid_cols:
                    df_fold[c] = 0.0

            fold_processors.append((imputer, scaler, valid_cols))

            data_module = DaconDataModule(
                full_df=df_fold,
                train_idx=train_idx,
                val_idx=val_idx,
                feature_cols=numeric_cols,
                batch_size=self.cfg.model.get("batch_size", 64),
                use_sequence=use_sequence,
                seq_len=seq_len,
            )

            if use_sequence:
                if model_name == "dl_transformer":
                    from src.models.dl_models import MultiTaskTransformer

                    model = MultiTaskTransformer(
                        input_dim=len(numeric_cols),
                        hidden_dim=self.cfg.model.get("hidden_dim", 128),
                        num_layers=self.cfg.model.get("num_layers", 2),
                        lr=self.cfg.model.get("lr", 1e-3),
                    )
                    model_class = MultiTaskTransformer
                else:
                    from src.models.dl_models import MultiTaskLSTM

                    model = MultiTaskLSTM(
                        input_dim=len(numeric_cols),
                        hidden_dim=self.cfg.model.get("hidden_dim", 128),
                        num_layers=self.cfg.model.get("num_layers", 2),
                        lr=self.cfg.model.get("lr", 1e-3),
                    )
                    model_class = MultiTaskLSTM
            else:
                from src.models.dl_models import MultiTaskMLP

                model = MultiTaskMLP(
                    input_dim=len(numeric_cols),
                    hidden_dims=self.cfg.model.get("hidden_dims", [256, 128, 64]),
                    lr=self.cfg.model.get("lr", 1e-3),
                )
                model_class = MultiTaskMLP

            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(self.model_dir, f"fold_{fold}"),
                filename="best-checkpoint",
                save_top_k=1,
                verbose=False,
                monitor="val_logloss_mean",
                mode="min",
            )

            early_stop_callback = EarlyStopping(monitor="val_logloss_mean", patience=7, mode="min")

            tb_logger = (
                WandbLogger(project=self.cfg.training.get("wandb_project", "dacon-etri")) if self.use_wandb else True
            )

            logger.info(f"Setting max_epochs to: {self.cfg.model.get('epochs', 50)}")
            trainer = pl.Trainer(
                max_epochs=self.cfg.model.get("epochs", 50),
                callbacks=[checkpoint_callback, early_stop_callback],
                logger=tb_logger,
                accelerator="auto",
                devices=1,
                enable_progress_bar=False,
            )
            logger.info(f"Trainer max_epochs is actually: {trainer.max_epochs}")

            trainer.fit(model, data_module)

            # Load best model for OOF prediction
            best_model = model_class.load_from_checkpoint(checkpoint_callback.best_model_path)
            best_model.eval()
            trained_models.append(checkpoint_callback.best_model_path)

            # Predict OOF
            # Setup val dataloader
            data_module.setup()
            val_loader = data_module.val_dataloader()

            fold_preds = {t: [] for t in self.targets}

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 5:
                        x, _, _, padding_mask, user_id = batch
                        padding_mask = padding_mask.to(best_model.device)
                        user_id = user_id.to(best_model.device)
                    elif len(batch) == 4:
                        x, _, _, padding_mask = batch
                        padding_mask = padding_mask.to(best_model.device)
                        user_id = None
                    else:
                        x, _, _ = batch
                        padding_mask = None
                        user_id = None

                    x = x.to(best_model.device)
                    logits = best_model(x, padding_mask, user_id)
                    for t in self.targets:
                        fold_preds[t].extend(torch.sigmoid(logits[t]).cpu().numpy())

            # Save User Embeddings for ML models
            try:
                if hasattr(best_model, "user_embedding"):
                    run_name = self.cfg.get("training", {}).get("run_name", "dl_baseline")
                    emb_weights = best_model.user_embedding.weight.detach().cpu().numpy()
                    emb_df = pd.DataFrame(emb_weights, columns=[f"user_emb_{i}" for i in range(emb_weights.shape[1])])
                    # We need to map back to original subject_ids.
                    # Our data module does: user_codes = df[id_col].str.extract(r'(\d+)')[0].astype(int).values - 1
                    # So row i corresponds to id{i+1:02d}
                    emb_df["subject_id"] = [f"id{i + 1:02d}" for i in range(len(emb_df))]

                    emb_path = os.path.join(self.output_dir, f"user_embeddings_{run_name}.csv")
                    emb_df.to_csv(emb_path, index=False)
                    logger.info(f"Saved user embeddings to {emb_path}")
            except Exception as e:
                logger.warning(f"Failed to save user embeddings: {e}")

            for t in self.targets:
                oof_preds[t][val_idx] = np.array(fold_preds[t])

        # Calculate OOF Log-Loss
        oof_scores = []
        for t in self.targets:
            mask_t = df_train[t].notna() & has_prediction
            if mask_t.sum() > 0:
                y_true = df_train[t][mask_t].values
                y_pred = oof_preds[t][mask_t]
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                t_score = log_loss(y_true, y_pred)
                oof_scores.append(t_score)

        avg_score = np.mean(oof_scores) if len(oof_scores) > 0 else 0.0
        logger.info(f"Overall DL CV Score (Average Log-Loss): {avg_score:.4f}")

        # Save OOF predictions
        try:
            oof_df = df_train[[id_col, "date"]].copy() if "date" in df_train.columns else df_train[[id_col]].copy()
            for t in self.targets:
                oof_df[t] = oof_preds[t]

            os.makedirs(self.output_dir, exist_ok=True)
            run_name = self.cfg.get("training", {}).get("run_name", "dl_baseline")
            oof_path = os.path.join(self.output_dir, f"oof_{run_name}_score{avg_score:.4f}.csv")
            oof_df.to_csv(oof_path, index=False)
            logger.info(f"OOF predictions saved to {oof_path}")
        except Exception as e:
            logger.warning(f"Failed to save OOF predictions: {e}")

        # 4. Predict on Test using Fold-Ensembled Inference
        data_loader = DataLoader(data_dir=self.data_dir)
        sample_sub_df = data_loader.load_sample_submission()
        if sample_sub_df is not None and self.df_test_base is not None:
            logger.info("Predicting on test set using Fold-Ensembled Inference (Average of 5 regularized models)...")

            # Setup KNN Imputation and Scaler based on 100% training data to be strictly leak-free.
            full_imputer = KNNImputer(n_neighbors=5)
            full_scaler = StandardScaler()

            df_train_full = df_train.copy()
            valid_cols_full = []
            for col in numeric_cols:
                missing_ratio = df_train_full[col].isna().sum() / len(df_train_full)
                if missing_ratio < 0.95:  # Only impute/scale if missing < 95%
                    valid_cols_full.append(col)

            if len(valid_cols_full) > 0:
                logger.info("Applying KNN Imputation and StandardScaler on 100% train data...")
                df_train_full[valid_cols_full] = full_imputer.fit_transform(df_train_full[valid_cols_full])
                df_train_full[valid_cols_full] = full_scaler.fit_transform(df_train_full[valid_cols_full])

            # Fill remaining columns with 0.0
            for c in numeric_cols:
                if c not in valid_cols_full:
                    df_train_full[c] = 0.0

            # Transform Test Set using fitted imputer/scaler
            df_test_full = self.df_test_base.copy()
            for col in numeric_cols:
                if col not in df_test_full.columns:
                    df_test_full[col] = np.nan

            if len(valid_cols_full) > 0:
                df_test_full[valid_cols_full] = full_imputer.transform(df_test_full[valid_cols_full])
                df_test_full[valid_cols_full] = full_scaler.transform(df_test_full[valid_cols_full])

            for c in numeric_cols:
                if c not in valid_cols_full:
                    df_test_full[c] = 0.0

            if use_sequence:
                from src.data.dl_data_module import MultiTaskSequenceDataset

                test_dataset = MultiTaskSequenceDataset(df_test_full, numeric_cols, self.targets, seq_len=seq_len)
            else:
                from src.data.dl_data_module import MultiTaskDataset

                test_dataset = MultiTaskDataset(df_test_full, numeric_cols, self.targets)

            import sys

            from torch.utils.data import DataLoader as TorchDataLoader

            num_workers = 0 if sys.platform == "win32" or torch.backends.mps.is_available() else 4
            test_loader = TorchDataLoader(
                test_dataset, batch_size=self.cfg.model.get("batch_size", 64), shuffle=False, num_workers=num_workers
            )

            # Aggregate predictions over all 5 folds
            submission = sample_sub_df.copy()
            for t in self.targets:
                submission[t] = 0.0

            ensembled_test_preds = {t: np.zeros(len(df_test_full)) for t in self.targets}

            for fold, model_path in enumerate(trained_models):
                logger.info(f"Inference on test using Fold {fold} checkpoint: {model_path}")
                fold_model = model_class.load_from_checkpoint(model_path)
                fold_model.eval()

                fold_test_preds = {t: [] for t in self.targets}
                with torch.no_grad():
                    for batch in test_loader:
                        if use_sequence:
                            x, _, _, padding_mask, user_codes = batch
                            x = x.to(fold_model.device)
                            padding_mask = padding_mask.to(fold_model.device)
                            user_codes = user_codes.to(fold_model.device)
                            logits = fold_model(x, padding_mask, user_codes)
                        else:
                            x, _, _ = batch
                            x = x.to(fold_model.device)
                            logits = fold_model(x)

                        for t in self.targets:
                            fold_test_preds[t].extend(torch.sigmoid(logits[t]).cpu().numpy())

                for t in self.targets:
                    ensembled_test_preds[t] += np.array(fold_test_preds[t]) / len(trained_models)

            for t in self.targets:
                submission[t] = ensembled_test_preds[t]

            submission = clip_probabilities(submission, self.targets)
            if not validate_submission(submission, sample_sub_df):
                logger.warning("Submission validation failed. Please check the logs. Proceeding anyway.")

            sub_filename = f"submission_{run_name}_score{avg_score:.4f}.csv"
            sub_path = os.path.join(self.output_dir, sub_filename)
            submission.to_csv(sub_path, index=False)
            logger.info(f"Submission saved to {sub_path}")

            # Save metadata config alongside for reproducibility
            import json

            from omegaconf import OmegaConf

            meta_path = os.path.join(self.output_dir, f"submission_{run_name}_score{avg_score:.4f}_meta.json")
            with open(meta_path, "w") as f:
                json.dump(OmegaConf.to_container(self.cfg, resolve=True), f, indent=4)

        logger.info("Deep Learning Pipeline Completed.")
        return avg_score
