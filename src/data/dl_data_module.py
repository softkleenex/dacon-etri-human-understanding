import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

# Use 0 workers on macOS/Windows to avoid spawn issues, 4 on Linux/Kaggle
NUM_WORKERS = 0 if os.name == "nt" or torch.backends.mps.is_available() else 4


class MultiTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_cols: list):
        self.x = df[feature_cols].values.astype(np.float32)

        # Some targets might be missing (NaN). We need to create a mask for valid targets.
        y_data = []
        mask_data = []

        for col in target_cols:
            if col in df.columns:
                # 1 if valid, 0 if NaN
                valid_mask = ~df[col].isna().values
                mask_data.append(valid_mask.astype(np.float32))

                # Fill NaN with 0 for now (loss will be masked out anyway)
                vals = df[col].fillna(0).values
                y_data.append(vals.astype(np.float32))
            else:
                # If target is completely missing (e.g., in test set)
                mask_data.append(np.zeros(len(df), dtype=np.float32))
                y_data.append(np.zeros(len(df), dtype=np.float32))

        self.y = np.stack(y_data, axis=1)  # shape: (N, 7)
        self.mask = np.stack(mask_data, axis=1)  # shape: (N, 7)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx]), torch.tensor(self.mask[idx])


class MultiTaskSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_cols: list, seq_len: int = 7):
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_cols = target_cols

        # We need to group by user to create valid sequences
        id_col = "subject_id" if "subject_id" in df.columns else "user_id"

        # STRICT SORTING: Guarantee chronological order before sequence generation
        if "date" in df.columns:
            df = df.sort_values(by=[id_col, "date"]).reset_index(drop=True)

        self.x_seqs = []
        self.y_seqs = []
        self.mask_seqs = []

        # Precompute values
        features = df[feature_cols].values.astype(np.float32)

        y_dict = {}
        mask_dict = {}
        for col in target_cols:
            if col in df.columns:
                valid_mask = ~df[col].isna().values
                mask_dict[col] = valid_mask.astype(np.float32)
                y_dict[col] = df[col].fillna(0).values.astype(np.float32)
            else:
                mask_dict[col] = np.zeros(len(df), dtype=np.float32)
                y_dict[col] = np.zeros(len(df), dtype=np.float32)

        # To maintain the 1-to-1 mapping with the original df for test set ordering,
        # we generate sequences ending at each row in the dataframe.
        # If a user doesn't have enough history, we pad with zeros.

        df_reset = df.reset_index()

        if id_col in df.columns:
            groups = df_reset.groupby(id_col, sort=False)  # sort=False because we already sorted
            user_indices = groups.indices
        else:
            user_indices = {"all": np.arange(len(df))}

        # We want x_seqs to be same length as df so predictions align perfectly
        self.x_seqs = np.zeros((len(df), seq_len, len(feature_cols)), dtype=np.float32)
        self.y_seqs = np.zeros((len(df), len(target_cols)), dtype=np.float32)
        self.mask_seqs = np.zeros((len(df), len(target_cols)), dtype=np.float32)
        self.padding_mask_seqs = np.ones((len(df), seq_len), dtype=bool)  # True means valid data, False means padded

        # User ID encoding
        if id_col in df.columns:
            # Safe encoding: Extract integer from 'id01' format.
            try:
                user_codes = df[id_col].str.extract(r"(\d+)")[0].astype(int).values - 1
            except Exception:
                user_codes = df[id_col].astype("category").cat.codes.values
        else:
            user_codes = np.zeros(len(df), dtype=int)

        self.user_codes = user_codes

        for _user, indices in user_indices.items():
            user_features = features[indices]

            for i, global_idx in enumerate(indices):
                # sequence ends at current day (i)
                start_i = max(0, i - seq_len + 1)
                seq = user_features[start_i : i + 1]

                # pad if shorter than seq_len
                if len(seq) < seq_len:
                    pad_len = seq_len - len(seq)
                    padding = np.zeros((pad_len, len(feature_cols)), dtype=np.float32)
                    seq = np.vstack([padding, seq])
                    self.padding_mask_seqs[global_idx, :pad_len] = False  # Mark as padded

                self.x_seqs[global_idx] = seq

                for t_idx, target in enumerate(target_cols):
                    self.y_seqs[global_idx, t_idx] = y_dict[target][global_idx]
                    self.mask_seqs[global_idx, t_idx] = mask_dict[target][global_idx]

    def __len__(self):
        return len(self.x_seqs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_seqs[idx]),
            torch.tensor(self.y_seqs[idx]),
            torch.tensor(self.mask_seqs[idx]),
            torch.tensor(self.padding_mask_seqs[idx]),
            torch.tensor(self.user_codes[idx], dtype=torch.long),
        )


class DaconDataModule(pl.LightningDataModule):
    def __init__(
        self,
        full_df: pd.DataFrame,
        train_idx: list,
        val_idx: list,
        feature_cols: list,
        batch_size: int = 64,
        use_sequence: bool = False,
        seq_len: int = 7,
    ):
        super().__init__()
        self.full_df = full_df
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.feature_cols = feature_cols
        self.batch_size = batch_size
        self.use_sequence = use_sequence
        self.seq_len = seq_len
        self.targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    def setup(self, stage=None):
        from torch.utils.data import Subset

        if self.use_sequence:
            full_dataset = MultiTaskSequenceDataset(self.full_df, self.feature_cols, self.targets, seq_len=self.seq_len)
        else:
            full_dataset = MultiTaskDataset(self.full_df, self.feature_cols, self.targets)

        self.train_dataset = Subset(full_dataset, self.train_idx)
        if self.val_idx is not None and len(self.val_idx) > 0:
            self.val_dataset = Subset(full_dataset, self.val_idx)
        else:
            self.val_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if getattr(self, "val_dataset", None) is not None:
            return DataLoader(
                self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
            )
        return None
