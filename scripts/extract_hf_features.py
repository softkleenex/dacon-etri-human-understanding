import sys

sys.path.append('.')
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline

from src.data.data_loader import DataLoader
from src.data.make_dataset import build_training_dataset
from src.features.build_features import engineer_features
from src.utils.common import get_kaggle_paths


def main():
    print("🚀 Extracting Zero-Shot Time Series Features using Chronos-T5")

    # 1. Load Data
    is_kaggle, data_dir, conf_dir = get_kaggle_paths()

    loader = DataLoader(data_dir=data_dir)
    train_metrics = loader.load_metrics()
    sample_sub = loader.load_sample_submission()
    sensors = loader.load_sensor_data()

    df_train = build_training_dataset(train_metrics, sensors)
    df_test = build_training_dataset(sample_sub, sensors)

    df_train['is_test'] = 0
    df_test['is_test'] = 1
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all = engineer_features(df_all)

    # 2. Select columns to embed
    target_cols = [
        'ch2025_wHr_hr_mean_mean',
        'ch2025_mActivity_m_activity_mean',
        'ch2025_mLight_light_mean_mean'
    ]

    valid_cols = [c for c in target_cols if c in df_all.columns]
    if not valid_cols:
        print("Required columns not found in dataset!")
        sys.exit(1)

    id_col = 'subject_id' if 'subject_id' in df_all.columns else 'user_id'

    # 3. Load Pipeline
    device = "cpu"
    print(f"Loading Chronos-T5-mini on {device}...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-mini",
        device_map=device,
        torch_dtype=torch.float32
    )

    # 4. Extract
    embeddings_list = []

    print("Extracting embeddings per user...")
    for subject_id, group in df_all.sort_values([id_col, 'date']).groupby(id_col):
        for col in valid_cols:
            series = group[col].fillna(0).values
            context = torch.tensor(series, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                emb, _ = pipeline.embed(context)

            # Chronos outputs context_len + prediction_len.
            # Usually predict_len defaults to 64 or something, but context length is what we need.
            context_len = context.shape[1]
            emb = emb[0, :context_len, :].cpu().numpy()

            # Reduce dimension
            hidden_dim = emb.shape[1]
            reduced_emb = np.zeros((emb.shape[0], 16))
            chunk_size = hidden_dim // 16
            for i in range(16):
                reduced_emb[:, i] = np.mean(emb[:, i*chunk_size:(i+1)*chunk_size], axis=1)

            # Save
            for i in range(len(group)):
                row_dict = {
                    id_col: subject_id,
                    'date': group.iloc[i]['date'],
                    'feature_src': col
                }
                for j in range(16):
                    row_dict[f'chronos_emb_{j}'] = reduced_emb[i, j]
                embeddings_list.append(row_dict)

    emb_df = pd.DataFrame(embeddings_list)
    pivot_df = emb_df.pivot_table(index=[id_col, 'date'], columns='feature_src', values=[f'chronos_emb_{i}' for i in range(16)])
    pivot_df.columns = [f"{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    out_dir = Path('data/processed')
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / 'hf_chronos_embeddings.csv'
    pivot_df.to_csv(out_path, index=False)
    print(f"✅ Saved HF Chronos embeddings to {out_path}")

if __name__ == "__main__":
    main()
