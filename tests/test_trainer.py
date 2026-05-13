import os
import numpy as np
import pandas as pd
from unittest.mock import patch

from omegaconf import OmegaConf

from src.models.trainer import Trainer, evaluate_model


def test_trainer_initialization(tmp_path):
    cfg = OmegaConf.create(
        {
            "env": "local",
            "paths": {
                "local": {
                    "data_dir": str(tmp_path / "data"),
                    "model_dir": str(tmp_path / "models"),
                }
            },
            "model": {"name": "lightgbm", "params": {"n_estimators": 2}},
            "training": {"use_wandb": False},
        }
    )

    trainer = Trainer(cfg)
    assert trainer.model_type == "lightgbm"
    assert not trainer.use_wandb


def test_trainer_run_dummy_data(tmp_path):
    cfg = OmegaConf.create(
        {
            "env": "local",
            "paths": {
                "local": {
                    "data_dir": str(tmp_path / "data"),
                    "model_dir": str(tmp_path / "models"),
                }
            },
            "model": {"name": "lightgbm", "params": {"n_estimators": 2}},
            "training": {"use_wandb": False},
        }
    )

    trainer = Trainer(cfg)
    trainer.run()

    # Check if models were saved
    assert os.path.exists(trainer.model_dir)
    saved_files = os.listdir(trainer.model_dir)
    # We expect models for Q1~Q3, S1~S4 to be saved
    assert any("Q1" in f for f in saved_files)
    assert any("S1" in f for f in saved_files)

def test_evaluate_model():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_prob_binary = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    y_prob_single = np.array([0.1, 0.9, 0.2, 0.6])
    
    # without prob
    res1 = evaluate_model(y_true, y_pred)
    assert "macro_f1" in res1
    assert "auc" not in res1
    
    # with prob binary (shape > 1, shape[1] == 2)
    res2 = evaluate_model(y_true, y_pred, y_prob_binary)
    assert "auc" in res2
    
    # with prob single
    res2_1 = evaluate_model(y_true, y_pred, y_prob_single)
    assert "auc" in res2_1
    
    # multi-class
    y_true_multi = np.array([0, 1, 2, 0, 1, 2])
    y_prob_multi = np.array([
        [0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]
    ])
    res3 = evaluate_model(y_true_multi, y_true_multi, y_prob_multi)
    assert "auc" in res3

    # Exception in AUC
    res4 = evaluate_model(y_true, y_pred, np.array(["invalid", "data"]))
    assert "macro_f1" in res4
    assert "auc" not in res4


@patch("src.models.trainer.wandb")
def test_trainer_run_with_metrics_and_wandb(mock_wandb, tmp_path):
    # Setup dummy data so that DataLoader finds something
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    items_dir = raw_dir / "ch2025_data_items"
    items_dir.mkdir(parents=True)
    
    df_metrics = pd.DataFrame({
        "subject_id": [1, 2],
        "date": ["2025-01-01", "2025-01-02"],
        "Q1": [0, 1], "Q2": [1, 0], "Q3": [0, 1],
        "S1": [1, 0], "S2": [0, 1], "S3": [1, 0], "S4": [0, 1]
    })
    df_metrics.to_csv(raw_dir / "ch2026_metrics_train.csv", index=False)
    
    cfg = OmegaConf.create({
        "env": "local",
        "paths": {
            "local": {
                "data_dir": str(data_dir),
                "model_dir": str(tmp_path / "models"),
            }
        },
        "model": {"name": "lightgbm", "params": {"n_estimators": 2}},
        "training": {"use_wandb": True},
    })
    
    trainer = Trainer(cfg)
    trainer.run()
    
    mock_wandb.init.assert_called_once()
    mock_wandb.log.assert_called()
    mock_wandb.finish.assert_called_once()
