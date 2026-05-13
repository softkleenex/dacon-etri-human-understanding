import pandas as pd
import pytest
from src.data.make_dataset import create_daily_features, build_training_dataset

def test_create_daily_features_empty():
    res = create_daily_features({})
    assert res.empty

def test_create_daily_features_df_empty():
    df = pd.DataFrame()
    res = create_daily_features({"sensor": df})
    assert res.empty


def test_create_daily_features_no_id():
    df = pd.DataFrame({"timestamp": ["2025-01-01"], "val": [1]})
    res = create_daily_features({"sensor": df})
    assert res.empty

def test_create_daily_features_success():
    df1 = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "timestamp": ["2025-01-01 10:00", "2025-01-01 11:00", "2025-01-02 10:00"],
        "val1": [10, 20, 30]
    })
    df2 = pd.DataFrame({
        "user_id": [1, 2],
        "timestamp": ["2025-01-01 10:30", "2025-01-02 10:30"],
        "val2": [5, 15]
    })
    
    res = create_daily_features({"sensor1": df1, "sensor2": df2})
    assert not res.empty
    assert "sensor1_val1_mean" in res.columns
    assert "sensor2_val2_mean" in res.columns
    assert len(res) == 2 # user 1 on day 1, user 2 on day 2

def test_build_training_dataset_no_metrics():
    with pytest.raises(ValueError):
        build_training_dataset(None, {})

def test_build_training_dataset_no_id_col_in_metrics():
    metrics = pd.DataFrame({"target": [1]})
    df1 = pd.DataFrame({"subject_id": [1], "timestamp": ["2025-01-01 10:00"], "val": [10]})
    res = build_training_dataset(metrics, {"sensor": df1})
    assert len(res) == 1
    assert "sensor_val_mean" not in res.columns


def test_build_training_dataset_empty_features():
    metrics = pd.DataFrame({"subject_id": [1], "date": ["2025-01-01"], "target": [1]})
    res = build_training_dataset(metrics, {})
    assert len(res) == 1
    assert "target" in res.columns

def test_build_training_dataset_with_features():
    metrics = pd.DataFrame({"subject_id": [1], "date": ["2025-01-01"], "target": [1]})
    df1 = pd.DataFrame({
        "subject_id": [1],
        "timestamp": ["2025-01-01 10:00"],
        "val": [10]
    })
    res = build_training_dataset(metrics, {"sensor": df1})
    assert len(res) == 1
    assert "sensor_val_mean" in res.columns

def test_build_training_dataset_no_date():
    metrics = pd.DataFrame({"subject_id": [1], "target": [1]})
    df1 = pd.DataFrame({
        "subject_id": [1],
        "timestamp": ["2025-01-01 10:00"],
        "val": [10]
    })
    res = build_training_dataset(metrics, {"sensor": df1})
    assert len(res) == 1
    assert "sensor_val_mean" in res.columns
