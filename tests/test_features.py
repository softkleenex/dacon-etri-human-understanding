import numpy as np
import pandas as pd

from src.features.build_features import (
    aggregate_time_series,
    apply_personalized_normalization,
    engineer_features,
    handle_missing_values,
)


def test_aggregate_time_series():
    df = pd.DataFrame(
        {"time": ["2024-01-01 10:00:00", "2024-01-01 10:05:00", "2024-01-02 10:00:00"], "val": [1.0, 3.0, 5.0]}
    )
    df_agg = aggregate_time_series(df, time_col="time", agg_col="val", freq="D")
    assert len(df_agg) == 2
    assert df_agg.iloc[0]["val"] == 2.0  # Mean of 1.0 and 3.0
    assert df_agg.iloc[1]["val"] == 5.0


def test_aggregate_time_series_no_time_col():
    df = pd.DataFrame({"val": [1.0, 3.0, 5.0]})
    df_agg = aggregate_time_series(df, time_col="time", agg_col="val", freq="D")
    assert len(df_agg) == 3


def test_handle_missing_values():
    df = pd.DataFrame({"val1": [1.0, np.nan, 3.0], "val2": [np.nan, 2.0, np.nan]})
    df_clean = handle_missing_values(df)
    assert not df_clean.isnull().any().any()
    assert "val1_is_missing" in df_clean.columns
    assert df_clean["val1_is_missing"].tolist() == [0, 1, 0]


def test_apply_personalized_normalization():
    df = pd.DataFrame({"user_id": [1, 1, 2, 2], "val": [10.0, 20.0, 100.0, 200.0]})
    df_norm = apply_personalized_normalization(df, "user_id", ["val"])
    assert "val_zscore" in df_norm.columns
    # Check zero mean
    assert np.isclose(df_norm.loc[df_norm["user_id"] == 1, "val_zscore"].mean(), 0)
    assert np.isclose(df_norm.loc[df_norm["user_id"] == 2, "val_zscore"].mean(), 0)


def test_engineer_features():
    df = pd.DataFrame({"user_id": [1, 1], "val": [1.0, np.nan], "mAmbience": ["['sound1', 'sound2']", "[]"]})
    df_engineered = engineer_features(df)
    assert not df_engineered.isnull().any().any()
    assert "val_is_missing" in df_engineered.columns
    assert "mAmbience_len" in df_engineered.columns
    assert "val_zscore" in df_engineered.columns
