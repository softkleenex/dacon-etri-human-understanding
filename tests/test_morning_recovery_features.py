import numpy as np
import pandas as pd

from src.features.morning_recovery import (
    anchor_lifelog_morning,
    mean_numeric_array,
    morning_recovery_feature_columns,
    morning_window_mask,
)


def test_anchor_lifelog_morning_maps_morning_to_previous_lifelog_date():
    df = pd.DataFrame(
        {
            "subject_id": ["id01", "id01", "id01", "id01"],
            "timestamp": pd.to_datetime(
                [
                    "2024-07-02 03:59:00",
                    "2024-07-02 04:00:00",
                    "2024-07-02 06:30:00",
                    "2024-07-02 12:00:00",
                ]
            ),
            "value": [0, 1, 2, 3],
        }
    )

    out = anchor_lifelog_morning(df)

    assert len(out) == 2
    assert out["anchor_date"].dt.date.tolist() == [pd.Timestamp("2024-07-01").date()] * 2
    assert out["rel_hour"].round(2).tolist() == [28.0, 30.5]


def test_morning_window_mask_uses_left_closed_right_open_bounds():
    rel_hour = np.array([27.99, 28.0, 30.0, 35.99, 36.0])

    assert morning_window_mask(rel_hour, "all28_36").tolist() == [False, True, True, True, False]
    assert morning_window_mask(rel_hour, "prewake28_30").tolist() == [False, True, False, False, False]


def test_mean_numeric_array_ignores_nonpositive_and_invalid_values():
    assert np.isnan(mean_numeric_array([]))
    assert mean_numeric_array([0, 60, 80]) == 70.0
    assert np.isnan(mean_numeric_array("not-an-array"))


def test_morning_recovery_feature_columns_keeps_numeric_morning_columns_only():
    train = pd.DataFrame(
        {
            "morn_hr_mean": [70.0, np.nan],
            "morn_text": ["a", "b"],
            "raw_value": [1.0, 2.0],
            "morn_all_missing": [np.nan, np.nan],
        }
    )
    test = pd.DataFrame(
        {
            "morn_hr_mean": [72.0],
            "morn_text": ["c"],
            "raw_value": [3.0],
            "morn_all_missing": [np.nan],
        }
    )

    assert morning_recovery_feature_columns(train, test) == ["morn_hr_mean"]
