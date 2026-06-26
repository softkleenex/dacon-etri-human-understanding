import numpy as np
import pandas as pd

from src.features.q_context import add_interaction_features, dateize, in_window, safe_entropy


def test_dateize_adds_date_and_fractional_hour_without_reordering():
    df = pd.DataFrame(
        {
            "subject_id": ["id01", "id02"],
            "timestamp": ["2024-06-26 12:30:00", "2024-06-27 23:45:00"],
        }
    )

    out = dateize(df)

    assert out["subject_id"].tolist() == ["id01", "id02"]
    assert out["date"].tolist() == [pd.Timestamp("2024-06-26").date(), pd.Timestamp("2024-06-27").date()]
    assert out["hour_float"].tolist() == [12.5, 23.75]


def test_in_window_uses_left_closed_right_open_interval():
    hours = np.array([11.99, 12.0, 18.0, 23.99, 24.0])

    assert in_window(hours, 12.0, 24.0).tolist() == [False, True, True, True, False]


def test_safe_entropy_handles_empty_and_peaked_values():
    assert safe_entropy([]) == 0.0
    assert safe_entropy([0.0, 0.0]) == 0.0
    assert safe_entropy([1.0, 1.0]) > safe_entropy([2.0, 0.0])


def test_add_interaction_features_only_adds_available_products():
    df = pd.DataFrame(
        {
            "strain_gps_late22_24_moving": [0.5],
            "strain_usage_late22_24_total_min": [20.0],
            "strain_wifi_late_home_absent": [1.0],
        }
    )

    out = add_interaction_features(df)

    assert out.loc[0, "strain_inter_late_move_usage"] == 10.0
    assert "strain_inter_away_comm" not in out.columns
