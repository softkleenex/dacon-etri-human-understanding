import numpy as np
import pandas as pd

from src.features.shutdown import anchor_lifelog_night, safe_entropy, window_mask


def test_anchor_lifelog_night_maps_after_midnight_to_previous_date():
    df = pd.DataFrame(
        {
            "subject_id": ["id01", "id01", "id01"],
            "timestamp": pd.to_datetime(["2024-07-01 21:30:00", "2024-07-02 01:15:00", "2024-07-02 13:00:00"]),
            "value": [1, 2, 3],
        }
    )

    out = anchor_lifelog_night(df)

    assert len(out) == 2
    assert out["anchor_date"].dt.date.tolist() == [pd.Timestamp("2024-07-01").date()] * 2
    assert out["rel_hour"].round(2).tolist() == [21.5, 25.25]


def test_window_mask_and_entropy_are_stable():
    rel_hour = np.array([21.0, 22.5, 25.0, 31.0])

    assert window_mask(rel_hour, "pre22_24").tolist() == [False, True, False, False]
    assert safe_entropy([1.0, 1.0]) > safe_entropy([2.0, 0.0])
