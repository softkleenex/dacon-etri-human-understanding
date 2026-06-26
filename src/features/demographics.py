"""Official static subject demographics from the ETRI 2024 paper.

The DACON challenge provides a paper describing the same 10 subjects used in
the 2025/2026 files. These features are static by subject and can be merged into
train and test without row-level target leakage.
"""

from __future__ import annotations

import pandas as pd

DEMOGRAPHIC_FEATURE_COLUMNS = [
    "demo_gender_female",
    "demo_gender_male",
    "demo_age_decade",
    "demo_age_ge_40",
    "demo_employed",
    "demo_bmi",
    "demo_bmi_underweight",
    "demo_bmi_normal",
    "demo_bmi_overweight",
    "demo_bmi_obese",
]


def official_demographics_frame() -> pd.DataFrame:
    """Return numeric features for id01-id10 from the official ETRI paper table."""
    rows = [
        ("id01", "female", 40, 1, 23.7),
        ("id02", "male", 50, 1, 22.7),
        ("id03", "female", 30, 0, 29.1),
        ("id04", "female", 40, 0, 19.6),
        ("id05", "female", 50, 1, 26.3),
        ("id06", "male", 20, 1, 27.1),
        ("id07", "female", 40, 1, 26.3),
        ("id08", "male", 30, 1, 29.3),
        ("id09", "male", 30, 1, 25.4),
        ("id10", "female", 30, 1, 26.0),
    ]
    demo = pd.DataFrame(rows, columns=["subject_id", "gender", "age_decade", "employed", "bmi"])
    demo["demo_gender_female"] = (demo["gender"] == "female").astype(int)
    demo["demo_gender_male"] = (demo["gender"] == "male").astype(int)
    demo["demo_age_decade"] = demo["age_decade"].astype(float)
    demo["demo_age_ge_40"] = (demo["age_decade"] >= 40).astype(int)
    demo["demo_employed"] = demo["employed"].astype(int)
    demo["demo_bmi"] = demo["bmi"].astype(float)
    demo["demo_bmi_underweight"] = (demo["bmi"] < 18.5).astype(int)
    demo["demo_bmi_normal"] = demo["bmi"].between(18.5, 24.9, inclusive="both").astype(int)
    demo["demo_bmi_overweight"] = demo["bmi"].between(25.0, 29.9, inclusive="both").astype(int)
    demo["demo_bmi_obese"] = (demo["bmi"] >= 30.0).astype(int)
    return demo[["subject_id", *DEMOGRAPHIC_FEATURE_COLUMNS]]


def add_official_demographics(df: pd.DataFrame, id_col: str = "subject_id") -> pd.DataFrame:
    """Merge official paper demographics into a dataframe when subject ids are available."""
    if id_col not in df.columns:
        return df.copy()

    out = df.copy()
    out = out.drop(columns=[col for col in DEMOGRAPHIC_FEATURE_COLUMNS if col in out.columns])
    demo = official_demographics_frame().rename(columns={"subject_id": id_col})
    out[id_col] = out[id_col].astype(str)
    return out.merge(demo, on=id_col, how="left")
