import pandas as pd

from src.features.demographics import (
    DEMOGRAPHIC_FEATURE_COLUMNS,
    add_official_demographics,
    official_demographics_frame,
)


def test_official_demographics_frame_contains_all_subjects():
    demo = official_demographics_frame()

    assert demo["subject_id"].tolist() == [f"id{i:02d}" for i in range(1, 11)]
    assert set(DEMOGRAPHIC_FEATURE_COLUMNS).issubset(demo.columns)
    assert demo.loc[demo["subject_id"] == "id02", "demo_gender_male"].item() == 1
    assert demo.loc[demo["subject_id"] == "id03", "demo_employed"].item() == 0
    assert demo.loc[demo["subject_id"] == "id08", "demo_bmi"].item() == 29.3


def test_add_official_demographics_merges_without_reordering_rows():
    df = pd.DataFrame({"subject_id": ["id03", "id01", "unknown"], "value": [1, 2, 3]})

    out = add_official_demographics(df)

    assert out["subject_id"].tolist() == ["id03", "id01", "unknown"]
    assert out.loc[0, "demo_gender_female"] == 1
    assert out.loc[1, "demo_age_ge_40"] == 1
    assert pd.isna(out.loc[2, "demo_bmi"])


def test_add_official_demographics_is_idempotent():
    df = pd.DataFrame({"subject_id": ["id01", "id02"], "value": [1, 2]})

    once = add_official_demographics(df)
    twice = add_official_demographics(once)

    assert set(DEMOGRAPHIC_FEATURE_COLUMNS).issubset(twice.columns)
    assert not any(col.endswith(("_x", "_y")) for col in twice.columns)
    assert twice[["subject_id", "value", *DEMOGRAPHIC_FEATURE_COLUMNS]].equals(
        once[["subject_id", "value", *DEMOGRAPHIC_FEATURE_COLUMNS]]
    )
