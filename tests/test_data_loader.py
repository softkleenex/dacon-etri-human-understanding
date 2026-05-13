from pathlib import Path

import pandas as pd

from src.data.data_loader import DataLoader


def test_data_loader_initialization():
    loader = DataLoader(data_dir="dummy_dir")
    assert loader.data_dir == Path("dummy_dir")
    assert loader.raw_dir == Path("dummy_dir/raw")
    assert loader.processed_dir == Path("dummy_dir/processed")
    assert loader.items_dir == Path("dummy_dir/raw/ch2025_data_items")


def test_load_metrics_empty(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    loader = DataLoader(data_dir=str(data_dir))
    df = loader.load_metrics()

    assert df is None


def test_load_metrics_success(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    df_expected = pd.DataFrame({"target": [0, 1]})
    df_expected.to_csv(raw_dir / "ch2026_metrics_train.csv", index=False)

    loader = DataLoader(data_dir=str(data_dir))
    df = loader.load_metrics()

    pd.testing.assert_frame_equal(df, df_expected)


def test_load_sensor_data_empty_dir(tmp_path):
    data_dir = tmp_path / "data"
    items_dir = data_dir / "raw" / "ch2025_data_items"
    items_dir.mkdir(parents=True)

    loader = DataLoader(data_dir=str(data_dir))
    data_dict = loader.load_sensor_data()

    assert isinstance(data_dict, dict)
    assert len(data_dict) == 0


def test_load_sensor_data_success(tmp_path):
    data_dir = tmp_path / "data"
    items_dir = data_dir / "raw" / "ch2025_data_items"
    items_dir.mkdir(parents=True)

    df1 = pd.DataFrame({"colA": ["a", "b"]})
    df1.to_parquet(items_dir / "ch2025_mACStatus.parquet")

    loader = DataLoader(data_dir=str(data_dir))
    data_dict = loader.load_sensor_data()

    assert len(data_dict) == 1
    assert "ch2025_mACStatus" in data_dict
    pd.testing.assert_frame_equal(data_dict["ch2025_mACStatus"], df1)


def test_load_sample_submission_empty(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    loader = DataLoader(data_dir=str(data_dir))
    df = loader.load_sample_submission()

    assert df is None


def test_load_sample_submission_success(tmp_path):
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)

    df_expected = pd.DataFrame({"submission": [0, 1]})
    df_expected.to_csv(raw_dir / "ch2026_submission_sample.csv", index=False)

    loader = DataLoader(data_dir=str(data_dir))
    df = loader.load_sample_submission()

    pd.testing.assert_frame_equal(df, df_expected)
