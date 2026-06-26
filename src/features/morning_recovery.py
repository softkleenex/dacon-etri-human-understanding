"""Morning recovery features for Q1 subjective sleep-quality prediction.

Rows are keyed by ``subject_id`` and lifelog ``date``. Because the competition
row date is the pre-sleep lifelog date, timestamps from 00:00 to 11:59 are
assigned to the previous lifelog date. A row for date ``D`` can therefore use
the post-sleep morning window from ``D+1 04:00`` through ``D+1 12:00``.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

MORNING_WINDOWS = {
    "prewake28_30": (28.0, 30.0),
    "wake30_33": (30.0, 33.0),
    "late33_36": (33.0, 36.0),
    "all28_36": (28.0, 36.0),
    "core30_36": (30.0, 36.0),
}


def anchor_lifelog_morning(df: pd.DataFrame) -> pd.DataFrame:
    """Map post-midnight morning timestamps to the previous lifelog date."""

    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hour = out["timestamp"].dt.hour
    out["anchor_date"] = out["timestamp"].dt.normalize()
    out.loc[hour < 12, "anchor_date"] = out.loc[hour < 12, "anchor_date"] - pd.Timedelta(days=1)
    out["rel_hour"] = hour.astype(float) + (hour < 12).astype(float) * 24.0 + out["timestamp"].dt.minute / 60.0
    return out[(out["rel_hour"] >= 28.0) & (out["rel_hour"] < 36.0)].copy()


def morning_window_mask(rel_hour: np.ndarray, window_name: str) -> np.ndarray:
    lo, hi = MORNING_WINDOWS[window_name]
    return (rel_hour >= lo) & (rel_hour < hi)


def mean_numeric_array(value: object) -> float:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.nan
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(arr.mean()) if arr.size else np.nan


def extract_morning_scalar_features(
    raw_dir: Path,
    filename: str,
    value_col: str,
    prefix: str,
    *,
    dark_threshold: float | None = None,
    bright_threshold: float | None = None,
) -> pd.DataFrame:
    df = anchor_lifelog_morning(pd.read_parquet(raw_dir / filename))
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel_hour = group["rel_hour"].to_numpy()
        values_all = group[value_col].to_numpy(dtype=float)
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in MORNING_WINDOWS:
            values = values_all[morning_window_mask(rel_hour, window)]
            values = values[np.isfinite(values)]
            row[f"morn_{prefix}_{window}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"morn_{prefix}_{window}_std"] = float(values.std()) if values.size else np.nan
            row[f"morn_{prefix}_{window}_max"] = float(values.max()) if values.size else np.nan
            row[f"morn_{prefix}_{window}_min"] = float(values.min()) if values.size else np.nan
            if dark_threshold is not None:
                row[f"morn_{prefix}_{window}_dark_ratio"] = (
                    float((values < dark_threshold).mean()) if values.size else np.nan
                )
            if bright_threshold is not None:
                row[f"morn_{prefix}_{window}_bright_ratio"] = (
                    float((values > bright_threshold).mean()) if values.size else np.nan
                )
        row[f"morn_{prefix}_rise_30_36_minus_28_30"] = row.get(f"morn_{prefix}_core30_36_mean", np.nan) - row.get(
            f"morn_{prefix}_prewake28_30_mean", np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def extract_morning_screen_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_morning(pd.read_parquet(raw_dir / "ch2025_mScreenStatus.parquet"))
    df["screen_on"] = (pd.to_numeric(df["m_screen_use"], errors="coerce").fillna(0) > 0).astype(float)
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel_hour = group["rel_hour"].to_numpy()
        screen_on = group["screen_on"].to_numpy(dtype=float)
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in MORNING_WINDOWS:
            values = screen_on[morning_window_mask(rel_hour, window)]
            row[f"morn_screen_{window}_ratio"] = float(values.mean()) if values.size else np.nan
            row[f"morn_screen_{window}_sum"] = float(values.sum()) if values.size else np.nan
        on_rel_hour = rel_hour[screen_on > 0]
        row["morn_screen_first_on_rel"] = float(on_rel_hour.min()) if on_rel_hour.size else 37.0
        row["morn_screen_last_on_rel"] = float(on_rel_hour.max()) if on_rel_hour.size else -1.0
        row["morn_screen_span"] = float(on_rel_hour.max() - on_rel_hour.min()) if on_rel_hour.size else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def extract_morning_activity_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_morning(pd.read_parquet(raw_dir / "ch2025_mActivity.parquet"))
    activity = pd.to_numeric(df["m_activity"], errors="coerce")
    df["still"] = (activity == 0).astype(float)
    df["active"] = activity.isin([3, 7, 8]).astype(float)
    df["walkish"] = activity.isin([7, 8]).astype(float)
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel_hour = group["rel_hour"].to_numpy()
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in MORNING_WINDOWS:
            mask = morning_window_mask(rel_hour, window)
            for col in ["still", "active", "walkish"]:
                values = group[col].to_numpy(dtype=float)[mask]
                row[f"morn_act_{window}_{col}"] = float(values.mean()) if values.size else np.nan
        active_rel_hour = rel_hour[group["active"].to_numpy(dtype=float) > 0]
        row["morn_act_first_active_rel"] = float(active_rel_hour.min()) if active_rel_hour.size else 37.0
        row["morn_act_last_active_rel"] = float(active_rel_hour.max()) if active_rel_hour.size else -1.0
        rows.append(row)
    return pd.DataFrame(rows)


def extract_morning_pedo_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_morning(pd.read_parquet(raw_dir / "ch2025_wPedo.parquet"))
    for col in ["step", "distance", "burned_calories", "speed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel_hour = group["rel_hour"].to_numpy()
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in MORNING_WINDOWS:
            subset = group.loc[morning_window_mask(rel_hour, window)]
            row[f"morn_pedo_{window}_steps"] = float(subset["step"].sum()) if len(subset) else np.nan
            row[f"morn_pedo_{window}_dist"] = float(subset["distance"].sum()) if len(subset) else np.nan
            row[f"morn_pedo_{window}_cal"] = float(subset["burned_calories"].sum()) if len(subset) else np.nan
            row[f"morn_pedo_{window}_speed_mean"] = float(subset["speed"].mean()) if len(subset) else np.nan
            row[f"morn_pedo_{window}_speed_max"] = float(subset["speed"].max()) if len(subset) else np.nan
            row[f"morn_pedo_{window}_active_records"] = (
                float((subset["step"].fillna(0) > 0).sum()) if len(subset) else np.nan
            )
        step_rel_hour = rel_hour[group["step"].fillna(0).to_numpy(dtype=float) > 0]
        row["morn_pedo_first_step_rel"] = float(step_rel_hour.min()) if step_rel_hour.size else 37.0
        row["morn_pedo_last_step_rel"] = float(step_rel_hour.max()) if step_rel_hour.size else -1.0
        rows.append(row)
    return pd.DataFrame(rows)


def extract_morning_ac_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_morning(pd.read_parquet(raw_dir / "ch2025_mACStatus.parquet"))
    df["charging"] = (pd.to_numeric(df["m_charging"], errors="coerce").fillna(0) > 0).astype(float)
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel_hour = group["rel_hour"].to_numpy()
        charging = group["charging"].to_numpy(dtype=float)
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in MORNING_WINDOWS:
            values = charging[morning_window_mask(rel_hour, window)]
            row[f"morn_ac_{window}_charging"] = float(values.mean()) if values.size else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def extract_morning_hr_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_morning(pd.read_parquet(raw_dir / "ch2025_wHr.parquet"))
    df["hr_mean"] = df["heart_rate"].map(mean_numeric_array)
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel_hour = group["rel_hour"].to_numpy()
        values_all = group["hr_mean"].to_numpy(dtype=float)
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in MORNING_WINDOWS:
            values = values_all[morning_window_mask(rel_hour, window)]
            values = values[np.isfinite(values)]
            row[f"morn_hr_{window}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"morn_hr_{window}_std"] = float(values.std()) if values.size else np.nan
            row[f"morn_hr_{window}_min"] = float(values.min()) if values.size else np.nan
            row[f"morn_hr_{window}_max"] = float(values.max()) if values.size else np.nan
            row[f"morn_hr_{window}_high75"] = float((values > 75).mean()) if values.size else np.nan
            row[f"morn_hr_{window}_low55"] = float((values < 55).mean()) if values.size else np.nan
        row["morn_hr_rise_30_36_minus_28_30"] = row.get("morn_hr_core30_36_mean", np.nan) - row.get(
            "morn_hr_prewake28_30_mean", np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_morning_recovery_features(raw_dir: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    feature_frames = [
        extract_morning_screen_features(raw_path),
        extract_morning_activity_features(raw_path),
        extract_morning_pedo_features(raw_path),
        extract_morning_ac_features(raw_path),
        extract_morning_hr_features(raw_path),
        extract_morning_scalar_features(
            raw_path,
            "ch2025_mLight.parquet",
            "m_light",
            "mlight",
            dark_threshold=10.0,
            bright_threshold=1000.0,
        ),
        extract_morning_scalar_features(
            raw_path,
            "ch2025_wLight.parquet",
            "w_light",
            "wlight",
            dark_threshold=10.0,
            bright_threshold=1000.0,
        ),
    ]
    merged = reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), feature_frames)
    return merged.sort_values(["subject_id", "date"]).reset_index(drop=True)


def morning_recovery_feature_columns(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    return [
        col
        for col in train.columns
        if col.startswith("morn_")
        and pd.api.types.is_numeric_dtype(train[col])
        and (train[col].notna().any() or test[col].notna().any())
    ]
