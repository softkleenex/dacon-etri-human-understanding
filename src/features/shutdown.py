"""Evening/night shutdown trajectory features from raw sensors.

The features are keyed by ``subject_id`` and lifelog ``date``. Timestamps from
00:00 to 11:59 are assigned to the previous lifelog date, so a row for date
``D`` can summarize the night from ``D 18:00`` through ``D+1 09:00``.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = ["eve18_22", "pre22_24", "late24_26", "mid26_30", "morn30_33", "night22_33", "all18_33"]
WINDOW_BOUNDS = {
    "eve18_22": (18.0, 22.0),
    "pre22_24": (22.0, 24.0),
    "late24_26": (24.0, 26.0),
    "mid26_30": (26.0, 30.0),
    "morn30_33": (30.0, 33.0),
    "night22_33": (22.0, 33.0),
    "all18_33": (18.0, 33.0),
}


def anchor_lifelog_night(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    hour = out["timestamp"].dt.hour
    out["anchor_date"] = out["timestamp"].dt.normalize()
    out.loc[hour < 12, "anchor_date"] = out.loc[hour < 12, "anchor_date"] - pd.Timedelta(days=1)
    out["rel_hour"] = hour.astype(float) + (hour < 12).astype(float) * 24.0 + out["timestamp"].dt.minute / 60.0
    return out[(out["rel_hour"] >= 18.0) & (out["rel_hour"] < 33.0)].copy()


def window_mask(rel_hour: np.ndarray, window_name: str) -> np.ndarray:
    lo, hi = WINDOW_BOUNDS[window_name]
    return (rel_hour >= lo) & (rel_hour < hi)


def safe_entropy(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    probs = arr / arr.sum()
    return float(-(probs * np.log(probs + 1e-9)).sum())


def extract_screen_shutdown_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_night(pd.read_parquet(raw_dir / "ch2025_mScreenStatus.parquet"))
    df["screen_on"] = (df["m_screen_use"] > 0).astype(float)
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel = group["rel_hour"].to_numpy()
        on = group["screen_on"].to_numpy()
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in WINDOWS:
            mask = window_mask(rel, window)
            values = on[mask]
            row[f"shutdown_screen_{window}_ratio"] = float(values.mean()) if values.size else np.nan
            row[f"shutdown_screen_{window}_sum"] = float(values.sum()) if values.size else np.nan
        on_rel = rel[on > 0]
        row["shutdown_screen_last_on_rel"] = float(on_rel.max()) if on_rel.size else -1.0
        row["shutdown_screen_first_on_rel"] = float(on_rel.min()) if on_rel.size else -1.0
        row["shutdown_screen_on_span"] = float(on_rel.max() - on_rel.min()) if on_rel.size else 0.0
        row["shutdown_screen_unlock_transitions"] = (
            float(np.sum((on[1:] > on[:-1]) & (on[1:] > 0))) if on.size > 1 else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def extract_activity_shutdown_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_night(pd.read_parquet(raw_dir / "ch2025_mActivity.parquet"))
    df["still"] = (df["m_activity"] == 0).astype(float)
    df["active"] = df["m_activity"].isin([3, 7, 8]).astype(float)
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel = group["rel_hour"].to_numpy()
        still = group["still"].to_numpy()
        active = group["active"].to_numpy()
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in WINDOWS:
            mask = window_mask(rel, window)
            row[f"shutdown_act_{window}_still"] = float(still[mask].mean()) if mask.any() else np.nan
            row[f"shutdown_act_{window}_active"] = float(active[mask].mean()) if mask.any() else np.nan
        active_rel = rel[active > 0]
        row["shutdown_act_last_active_rel"] = float(active_rel.max()) if active_rel.size else -1.0
        rows.append(row)
    return pd.DataFrame(rows)


def extract_scalar_window_features(raw_dir: Path, filename: str, value_col: str, prefix: str) -> pd.DataFrame:
    df = anchor_lifelog_night(pd.read_parquet(raw_dir / filename))
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel = group["rel_hour"].to_numpy()
        values_all = group[value_col].to_numpy(dtype=float)
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in WINDOWS:
            mask = window_mask(rel, window)
            values = values_all[mask]
            values = values[np.isfinite(values)]
            row[f"shutdown_{prefix}_{window}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"shutdown_{prefix}_{window}_dark_ratio"] = float((values < 10).mean()) if values.size else np.nan
            row[f"shutdown_{prefix}_{window}_bright_ratio"] = float((values > 1000).mean()) if values.size else np.nan
        row[f"shutdown_{prefix}_drop_eve_to_late"] = row.get(f"shutdown_{prefix}_eve18_22_mean", np.nan) - row.get(
            f"shutdown_{prefix}_late24_26_mean", np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def mean_heart_rate(value: object) -> float:
    try:
        arr = np.asarray(value, dtype=float).ravel()
    except (TypeError, ValueError):
        return np.nan
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return float(arr.mean()) if arr.size else np.nan


def extract_hr_shutdown_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_night(pd.read_parquet(raw_dir / "ch2025_wHr.parquet"))
    df["hr_mean_row"] = df["heart_rate"].map(mean_heart_rate)
    rows: list[dict] = []
    for (subject_id, anchor_date), group in df.groupby(["subject_id", "anchor_date"], sort=False):
        rel = group["rel_hour"].to_numpy()
        values_all = group["hr_mean_row"].to_numpy(dtype=float)
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in WINDOWS:
            mask = window_mask(rel, window)
            values = values_all[mask]
            values = values[np.isfinite(values)]
            row[f"shutdown_hr_{window}_mean"] = float(values.mean()) if values.size else np.nan
            row[f"shutdown_hr_{window}_std"] = float(values.std()) if values.size else np.nan
            row[f"shutdown_hr_{window}_high_ratio"] = float((values > 75).mean()) if values.size else np.nan
            row[f"shutdown_hr_{window}_low_ratio"] = float((values < 55).mean()) if values.size else np.nan
        row["shutdown_hr_drop_eve_to_mid"] = row.get("shutdown_hr_eve18_22_mean", np.nan) - row.get(
            "shutdown_hr_mid26_30_mean", np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def categorize_usage(apps: object) -> dict[str, float]:
    categories = {
        "social": ["kakao", "카카오", "instagram", "인스타", "facebook", "twitter", "band"],
        "video": ["youtube", "유튜브", "netflix", "tiktok", "동영상"],
        "game": ["game", "게임"],
        "comm": ["전화", "통화", "message", "sms"],
        "search": ["naver", "chrome", "인터넷", "samsung internet", "검색"],
    }
    out = {"usage_total": 0.0, "usage_entropy": 0.0, "usage_n_apps": 0.0}
    out.update({f"usage_{name}": 0.0 for name in categories})
    app_times: list[float] = []
    if not isinstance(apps, (list, np.ndarray)):
        return out
    for app in apps:
        if not isinstance(app, dict):
            continue
        app_name = str(app.get("app_name", "")).lower()
        total_time = float(app.get("total_time", 0) or 0)
        out["usage_total"] += total_time
        if total_time > 0:
            app_times.append(total_time)
            out["usage_n_apps"] += 1.0
        for category, keywords in categories.items():
            if any(keyword in app_name for keyword in keywords):
                out[f"usage_{category}"] += total_time
    out["usage_entropy"] = safe_entropy(app_times)
    return out


def extract_usage_shutdown_features(raw_dir: Path) -> pd.DataFrame:
    df = anchor_lifelog_night(pd.read_parquet(raw_dir / "ch2025_mUsageStats.parquet"))
    usage_rows = []
    for _, row in df.iterrows():
        usage = categorize_usage(row["m_usage_stats"])
        usage.update({"subject_id": row["subject_id"], "anchor_date": row["anchor_date"], "rel_hour": row["rel_hour"]})
        usage_rows.append(usage)
    flat = pd.DataFrame(usage_rows)
    categories = ["social", "video", "game", "comm", "search"]
    rows: list[dict] = []
    for (subject_id, anchor_date), group in flat.groupby(["subject_id", "anchor_date"], sort=False):
        rel = group["rel_hour"].to_numpy()
        row = {"subject_id": subject_id, "date": anchor_date.date()}
        for window in ["eve18_22", "pre22_24", "late24_26", "night22_33", "all18_33"]:
            mask = window_mask(rel, window)
            subset = group.loc[mask]
            row[f"shutdown_usage_{window}_total_min"] = (
                float(subset["usage_total"].sum() / 60000.0) if len(subset) else np.nan
            )
            row[f"shutdown_usage_{window}_n_apps"] = float(subset["usage_n_apps"].sum()) if len(subset) else np.nan
            row[f"shutdown_usage_{window}_entropy_mean"] = (
                float(subset["usage_entropy"].mean()) if len(subset) else np.nan
            )
            denom = float(subset["usage_total"].sum() + 1e-6)
            for category in categories:
                row[f"shutdown_usage_{window}_{category}_ratio"] = (
                    float(subset[f"usage_{category}"].sum() / denom) if len(subset) else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def build_shutdown_sequence_features(raw_dir: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    feature_frames = [
        extract_screen_shutdown_features(raw_path),
        extract_activity_shutdown_features(raw_path),
        extract_scalar_window_features(raw_path, "ch2025_mLight.parquet", "m_light", "mlight"),
        extract_scalar_window_features(raw_path, "ch2025_wLight.parquet", "w_light", "wlight"),
        extract_hr_shutdown_features(raw_path),
        extract_usage_shutdown_features(raw_path),
    ]
    merged = reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), feature_frames)
    return merged.sort_values(["subject_id", "date"]).reset_index(drop=True)
