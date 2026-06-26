"""Q2/Q3 pre-sleep strain features from raw context sensors.

These features deliberately use same-day 12:00-24:00 windows rather than
post-midnight sensors because Q2/Q3 are pre-sleep self-reports for fatigue and
stress.
"""

from __future__ import annotations

from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd


def dateize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["date"] = out["timestamp"].dt.date
    out["hour_float"] = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    return out


def in_window(hour: pd.Series | np.ndarray, lo: float, hi: float) -> pd.Series | np.ndarray:
    return (hour >= lo) & (hour < hi)


def safe_entropy(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0.0
    probs = arr / arr.sum()
    return float(-(probs * np.log(probs + 1e-9)).sum())


def extract_gps_strain_features(raw_dir: Path) -> pd.DataFrame:
    df = dateize(pd.read_parquet(raw_dir / "ch2025_mGps.parquet"))
    rows: list[dict] = []
    for _, row in df.iterrows():
        points = row["m_gps"]
        speeds: list[float] = []
        lats: list[float] = []
        lons: list[float] = []
        if isinstance(points, (list, np.ndarray)):
            for point in points:
                if not isinstance(point, dict):
                    continue
                speeds.append(float(point.get("speed", 0) or 0))
                lats.append(float(point.get("latitude", point.get("lat", 0)) or 0))
                lons.append(float(point.get("longitude", point.get("lon", 0)) or 0))
        distance = 0.0
        if len(lats) > 1:
            distance = float(np.sum(np.sqrt(np.diff(lats) ** 2 + np.diff(lons) ** 2)))
        rows.append(
            {
                "subject_id": row["subject_id"],
                "date": row["date"],
                "hour_float": row["hour_float"],
                "gps_n": len(speeds),
                "gps_speed_mean": np.mean(speeds) if speeds else np.nan,
                "gps_speed_max": np.max(speeds) if speeds else np.nan,
                "gps_moving": float(np.mean(np.asarray(speeds) > 0.5)) if speeds else np.nan,
                "gps_dist": distance,
            }
        )
    flat = pd.DataFrame(rows)
    windows = {
        "aft12_18": (12.0, 18.0),
        "eve18_22": (18.0, 22.0),
        "pre20_24": (20.0, 24.0),
        "late22_24": (22.0, 24.0),
        "all12_24": (12.0, 24.0),
    }
    daily_rows: list[dict] = []
    for (subject_id, date), group in flat.groupby(["subject_id", "date"], sort=False):
        out = {"subject_id": subject_id, "date": date}
        for name, (lo, hi) in windows.items():
            subset = group[in_window(group["hour_float"], lo, hi)]
            out[f"strain_gps_{name}_dist"] = float(subset["gps_dist"].sum()) if len(subset) else np.nan
            out[f"strain_gps_{name}_moving"] = float(subset["gps_moving"].mean()) if len(subset) else np.nan
            out[f"strain_gps_{name}_speed_max"] = float(subset["gps_speed_max"].max()) if len(subset) else np.nan
            out[f"strain_gps_{name}_records"] = float(subset["gps_n"].sum()) if len(subset) else np.nan
        moving = group[
            (group["hour_float"] >= 12.0) & (group["hour_float"] < 24.0) & (group["gps_moving"].fillna(0) > 0)
        ]
        out["strain_gps_last_move_hour"] = float(moving["hour_float"].max()) if len(moving) else -1.0
        daily_rows.append(out)
    return pd.DataFrame(daily_rows)


def extract_wifi_home_features(raw_dir: Path) -> pd.DataFrame:
    df = dateize(pd.read_parquet(raw_dir / "ch2025_mWifi.parquet"))
    records: list[dict] = []
    for _, row in df.iterrows():
        networks = row["m_wifi"]
        ids: list[str] = []
        strong: list[str] = []
        if isinstance(networks, (list, np.ndarray)):
            for network in networks:
                if not isinstance(network, dict):
                    continue
                bssid = network.get("bssid")
                rssi = float(network.get("rssi", -999) or -999)
                if bssid:
                    ids.append(str(bssid))
                if bssid and rssi >= -65:
                    strong.append(str(bssid))
        records.append(
            {
                "subject_id": row["subject_id"],
                "date": row["date"],
                "hour_float": row["hour_float"],
                "ids": set(ids),
                "strong": set(strong),
                "n": len(ids),
            }
        )
    flat = pd.DataFrame(records)

    home_bssid: dict[str, str | None] = {}
    evening = flat[(flat["hour_float"] >= 20.0) & (flat["hour_float"] < 24.0)]
    for subject_id, group in evening.groupby("subject_id"):
        counts: dict[str, int] = {}
        for strong_ids in group["strong"]:
            for bssid in strong_ids:
                counts[bssid] = counts.get(bssid, 0) + 1
        home_bssid[subject_id] = max(counts.items(), key=lambda item: item[1])[0] if counts else None

    daily_rows: list[dict] = []
    for (subject_id, date), group in flat.groupby(["subject_id", "date"], sort=False):
        home = home_bssid.get(subject_id)
        out = {"subject_id": subject_id, "date": date}
        for name, lo, hi in [("eve18_24", 18.0, 24.0), ("pre20_24", 20.0, 24.0), ("late22_24", 22.0, 24.0)]:
            subset = group[in_window(group["hour_float"], lo, hi)]
            ids = set().union(*subset["ids"].tolist()) if len(subset) else set()
            strong_ids = set().union(*subset["strong"].tolist()) if len(subset) else set()
            out[f"strain_wifi_{name}_unique"] = float(len(ids))
            out[f"strain_wifi_{name}_strong_unique"] = float(len(strong_ids))
            out[f"strain_wifi_{name}_seen_sum"] = float(subset["n"].sum()) if len(subset) else np.nan
            out[f"strain_wifi_{name}_home_seen_ratio"] = (
                float(np.mean([home in ids_set for ids_set in subset["ids"]])) if home and len(subset) else np.nan
            )
        if home:
            home_seen = group["ids"].map(lambda ids_set, home_id=home: home_id in ids_set)
        else:
            home_seen = pd.Series(False, index=group.index)
        home_rows = group[(group["hour_float"] >= 12.0) & (group["hour_float"] < 24.0) & home_seen]
        out["strain_wifi_home_first_hour"] = float(home_rows["hour_float"].min()) if len(home_rows) else 25.0
        out["strain_wifi_home_last_hour"] = float(home_rows["hour_float"].max()) if len(home_rows) else -1.0
        out["strain_wifi_late_home_absent"] = float(1.0 - (out.get("strain_wifi_late22_24_home_seen_ratio", 0) or 0))
        daily_rows.append(out)
    return pd.DataFrame(daily_rows)


def extract_usage_strain_features(raw_dir: Path) -> pd.DataFrame:
    df = dateize(pd.read_parquet(raw_dir / "ch2025_mUsageStats.parquet"))
    categories = {
        "comm": ["전화", "통화", "message", "sms", "문자", "kakao", "카카오", "톡"],
        "work_search": [
            "naver",
            "chrome",
            "google",
            "검색",
            "메일",
            "mail",
            "office",
            "teams",
            "zoom",
            "토스",
            "bank",
            "은행",
        ],
        "video_social": ["youtube", "유튜브", "instagram", "인스타", "netflix", "tiktok", "facebook"],
        "game": ["game", "게임"],
        "shopping": ["쿠팡", "쇼핑", "배민", "요기요", "coupang"],
    }
    records: list[dict] = []
    for _, row in df.iterrows():
        apps = row["m_usage_stats"]
        totals = dict.fromkeys(categories, 0.0)
        total = 0.0
        times: list[float] = []
        if isinstance(apps, (list, np.ndarray)):
            for app in apps:
                if not isinstance(app, dict):
                    continue
                app_name = str(app.get("app_name", "")).lower()
                total_time = float(app.get("total_time", 0) or 0)
                total += total_time
                if total_time > 0:
                    times.append(total_time)
                for category, keywords in categories.items():
                    if any(keyword in app_name for keyword in keywords):
                        totals[category] += total_time
        records.append(
            {
                "subject_id": row["subject_id"],
                "date": row["date"],
                "hour_float": row["hour_float"],
                "total": total,
                "n_apps": len(times),
                "entropy": safe_entropy(times),
                "top_share": max(times) / (sum(times) + 1e-6) if times else 0.0,
                **{f"u_{key}": value for key, value in totals.items()},
            }
        )
    flat = pd.DataFrame(records)
    windows = {
        "eve18_22": (18.0, 22.0),
        "pre20_24": (20.0, 24.0),
        "late22_24": (22.0, 24.0),
        "all12_24": (12.0, 24.0),
    }
    daily_rows: list[dict] = []
    for (subject_id, date), group in flat.groupby(["subject_id", "date"], sort=False):
        out = {"subject_id": subject_id, "date": date}
        for name, (lo, hi) in windows.items():
            subset = group[in_window(group["hour_float"], lo, hi)]
            denom = float(subset["total"].sum() + 1e-6)
            out[f"strain_usage_{name}_total_min"] = float(subset["total"].sum() / 60000.0) if len(subset) else np.nan
            out[f"strain_usage_{name}_n_apps"] = float(subset["n_apps"].sum()) if len(subset) else np.nan
            out[f"strain_usage_{name}_entropy"] = float(subset["entropy"].mean()) if len(subset) else np.nan
            out[f"strain_usage_{name}_top_share"] = float(subset["top_share"].mean()) if len(subset) else np.nan
            for category in categories:
                out[f"strain_usage_{name}_{category}_ratio"] = (
                    float(subset[f"u_{category}"].sum() / denom) if len(subset) else np.nan
                )
        active = group[(group["hour_float"] >= 12.0) & (group["hour_float"] < 24.0) & (group["total"] > 0)]
        out["strain_usage_last_active_hour"] = float(active["hour_float"].max()) if len(active) else -1.0
        daily_rows.append(out)
    return pd.DataFrame(daily_rows)


def extract_ambience_ble_strain_features(raw_dir: Path) -> pd.DataFrame:
    ambience = dateize(pd.read_parquet(raw_dir / "ch2025_mAmbience.parquet"))
    labels = {
        "speech": ["speech"],
        "music": ["music"],
        "vehicle": ["vehicle", "car", "motor", "traffic"],
        "outside": ["outside", "urban", "rural"],
    }
    ambience_records: list[dict] = []
    for _, row in ambience.iterrows():
        values = dict.fromkeys(labels, 0.0)
        items = row["m_ambience"]
        if isinstance(items, (list, np.ndarray)):
            for item in items:
                if not isinstance(item, (list, tuple, np.ndarray)) or len(item) < 2:
                    continue
                label = str(item[0]).lower()
                probability = float(item[1])
                for key, keywords in labels.items():
                    if any(keyword in label for keyword in keywords):
                        values[key] = max(values[key], probability)
        ambience_records.append(
            {"subject_id": row["subject_id"], "date": row["date"], "hour_float": row["hour_float"], **values}
        )
    flat_ambience = pd.DataFrame(ambience_records)
    ambience_rows: list[dict] = []
    for (subject_id, date), group in flat_ambience.groupby(["subject_id", "date"], sort=False):
        out = {"subject_id": subject_id, "date": date}
        for name, lo, hi in [("eve18_22", 18.0, 22.0), ("late22_24", 22.0, 24.0), ("pre20_24", 20.0, 24.0)]:
            subset = group[in_window(group["hour_float"], lo, hi)]
            for key in labels:
                out[f"strain_amb_{name}_{key}_mean"] = float(subset[key].mean()) if len(subset) else np.nan
                out[f"strain_amb_{name}_{key}_max"] = float(subset[key].max()) if len(subset) else np.nan
        ambience_rows.append(out)
    ambience_features = pd.DataFrame(ambience_rows)

    ble = dateize(pd.read_parquet(raw_dir / "ch2025_mBle.parquet"))
    ble_records: list[dict] = []
    for _, row in ble.iterrows():
        devices = row["m_ble"]
        ids: list[str] = []
        strong = 0
        if isinstance(devices, (list, np.ndarray)):
            for device in devices:
                if not isinstance(device, dict):
                    continue
                address = device.get("address")
                rssi = float(device.get("rssi", -999) or -999)
                if address:
                    ids.append(str(address))
                if rssi >= -65:
                    strong += 1
        ble_records.append(
            {
                "subject_id": row["subject_id"],
                "date": row["date"],
                "hour_float": row["hour_float"],
                "ids": set(ids),
                "n": len(ids),
                "strong": strong,
            }
        )
    flat_ble = pd.DataFrame(ble_records)
    ble_rows: list[dict] = []
    for (subject_id, date), group in flat_ble.groupby(["subject_id", "date"], sort=False):
        out = {"subject_id": subject_id, "date": date}
        for name, lo, hi in [("eve18_22", 18.0, 22.0), ("late22_24", 22.0, 24.0), ("pre20_24", 20.0, 24.0)]:
            subset = group[in_window(group["hour_float"], lo, hi)]
            ids = set().union(*subset["ids"].tolist()) if len(subset) else set()
            out[f"strain_ble_{name}_unique"] = float(len(ids))
            out[f"strain_ble_{name}_seen"] = float(subset["n"].sum()) if len(subset) else np.nan
            out[f"strain_ble_{name}_strong"] = float(subset["strong"].sum()) if len(subset) else np.nan
        ble_rows.append(out)
    return ambience_features.merge(pd.DataFrame(ble_rows), on=["subject_id", "date"], how="outer")


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    interactions = [
        ("strain_gps_late22_24_moving", "strain_usage_late22_24_total_min", "late_move_usage"),
        ("strain_wifi_late_home_absent", "strain_usage_late22_24_comm_ratio", "away_comm"),
        ("strain_wifi_late_home_absent", "strain_amb_late22_24_vehicle_mean", "away_vehicle"),
        ("strain_ble_late22_24_unique", "strain_amb_late22_24_speech_mean", "crowd_speech"),
    ]
    for left, right, name in interactions:
        if left in out.columns and right in out.columns:
            out[f"strain_inter_{name}"] = out[left] * out[right]
    return out


def build_q23_strain_features(raw_dir: str | Path) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    frames = [
        extract_gps_strain_features(raw_path),
        extract_wifi_home_features(raw_path),
        extract_usage_strain_features(raw_path),
        extract_ambience_ble_strain_features(raw_path),
    ]
    merged = reduce(lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), frames)
    merged = add_interaction_features(merged)
    return merged.sort_values(["subject_id", "date"]).reset_index(drop=True)
