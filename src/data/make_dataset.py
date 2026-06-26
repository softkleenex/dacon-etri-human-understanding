from functools import reduce

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from src.features.demographics import add_official_demographics
from src.utils.logger import get_logger

logger = get_logger(__name__)


def process_hr_stat(x, stat="mean"):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            if stat == "mean":
                return np.mean(x)
            if stat == "max":
                return np.max(x)
            if stat == "min":
                return np.min(x)
            if stat == "std":
                return np.std(x)
        return np.nan
    except Exception:
        return np.nan


def process_usage_array(x):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            return sum(item.get("total_time", 0) for item in x if isinstance(item, dict))
        return 0
    except Exception:
        return 0


def process_usage_dict(x):
    """Returns a dict of app_name: total_time"""
    app_dict = {}
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            for item in x:
                if isinstance(item, dict):
                    name = item.get("app_name", "unknown")
                    name = str(name).strip().lower()
                    app_dict[name] = app_dict.get(name, 0) + item.get("total_time", 0)
    except Exception:
        pass
    return app_dict


def process_ambience_dict(x):
    """Returns a dict of sound_label: max_probability"""
    amb_dict = {}
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            for item in x:
                if isinstance(item, (list, np.ndarray)) and len(item) == 2:
                    label = str(item[0]).strip().lower()
                    try:
                        prob = float(item[1])
                        amb_dict[label] = max(amb_dict.get(label, 0.0), prob)
                    except Exception:
                        pass
    except Exception:
        pass
    return amb_dict


def process_hr_rmssd(x):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 1:
            diffs = np.diff(x)
            rmssd = np.sqrt(np.mean(diffs**2))
            return rmssd
        return 0.0
    except Exception:
        return 0.0


# ==============================================================================
# Helper functions from public v9 shared notebook
# ==============================================================================


def agg_stats(vals, prefix):
    if len(vals) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
        }
    return {
        f"{prefix}_mean": np.nanmean(vals),
        f"{prefix}_std": np.nanstd(vals),
        f"{prefix}_min": np.nanmin(vals),
        f"{prefix}_max": np.nanmax(vals),
        f"{prefix}_median": np.nanmedian(vals),
        f"{prefix}_q25": np.nanpercentile(vals, 25),
        f"{prefix}_q75": np.nanpercentile(vals, 75),
    }


def safe_mean(vals):
    arr = np.array(vals)
    return np.nanmean(arr) if len(arr) > 0 else np.nan


# ==============================================================================
# Base Daily Features Aggregator from previous SOTA pipeline
# ==============================================================================


def create_daily_features(sensor_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Creating base daily features from sensor data...")
    daily_features = []

    for sensor_name, df in sensor_data_dict.items():
        if df.empty:
            continue

        df = df.copy()

        if "user_id" in df.columns and "subject_id" not in df.columns:
            df = df.rename(columns={"user_id": "subject_id"})

        if "subject_id" not in df.columns or "timestamp" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        if sensor_name == "ch2025_wHr":
            df["hr_mean"] = df["heart_rate"].apply(lambda x: process_hr_stat(x, "mean"))
            df["hr_max"] = df["heart_rate"].apply(lambda x: process_hr_stat(x, "max"))
            df["hr_min"] = df["heart_rate"].apply(lambda x: process_hr_stat(x, "min"))
            df["hr_rmssd"] = df["heart_rate"].apply(process_hr_rmssd)

            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            df["is_night"] = df["hour"].isin([2, 3, 4, 5])
            df["is_day"] = df["hour"].isin([10, 11, 12, 13, 14, 15, 16])

            df["hr_night"] = df["hr_mean"].where(df["is_night"])
            df["hr_day"] = df["hr_mean"].where(df["is_day"])

            agg_dict = {
                "hr_mean": ["mean", "std"],
                "hr_max": ["max"],
                "hr_min": ["min"],
                "hr_rmssd": ["mean", "max"],
                "hr_night": ["mean"],
                "hr_day": ["mean"],
            }
        elif sensor_name in ["ch2025_mLight", "ch2025_wLight"]:
            light_col = (
                "light"
                if "light" in df.columns
                else "m_light"
                if "m_light" in df.columns
                else "w_light"
                if "w_light" in df.columns
                else None
            )
            if light_col:
                df["light_mean"] = df[light_col].apply(
                    lambda x: (
                        np.mean(x)
                        if isinstance(x, (list, np.ndarray)) and len(x) > 0
                        else x
                        if pd.api.types.is_numeric_dtype(type(x))
                        else 0
                    )
                )
                df["light_max"] = df[light_col].apply(
                    lambda x: (
                        np.max(x)
                        if isinstance(x, (list, np.ndarray)) and len(x) > 0
                        else x
                        if pd.api.types.is_numeric_dtype(type(x))
                        else 0
                    )
                )
                agg_dict = {"light_mean": ["mean", "max"], "light_max": ["max"]}
            else:
                numeric_cols = [
                    c
                    for c in df.select_dtypes(include="number").columns.tolist()
                    if c not in ["subject_id", "timestamp", "date"]
                ]
                agg_dict = {col: ["mean", "max"] for col in numeric_cols}
        elif sensor_name == "ch2025_mUsageStats":
            df["total_usage_time"] = df["m_usage_stats"].apply(process_usage_array)
            df["num_apps"] = df["m_usage_stats"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

            target_apps = [
                "youtube",
                "kakao",
                "instagram",
                "naver",
                "chrome",
                "netflix",
                "tiktok",
                "game",
                "camera",
                "gallery",
                "music",
            ]
            for app in target_apps:
                app_name = app
                df[f"app_{app}"] = df["m_usage_stats"].apply(
                    lambda x, app_name=app_name: (
                        sum(
                            i.get("total_time", 0)
                            for i in x
                            if isinstance(i, dict) and app_name in str(i.get("app_name", "")).lower()
                        )
                        if isinstance(x, (list, np.ndarray))
                        else 0
                    )
                )

            agg_dict = {"total_usage_time": ["sum", "mean"], "num_apps": ["mean", "max"]}
            for app in target_apps:
                agg_dict[f"app_{app}"] = ["sum"]

        elif sensor_name == "ch2025_wPedo":
            numeric_cols = ["step", "distance", "burned_calories", "running_step", "walking_step"]
            agg_dict = {col: ["sum", "max"] for col in numeric_cols if col in df.columns}
        elif sensor_name == "ch2025_mActivity":
            df["is_active"] = (df["m_activity"] > 0).astype(int)
            agg_dict = {"m_activity": ["mean", "std", "max"], "is_active": ["sum", "mean"]}
        elif sensor_name == "ch2025_mACStatus":
            agg_dict = {"m_charging": ["sum", "mean"]}
        elif sensor_name == "ch2025_mScreenStatus":
            df = df.sort_values(by=["subject_id", "timestamp"])
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
            df["prev_state"] = df.groupby(["subject_id", "date"])["m_screen_use"].shift(1)
            df["is_unlock"] = ((df["m_screen_use"] == 1) & (df["prev_state"] == 0)).astype(int)
            df["bedtime_screen_use"] = df["m_screen_use"].where(df["hour"].isin([22, 23, 0, 1]))
            df["screen_first_hour"] = df["hour"].where(df["m_screen_use"] > 0)
            df["screen_last_hour"] = df["hour"].where(df["m_screen_use"] > 0)

            agg_dict = {
                "m_screen_use": ["sum", "mean"],
                "is_unlock": ["sum"],
                "bedtime_screen_use": ["sum"],
                "screen_first_hour": ["min"],
                "screen_last_hour": ["max"],
            }
        elif sensor_name == "ch2025_mAmbience":
            df["ambience_changes"] = df["m_ambience"].apply(
                lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0
            )
            target_sounds = ["speech", "music", "vehicle", "silence", "outdoor"]
            for snd in target_sounds:
                sound_name = snd
                df[f"snd_{snd}"] = df["m_ambience"].apply(
                    lambda x, sound_name=sound_name: (
                        max(
                            [
                                float(i[1])
                                for i in x
                                if isinstance(i, (list, np.ndarray)) and len(i) == 2 and sound_name in str(i[0]).lower()
                            ]
                            + [0.0]
                        )
                        if isinstance(x, (list, np.ndarray))
                        else 0.0
                    )
                )
            agg_dict = {"ambience_changes": ["sum", "mean"]}
            for snd in target_sounds:
                agg_dict[f"snd_{snd}"] = ["max", "mean"]
        elif sensor_name == "ch2025_mBle":

            def calc_ble_entropy(records):
                if not isinstance(records, (list, np.ndarray)) or len(records) == 0:
                    return 0.0
                addresses = [r.get("address") for r in records if isinstance(r, dict)]
                if len(addresses) == 0:
                    return 0.0
                counts = pd.Series(addresses).value_counts()
                probs = counts / len(addresses)
                return -np.sum(probs * np.log(probs + 1e-8))

            df["ble_entropy"] = df["m_ble"].apply(calc_ble_entropy)
            df["ble_devices"] = df["m_ble"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            agg_dict = {"ble_devices": ["sum", "mean", "max"], "ble_entropy": ["mean", "max"]}
        elif sensor_name == "ch2025_mWifi":
            df["primary_bssid"] = df["m_wifi"].apply(
                lambda x: (
                    x[0].get("bssid")
                    if isinstance(x, (list, np.ndarray)) and len(x) > 0 and isinstance(x[0], dict)
                    else None
                )
            )
            df["prev_bssid"] = df.groupby(["subject_id", "date"])["primary_bssid"].shift(1)
            df["bssid_transition"] = (df["primary_bssid"] != df["prev_bssid"]).astype(int)
            df["wifi_devices"] = df["m_wifi"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            df["bssid_set"] = df["m_wifi"].apply(
                lambda x: (
                    {i.get("bssid") for i in x if isinstance(i, dict) and "bssid" in i}
                    if isinstance(x, (list, np.ndarray))
                    else set()
                )
            )

            agg_dict = {"wifi_devices": ["sum", "mean", "max"], "bssid_transition": ["sum"]}

            def set_union(series):
                res = set()
                for s in series:
                    res.update(s)
                return res

            agg_dict["bssid_set"] = [set_union]
        elif sensor_name == "ch2025_mGps":
            df["gps_records"] = df["m_gps"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

            def calc_gps_dist(x):
                if not isinstance(x, (list, np.ndarray)) or len(x) < 2:
                    return 0.0
                try:
                    lats = [float(i.get("lat", 0)) for i in x if isinstance(i, dict)]
                    lons = [float(i.get("lon", 0)) for i in x if isinstance(i, dict)]
                    if len(lats) < 2:
                        return 0.0
                    return sum(abs(lats[i] - lats[i - 1]) + abs(lons[i] - lons[i - 1]) for i in range(1, len(lats)))
                except Exception:
                    return 0.0

            df["gps_dist"] = df["m_gps"].apply(calc_gps_dist)
            agg_dict = {"gps_records": ["sum", "mean"], "gps_dist": ["sum"]}
        else:
            numeric_cols = [
                c
                for c in df.select_dtypes(include="number").columns.tolist()
                if c not in ["subject_id", "timestamp", "date"]
            ]
            agg_dict = {col: ["mean", "std", "min", "max", "sum"] for col in numeric_cols}

        if agg_dict:
            agg_df = df.groupby(["subject_id", "date"]).agg(agg_dict)
            if isinstance(agg_df.columns, pd.MultiIndex):
                agg_df.columns = [f"{sensor_name}_{col[0]}_{col[1]}" for col in agg_df.columns]
            else:
                agg_df.columns = [f"{sensor_name}_{col}" for col in agg_df.columns]
            agg_df = agg_df.reset_index()
            daily_features.append(agg_df)

    if not daily_features:
        logger.warning("No aggregatable features found.")
        return pd.DataFrame()

    merged_df = reduce(
        lambda left, right: pd.merge(left, right, on=["subject_id", "date"], how="outer"), daily_features
    )
    return merged_df


# ==============================================================================
# SOTA Multimodal Feature Engineering Extensions (GPS Clustering, App Entropy)
# ==============================================================================


def extract_location_clustering_features(mGps_df: pd.DataFrame) -> pd.DataFrame:
    if mGps_df is None or mGps_df.empty:
        return pd.DataFrame()

    df = mGps_df.copy()
    if "user_id" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"user_id": "subject_id"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    def parse_coordinates(row):
        records = row["m_gps"]
        coords = []
        if isinstance(records, (list, np.ndarray)):
            for r in records:
                if isinstance(r, dict):
                    lat = r.get("lat") or r.get("latitude")
                    lon = r.get("lon") or r.get("longitude")
                    if lat is not None and lon is not None and lat != -1 and lon != -1:
                        coords.append((float(lat), float(lon)))
        return coords

    df["coords_list"] = df.apply(parse_coordinates, axis=1)

    daily_records = []
    for (subject_id, date), group in df.groupby(["subject_id", "date"]):
        all_coords = []
        for c_list in group["coords_list"]:
            all_coords.extend(c_list)

        if len(all_coords) < 5:
            continue

        X = np.array(all_coords)

        try:
            db = DBSCAN(eps=0.001, min_samples=3).fit(X)
            labels = db.labels_

            unique_labels = set(labels) - {-1}
            num_locations = len(unique_labels)

            centroid = np.mean(X, axis=0)
            radius_of_gyration = np.sqrt(np.mean(np.sum((X - centroid) ** 2, axis=1)))

            total_pings = len(labels)
            counts = pd.Series(labels).value_counts()
            probs = counts / total_pings
            entropy = -np.sum(probs * np.log(probs + 1e-8))

            valid_counts = counts[counts.index != -1]
            if not valid_counts.empty:
                homestay_ratio = valid_counts.max() / total_pings
            else:
                homestay_ratio = counts.max() / total_pings

            daily_records.append(
                {
                    "subject_id": subject_id,
                    "date": date,
                    "mGps_unique_locations": float(num_locations),
                    "mGps_radius_of_gyration": float(radius_of_gyration),
                    "mGps_spatial_entropy": float(entropy),
                    "mGps_homestay_ratio": float(homestay_ratio),
                }
            )
        except Exception:
            pass

    return pd.DataFrame(daily_records)


def extract_app_usage_features(mUsage_df: pd.DataFrame) -> pd.DataFrame:
    if mUsage_df is None or mUsage_df.empty:
        return pd.DataFrame()

    df = mUsage_df.copy()
    if "user_id" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"user_id": "subject_id"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    dopamine_apps = {"youtube", "instagram", "netflix", "tiktok", "facebook", "game", "twitch", "gallery", "camera"}

    daily_records = []
    for (subject_id, date), group in df.groupby(["subject_id", "date"]):
        group = group.sort_values("timestamp")

        total_usage_ms = 0
        nighttime_usage_ms = 0
        dopamine_usage_ms = 0
        app_times = {}
        hourly_usage = dict.fromkeys(range(24), 0.0)
        sessions_count = 0

        for _, row in group.iterrows():
            hr = row["hour"]
            app_list = row["m_usage_stats"]

            if not isinstance(app_list, (list, np.ndarray)):
                continue

            row_total = 0
            for app in app_list:
                if isinstance(app, dict):
                    name = str(app.get("app_name", "")).lower().strip()
                    time_ms = float(app.get("total_time", 0))

                    if time_ms > 0:
                        row_total += time_ms
                        app_times[name] = app_times.get(name, 0.0) + time_ms

                        if any(da in name for da in dopamine_apps):
                            dopamine_usage_ms += time_ms

            if row_total > 0:
                sessions_count += 1
                total_usage_ms += row_total
                hourly_usage[hr] = hourly_usage.get(hr, 0.0) + row_total

                if hr in [22, 23, 0, 1, 2, 3, 4, 5]:
                    nighttime_usage_ms += row_total

        if total_usage_ms == 0:
            continue

        probs = np.array(list(app_times.values())) / total_usage_ms
        app_entropy = -np.sum(probs * np.log(probs + 1e-8))

        hourly_vals = np.array(list(hourly_usage.values()))
        hourly_volatility = np.std(hourly_vals)

        daily_records.append(
            {
                "subject_id": subject_id,
                "date": date,
                "mUsage_total_time_mins": float(total_usage_ms / 60000.0),
                "mUsage_night_time_ratio": float(nighttime_usage_ms / (total_usage_ms + 1e-5)),
                "mUsage_dopamine_ratio": float(dopamine_usage_ms / (total_usage_ms + 1e-5)),
                "mUsage_app_entropy": float(app_entropy),
                "mUsage_hourly_volatility": float(hourly_volatility / 60000.0),
                "mUsage_sessions_count": float(sessions_count),
            }
        )

    return pd.DataFrame(daily_records)


# ==============================================================================
# SOTA Sleep Phase transition features (aligned to sleep_date wake-up day)
# ==============================================================================


def extract_sleep_phase_features(wHr_df: pd.DataFrame) -> pd.DataFrame:
    if wHr_df is None or wHr_df.empty:
        return pd.DataFrame()

    df = wHr_df.copy()
    if "user_id" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"user_id": "subject_id"})

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Filter for typical sleep/night window (22:00 to 09:00 KST, with morning concentration)
    sleep_window = df[df["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])].copy()

    # Fast vectorized calculation of sleep wake-up date
    sleep_window["base_date"] = sleep_window["timestamp"].dt.normalize().dt.date
    sleep_window["sleep_date"] = sleep_window["base_date"]
    is_evening = sleep_window["hour"] >= 22
    sleep_window.loc[is_evening, "sleep_date"] = sleep_window.loc[is_evening, "base_date"] + pd.Timedelta(days=1)

    def flatten_hr(row):
        hr = row["heart_rate"]
        if isinstance(hr, (list, np.ndarray)):
            return [float(v) for v in hr if pd.notnull(v)]
        elif pd.notnull(hr):
            try:
                return [float(hr)]
            except Exception:
                return []
        return []

    sleep_window["hr_list"] = sleep_window.apply(flatten_hr, axis=1)

    daily_records = []
    for (subject_id, sleep_date), group in sleep_window.groupby(["subject_id", "sleep_date"]):
        group = group.sort_values("timestamp")

        all_hr = []
        for hr_list in group["hr_list"]:
            all_hr.extend(hr_list)

        if len(all_hr) < 10:
            continue

        all_hr = np.array(all_hr)

        # Personalized baseline quantiles for this user-night
        hr_low = np.percentile(all_hr, 30)
        hr_high = np.percentile(all_hr, 80)

        # Rolling variance (stability) proxy
        rolling_std = []
        for i in range(len(all_hr)):
            start = max(0, i - 2)
            end = min(len(all_hr), i + 3)
            rolling_std.append(np.std(all_hr[start:end]) if len(all_hr[start:end]) > 1 else 0.0)
        rolling_std = np.array(rolling_std)
        std_threshold = np.median(rolling_std)

        states = []
        for hr_val, r_std in zip(all_hr, rolling_std, strict=False):
            if hr_val <= hr_low and r_std <= std_threshold:
                states.append(0)
            elif hr_val >= hr_high or r_std > (std_threshold * 2):
                states.append(2)
            else:
                states.append(1)

        states = np.array(states)

        deep_dur = np.sum(states == 0)
        light_dur = np.sum(states == 1)
        awake_dur = np.sum(states == 2)
        total_dur = len(states)

        state_changes = np.sum(states[:-1] != states[1:])
        fragmentation_index = state_changes / (total_dur / 60.0 + 1e-5)

        slp_hr_deep_ratio = np.mean(all_hr < 55) if len(all_hr) > 0 else np.nan
        slp_hr_awake_ratio = np.mean(all_hr > 75) if len(all_hr) > 0 else np.nan
        slp_hr_light_ratio = np.mean((all_hr >= 55) & (all_hr <= 75)) if len(all_hr) > 0 else np.nan

        if len(all_hr) > 1:
            diffs = np.diff(all_hr)
            slp_hr_rmssd = float(np.sqrt(np.nanmean(diffs**2)))
        else:
            slp_hr_rmssd = np.nan

        slp_hr_spike_count = 0
        if len(all_hr) > 5:
            rolling_mean = pd.Series(all_hr).rolling(5, min_periods=1).mean().values
            slp_hr_spike_count = int(np.sum(np.abs(all_hr - rolling_mean) > 15))

        daily_records.append(
            {
                "subject_id": subject_id,
                "sleep_date": sleep_date,
                "wHr_sleep_duration_deep": float(deep_dur),
                "wHr_sleep_duration_light": float(light_dur),
                "wHr_sleep_duration_awake": float(awake_dur),
                "wHr_deep_sleep_ratio": float(deep_dur / (deep_dur + light_dur + 1e-5)),
                "wHr_sleep_efficiency_proxy": float((deep_dur + light_dur) / (total_dur + 1e-5)),
                "wHr_sleep_fragmentation_index": float(fragmentation_index),
                "wHr_sleep_hr_mean": float(np.mean(all_hr)),
                "wHr_sleep_hr_std": float(np.std(all_hr)),
                "wHr_slp_hr_deep_ratio": float(slp_hr_deep_ratio),
                "wHr_slp_hr_awake_ratio": float(slp_hr_awake_ratio),
                "wHr_slp_hr_light_ratio": float(slp_hr_light_ratio),
                "wHr_slp_hr_rmssd": float(slp_hr_rmssd),
                "wHr_slp_hr_spike_count": float(slp_hr_spike_count),
            }
        )

    return pd.DataFrame(daily_records)


# ==============================================================================
# Feature Extraction from public v9 shared notebook
# ==============================================================================


def extract_activity_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        acts = grp["m_activity"].values
        h = grp["timestamp"].dt.hour.values
        for a in [0, 3, 4, 7, 8]:
            row[f"act_{a}_ratio"] = (acts == a).mean()
        row["act_active_ratio"] = ((acts == 7) | (acts == 8) | (acts == 3)).mean()
        row["act_still_ratio"] = (acts == 0).mean()
        row["act_n_records"] = len(acts)
        for seg, mask in [
            ("morn", (h >= 6) & (h < 12)),
            ("aftn", (h >= 12) & (h < 18)),
            ("eve", (h >= 18) & (h < 22)),
            ("night", (h >= 22) | (h < 6)),
        ]:
            s_acts = acts[mask]
            row[f"act_{seg}_active"] = ((s_acts == 7) | (s_acts == 8)).mean() if len(s_acts) > 0 else np.nan
            row[f"act_{seg}_still"] = (s_acts == 0).mean() if len(s_acts) > 0 else np.nan
        pre = acts[(h >= 22) & (h < 24)]
        row["act_presleep_active"] = ((pre == 7) | (pre == 8)).mean() if len(pre) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_pedo_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        row["pedo_total_steps"] = grp["step"].sum()
        row["pedo_total_distance"] = grp["distance"].sum()
        row["pedo_total_calories"] = grp["burned_calories"].sum()
        row["pedo_max_speed"] = grp["speed"].max()
        row["pedo_mean_speed"] = grp["speed"].mean()
        row["pedo_running_steps"] = grp["running_step"].sum()
        row["pedo_walking_steps"] = grp["walking_step"].sum()
        row["pedo_run_ratio"] = grp["running_step"].sum() / (grp["step"].sum() + 1)
        eve = grp[grp["timestamp"].dt.hour.between(18, 21)]
        row["pedo_evening_steps"] = eve["step"].sum()
        row["pedo_step_freq_mean"] = grp["step_frequency"].mean()
        row["pedo_step_freq_max"] = grp["step_frequency"].max()
        hourly = grp.groupby(grp["timestamp"].dt.hour)["step"].sum()
        row["pedo_active_hours"] = (hourly > 50).sum()
        feats.append(row)
    return pd.DataFrame(feats)


def extract_screen_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        sc = grp["m_screen_use"].values
        h = grp["timestamp"].dt.hour.values
        row["screen_on_total"] = (sc > 0).sum()
        row["screen_on_ratio"] = (sc > 0).mean()
        row["screen_unlock_cnt"] = (sc[1:] > sc[:-1]).sum()
        for seg, mask in [
            ("night", (h >= 22) | (h < 2)),
            ("eve", (h >= 20) & (h <= 23)),
            ("presleep", (h >= 22) & (h < 24)),
        ]:
            s_sc = sc[mask]
            row[f"screen_{seg}_on"] = (s_sc > 0).sum()
            row[f"screen_{seg}_ratio"] = (s_sc > 0).mean() if len(s_sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_light_v9(df_raw: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        vals = grp[col].dropna().values
        for k, v in agg_stats(vals, f"{prefix}_all").items():
            row[k] = v
        h = grp["timestamp"].dt.hour
        for seg, (lo, hi) in [("eve", (18, 22)), ("morn", (6, 10)), ("night", (22, 24))]:
            sv = grp.loc[h.between(lo, hi - 1), col].dropna().values
            row[f"{prefix}_{seg}_mean"] = safe_mean(sv)
        row[f"{prefix}_dark_ratio"] = (vals < 10).mean() if len(vals) > 0 else np.nan
        row[f"{prefix}_bright_ratio"] = (vals > 1000).mean() if len(vals) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ac_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        ch = grp["m_charging"].values
        h = grp["timestamp"].dt.hour.values
        row["ac_charging_ratio"] = ch.mean()
        for seg, mask in [
            ("eve", (h >= 21) & (h <= 23)),
            ("night", (h >= 22) | (h < 4)),
            ("presleep", (h >= 22) & (h < 24)),
        ]:
            sc = ch[mask]
            row[f"ac_{seg}_charging"] = sc.mean() if len(sc) > 0 else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_gps_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        speeds, lats, lons = [], [], []
        for v in grp["m_gps"]:
            if isinstance(v, list):
                for pt in v:
                    if isinstance(pt, dict):
                        speeds.append(pt.get("speed", 0))
                        lats.append(pt.get("latitude", 0))
                        lons.append(pt.get("longitude", 0))
        speeds = np.array(speeds)
        row["gps_mean_speed"] = np.nanmean(speeds) if len(speeds) > 0 else np.nan
        row["gps_max_speed"] = np.nanmax(speeds) if len(speeds) > 0 else np.nan
        row["gps_moving_ratio"] = (speeds > 0.5).mean() if len(speeds) > 0 else np.nan
        row["gps_lat_std"] = np.nanstd(lats) if len(lats) > 0 else np.nan
        row["gps_lon_std"] = np.nanstd(lons) if len(lons) > 0 else np.nan
        if len(lats) > 1:
            dlat = np.diff(lats)
            dlon = np.diff(lons)
            row["gps_total_disp"] = float(np.sum(np.sqrt(dlat**2 + dlon**2)))
        else:
            row["gps_total_disp"] = 0.0
        feats.append(row)
    return pd.DataFrame(feats)


def extract_usage_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        total_time, late_time, eve_time, n_apps = 0, 0, 0, 0
        for ts, v in zip(grp["timestamp"], grp["m_usage_stats"], strict=False):
            if isinstance(v, list):
                for app in v:
                    if isinstance(app, dict):
                        t = app.get("total_time", 0) or 0
                        total_time += t
                        n_apps += 1
                        if ts.hour >= 22 or ts.hour < 2:
                            late_time += t
                        if ts.hour >= 18:
                            eve_time += t
        row["usage_total_time"] = total_time
        row["usage_n_apps"] = n_apps
        row["usage_late_time"] = late_time
        row["usage_late_ratio"] = late_time / (total_time + 1)
        row["usage_eve_time"] = eve_time
        row["usage_eve_ratio"] = eve_time / (total_time + 1)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_wifi_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        all_bssids, rssi_vals = set(), []
        for v in grp["m_wifi"]:
            if isinstance(v, list):
                for net in v:
                    if isinstance(net, dict):
                        all_bssids.add(net.get("bssid", ""))
                        rssi_vals.append(net.get("rssi", -100))
        row["wifi_n_unique"] = len(all_bssids)
        row["wifi_mean_rssi"] = np.mean(rssi_vals) if rssi_vals else np.nan
        row["wifi_max_rssi"] = np.max(rssi_vals) if rssi_vals else np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ble_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        addrs = set()
        for v in grp["m_ble"]:
            if isinstance(v, list):
                for dev in v:
                    if isinstance(dev, dict):
                        addrs.add(dev.get("address", ""))
        row["ble_n_unique"] = len(addrs)
        row["ble_n_scans"] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_ambience_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "date": d}
        music_s, speech_s, silence_s = [], [], []
        for v in grp["m_ambience"]:
            if isinstance(v, list):
                d_map = {item[0]: item[1] for item in v if isinstance(item, list) and len(item) == 2}
                music_s.append(d_map.get("Music", 0))
                speech_s.append(d_map.get("Speech", 0))
                silence_s.append(d_map.get("Silence", 0))
        row["amb_music_mean"] = np.mean(music_s) if music_s else np.nan
        row["amb_speech_mean"] = np.mean(speech_s) if speech_s else np.nan
        row["amb_silence_mean"] = np.mean(silence_s) if silence_s else np.nan
        row["amb_n_records"] = len(grp)
        feats.append(row)
    return pd.DataFrame(feats)


# ==============================================================================
# Sleep Morning Features from public v9 shared notebook
# ==============================================================================


def extract_sleep_hr_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    df_m = df_raw[df_raw["timestamp"].dt.hour < 9].copy()
    feats = []
    for (sid, d), grp in df_m.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "sleep_date": d}
        hour_vals = {h: [] for h in range(9)}
        all_v = []
        for ts, v in zip(grp["timestamp"], grp["heart_rate"], strict=False):
            try:
                arr = np.asarray(v, dtype=float).ravel()
                arr = arr[arr > 0]
            except Exception:
                arr = np.array([])
            all_v.extend(arr.tolist())
            hour_vals[ts.hour].extend(arr.tolist())
        sleep_hrs = np.array(all_v)
        sleep_hrs = sleep_hrs[sleep_hrs > 0] if len(sleep_hrs) > 0 else sleep_hrs
        for k, v in agg_stats(sleep_hrs, "slp_hr").items():
            row[k] = v
        row["slp_hr_deep_ratio"] = (sleep_hrs < 55).mean() if len(sleep_hrs) > 0 else np.nan
        row["slp_hr_awake_ratio"] = (sleep_hrs > 75).mean() if len(sleep_hrs) > 0 else np.nan
        row["slp_hr_light_ratio"] = ((sleep_hrs >= 55) & (sleep_hrs <= 75)).mean() if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 1:
            diffs = np.diff(sleep_hrs)
            row["slp_hr_rmssd"] = float(np.sqrt(np.nanmean(diffs**2)))
        else:
            row["slp_hr_rmssd"] = np.nan
        row["slp_hr_n_records"] = len(grp)
        row["slp_hr_early_mean"] = safe_mean(sum([hour_vals[h] for h in range(3)], []))
        row["slp_hr_late_mean"] = safe_mean(sum([hour_vals[h] for h in range(6, 9)], []))
        row["slp_hr_mid_mean"] = safe_mean(sum([hour_vals[h] for h in range(3, 6)], []))
        row["slp_hr_range"] = float(np.ptp(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        row["slp_hr_median"] = float(np.median(sleep_hrs)) if len(sleep_hrs) > 0 else np.nan
        if len(sleep_hrs) > 5:
            rolling = pd.Series(sleep_hrs).rolling(5, min_periods=1).mean().values
            row["slp_hr_spike_count"] = int((np.abs(sleep_hrs - rolling) > 15).sum())
        else:
            row["slp_hr_spike_count"] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_pedo_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "sleep_date": d}
        morn = grp[grp["timestamp"].dt.hour < 9]
        row["slp_pedo_steps"] = morn["step"].sum()
        row["slp_pedo_active"] = (morn["step"] > 5).sum()
        row["slp_pedo_calories"] = morn["burned_calories"].sum()
        row["slp_pedo_n_records"] = len(morn)
        mid = grp[grp["timestamp"].dt.hour.between(2, 4)]
        row["slp_pedo_mid_steps"] = mid["step"].sum()
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_activity_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "sleep_date": d}
        morn = grp[grp["timestamp"].dt.hour < 9]
        if len(morn) == 0:
            row.update({"slp_act_still_ratio": np.nan, "slp_act_active_ratio": np.nan, "slp_act_n_records": 0})
        else:
            acts = morn["m_activity"].values
            row["slp_act_still_ratio"] = (acts == 0).mean()
            row["slp_act_active_ratio"] = ((acts == 7) | (acts == 8)).mean()
            row["slp_act_n_records"] = len(acts)
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_screen_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "sleep_date": d}
        morn = grp[grp["timestamp"].dt.hour < 9]
        if len(morn) > 0:
            sc = morn["m_screen_use"].values
            row["slp_screen_on"] = (sc > 0).sum()
            row["slp_screen_ratio"] = (sc > 0).mean()
        else:
            row["slp_screen_on"] = row["slp_screen_ratio"] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


def extract_sleep_light_v9(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()
    df_raw = df_raw.copy()
    df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    df_raw["date"] = df_raw["timestamp"].dt.normalize().dt.date
    feats = []
    for (sid, d), grp in df_raw.groupby(["subject_id", "date"]):
        row = {"subject_id": sid, "sleep_date": d}
        morn = grp[grp["timestamp"].dt.hour < 9]
        if len(morn) > 0:
            vals = morn["w_light"].dropna().values
            row["slp_wlight_mean"] = safe_mean(vals)
            row["slp_wlight_dark"] = (vals < 5).mean() if len(vals) > 0 else np.nan
            row["slp_wlight_light"] = (vals > 100).mean() if len(vals) > 0 else np.nan
        else:
            row["slp_wlight_mean"] = row["slp_wlight_dark"] = row["slp_wlight_light"] = np.nan
        feats.append(row)
    return pd.DataFrame(feats)


# ==============================================================================
# Training Dataset Builder
# ==============================================================================


def build_training_dataset(metrics_df: pd.DataFrame, sensor_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Building training dataset with SOTA v28 and Public-0.6003 unified feature sets...")

    if metrics_df is None or metrics_df.empty:
        raise ValueError("Metrics DataFrame is empty or None.")

    id_col = "subject_id"
    metrics_df = metrics_df.copy()
    if "user_id" in metrics_df.columns and "subject_id" not in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={"user_id": "subject_id"})
    if id_col not in metrics_df.columns:
        return metrics_df

    # 1. Base Aggregated Daily Features from previous SOTA pipeline
    logger.info("Extracting SOTA base daily features...")
    daily_features_sota = create_daily_features(sensor_data_dict)

    # Accept both official schemas (lifelog_date/sleep_date) and lightweight
    # local-test schemas (date or no date). If metrics lack dates, derive a
    # subject-level date from available daily sensor features.
    if "lifelog_date" not in metrics_df.columns:
        if "date" in metrics_df.columns:
            metrics_df["lifelog_date"] = metrics_df["date"]
        elif not daily_features_sota.empty and {id_col, "date"}.issubset(daily_features_sota.columns):
            first_dates = daily_features_sota.groupby(id_col)["date"].min()
            metrics_df["lifelog_date"] = metrics_df[id_col].map(first_dates)
        else:
            metrics_df["lifelog_date"] = "1970-01-01"
    if "sleep_date" not in metrics_df.columns:
        metrics_df["sleep_date"] = metrics_df["lifelog_date"]

    # Convert dates to datetime.date
    metrics_df["lifelog_date_dt"] = pd.to_datetime(metrics_df["lifelog_date"]).dt.date
    metrics_df["sleep_date_dt"] = pd.to_datetime(metrics_df["sleep_date"]).dt.date

    # 2. Extract SOTA location and app usage features
    logger.info("Extracting SOTA GPS and app usage features...")
    if "ch2025_mGps" in sensor_data_dict:
        gps_sota = extract_location_clustering_features(sensor_data_dict["ch2025_mGps"])
        if not gps_sota.empty:
            daily_features_sota = pd.merge(daily_features_sota, gps_sota, on=["subject_id", "date"], how="outer")

    if "ch2025_mUsageStats" in sensor_data_dict:
        usage_sota = extract_app_usage_features(sensor_data_dict["ch2025_mUsageStats"])
        if not usage_sota.empty:
            daily_features_sota = pd.merge(daily_features_sota, usage_sota, on=["subject_id", "date"], how="outer")

    # Make sure date column in daily_features_sota is datetime.date
    if not daily_features_sota.empty and "date" in daily_features_sota.columns:
        daily_features_sota["date"] = pd.to_datetime(daily_features_sota["date"]).dt.date

    # 3. Extract all daily features from public v9 SOTA notebook
    logger.info("Extracting public v9 baseline daily features...")
    v9_daily_dfs = []

    if "ch2025_mActivity" in sensor_data_dict:
        v9_daily_dfs.append(extract_activity_v9(sensor_data_dict["ch2025_mActivity"]))
    if "ch2025_wPedo" in sensor_data_dict:
        v9_daily_dfs.append(extract_pedo_v9(sensor_data_dict["ch2025_wPedo"]))
    if "ch2025_mScreenStatus" in sensor_data_dict:
        v9_daily_dfs.append(extract_screen_v9(sensor_data_dict["ch2025_mScreenStatus"]))
    if "ch2025_mLight" in sensor_data_dict:
        v9_daily_dfs.append(extract_light_v9(sensor_data_dict["ch2025_mLight"], "m_light", "mlight"))
    if "ch2025_wLight" in sensor_data_dict:
        v9_daily_dfs.append(extract_light_v9(sensor_data_dict["ch2025_wLight"], "w_light", "wlight"))
    if "ch2025_mACStatus" in sensor_data_dict:
        v9_daily_dfs.append(extract_ac_v9(sensor_data_dict["ch2025_mACStatus"]))
    if "ch2025_mGps" in sensor_data_dict:
        v9_daily_dfs.append(extract_gps_v9(sensor_data_dict["ch2025_mGps"]))
    if "ch2025_mUsageStats" in sensor_data_dict:
        v9_daily_dfs.append(extract_usage_v9(sensor_data_dict["ch2025_mUsageStats"]))
    if "ch2025_mWifi" in sensor_data_dict:
        v9_daily_dfs.append(extract_wifi_v9(sensor_data_dict["ch2025_mWifi"]))
    if "ch2025_mBle" in sensor_data_dict:
        v9_daily_dfs.append(extract_ble_v9(sensor_data_dict["ch2025_mBle"]))
    if "ch2025_mAmbience" in sensor_data_dict:
        v9_daily_dfs.append(extract_ambience_v9(sensor_data_dict["ch2025_mAmbience"]))

    daily_features_v9 = pd.DataFrame()
    for df in v9_daily_dfs:
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"]).dt.date
        if daily_features_v9.empty:
            daily_features_v9 = df
        else:
            daily_features_v9 = pd.merge(daily_features_v9, df, on=["subject_id", "date"], how="outer")

    # Combine SOTA and v9 daily features
    if daily_features_v9.empty:
        all_daily_features = daily_features_sota
    elif daily_features_sota.empty:
        all_daily_features = daily_features_v9
    else:
        all_daily_features = pd.merge(daily_features_sota, daily_features_v9, on=["subject_id", "date"], how="outer")

    # 4. Extract sleep features (aligned to sleep_date)
    logger.info("Extracting unified sleep features aligned to sleep_date...")
    sleep_features_list = []

    # (a) Previous SOTA sleep features
    if "ch2025_wHr" in sensor_data_dict:
        try:
            sleep_sota = extract_sleep_phase_features(sensor_data_dict["ch2025_wHr"])
            if not sleep_sota.empty:
                sleep_sota["sleep_date"] = pd.to_datetime(sleep_sota["sleep_date"]).dt.date
                sleep_features_list.append(sleep_sota)
        except Exception as e:
            logger.error(f"Error extracting SOTA sleep features: {e}")

    # (b) v9 sleep morning features
    if "ch2025_wHr" in sensor_data_dict:
        sleep_features_list.append(extract_sleep_hr_v9(sensor_data_dict["ch2025_wHr"]))
    if "ch2025_wPedo" in sensor_data_dict:
        sleep_features_list.append(extract_sleep_pedo_v9(sensor_data_dict["ch2025_wPedo"]))
    if "ch2025_mActivity" in sensor_data_dict:
        sleep_features_list.append(extract_sleep_activity_v9(sensor_data_dict["ch2025_mActivity"]))
    if "ch2025_mScreenStatus" in sensor_data_dict:
        sleep_features_list.append(extract_sleep_screen_v9(sensor_data_dict["ch2025_mScreenStatus"]))
    if "ch2025_wLight" in sensor_data_dict:
        sleep_features_list.append(extract_sleep_light_v9(sensor_data_dict["ch2025_wLight"]))

    all_sleep_features = pd.DataFrame()
    for df in sleep_features_list:
        if df.empty:
            continue
        df["sleep_date"] = pd.to_datetime(df["sleep_date"]).dt.date
        if all_sleep_features.empty:
            all_sleep_features = df
        else:
            all_sleep_features = pd.merge(all_sleep_features, df, on=["subject_id", "sleep_date"], how="outer")

    # 5. Merge all feature matrices into metrics_df
    logger.info("Merging daily and sleep feature sets with label metrics...")
    final_df = metrics_df.copy()

    if not all_daily_features.empty:
        final_df = pd.merge(
            final_df,
            all_daily_features,
            left_on=["subject_id", "lifelog_date_dt"],
            right_on=["subject_id", "date"],
            how="left",
        )
        final_df = final_df.drop(columns=["date"], errors="ignore")

    if not all_sleep_features.empty:
        final_df = pd.merge(
            final_df,
            all_sleep_features,
            left_on=["subject_id", "sleep_date_dt"],
            right_on=["subject_id", "sleep_date"],
            how="left",
            suffixes=("", "_sleep_dup"),
        )
        final_df = final_df.drop(columns=["sleep_date_sleep_dup"], errors="ignore")

    # Drop temporary merge keys
    final_df = final_df.drop(columns=["lifelog_date_dt", "sleep_date_dt"], errors="ignore")

    # SOTA-v28: Restore chronological date column used in engineering pipeline
    final_df["date"] = pd.to_datetime(final_df["lifelog_date"]).dt.date

    # 6. Add External Calendar Features (Holidays & Weekends)
    try:
        import holidays

        kr_holidays = holidays.KR()
        dates = pd.to_datetime(final_df["date"])
        final_df["dayofweek"] = dates.dt.dayofweek
        final_df["is_weekend"] = final_df["dayofweek"].isin([5, 6]).astype(int)
        final_df["is_holiday"] = dates.apply(lambda x: int(x in kr_holidays))
        final_df["is_rest_day"] = np.maximum(final_df["is_weekend"], final_df["is_holiday"])
        logger.info("Successfully added external calendar features (weekend, holiday, rest_day).")
    except Exception as e:
        logger.warning(f"Error adding calendar features: {e}")

    # 7. Add External Weather Features
    import os

    weather_path = "data/external/weather_seoul.csv"
    if os.path.exists(weather_path):
        try:
            weather_df = pd.read_csv(weather_path)
            weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
            final_df["date"] = pd.to_datetime(final_df["date"]).dt.date
            final_df = pd.merge(final_df, weather_df, on="date", how="left")
            logger.info("Successfully merged external weather features.")
        except Exception as e:
            logger.error(f"Error merging weather features: {e}")

    # 8. Add official static subject demographics from the ETRI paper.
    # These are subject-level constants disclosed in docs/domain/etri_lifelog_2024_paper.md.
    final_df = add_official_demographics(final_df, id_col=id_col)
    logger.info("Successfully merged official ETRI paper demographic features.")

    return final_df


def apply_transductive_subject_normalization(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_cols: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("Applying Transductive Subject-specific Z-Score Normalization...")

    train_copy = train_df.copy()
    test_copy = test_df.copy()

    id_cols = ["subject_id", "sleep_date", "lifelog_date", "date"]
    exclude_cols = set(id_cols + target_cols)

    full_df = pd.concat([train_copy, test_copy], axis=0, ignore_index=True)

    numeric_cols = [c for c in full_df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

    group_mean = full_df.groupby("subject_id")[numeric_cols].transform("mean")
    group_std = full_df.groupby("subject_id")[numeric_cols].transform("std").replace(0, np.nan)

    normalized_numeric = (full_df[numeric_cols] - group_mean) / (group_std + 1e-6)
    for col in numeric_cols:
        full_df[f"{col}_subject_zscore"] = normalized_numeric[col]

    train_norm = full_df.iloc[: len(train_copy)].copy().reset_index(drop=True)
    test_norm = full_df.iloc[len(train_copy) :].copy().reset_index(drop=True)

    for col in target_cols:
        if col in test_norm.columns and col in test_copy.columns:
            test_norm[col] = test_copy[col].values

    logger.info(f"Subject Normalization complete. Normalized {len(numeric_cols)} features.")
    return train_norm, test_norm
