from functools import reduce
import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)

def process_hr_stat(x, stat='mean'):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            if stat == 'mean': return np.mean(x)
            if stat == 'max': return np.max(x)
            if stat == 'min': return np.min(x)
            if stat == 'std': return np.std(x)
        return np.nan
    except:
        return np.nan

def process_usage_array(x):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            return sum(item.get('total_time', 0) for item in x if isinstance(item, dict))
        return 0
    except:
        return 0

def process_usage_dict(x):
    """Returns a dict of app_name: total_time"""
    app_dict = {}
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            for item in x:
                if isinstance(item, dict):
                    name = item.get('app_name', 'unknown')
                    # clean up names slightly
                    name = str(name).strip().lower()
                    app_dict[name] = app_dict.get(name, 0) + item.get('total_time', 0)
    except:
        pass
    return app_dict

def process_ambience_dict(x):
    """Returns a dict of sound_label: max_probability"""
    amb_dict = {}
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            for item in x:
                # item is like array(['Music', '0.30902618'], dtype=object)
                if isinstance(item, (list, np.ndarray)) and len(item) == 2:
                    label = str(item[0]).strip().lower()
                    try:
                        prob = float(item[1])
                        # Keep the max probability seen in this specific record
                        amb_dict[label] = max(amb_dict.get(label, 0.0), prob)
                    except:
                        pass
    except:
        pass
    return amb_dict

def process_hr_rmssd(x):
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 1:
            diffs = np.diff(x)
            rmssd = np.sqrt(np.mean(diffs**2))
            return rmssd
        return 0.0
    except:
        return 0.0

def create_daily_features(sensor_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Creating daily features from sensor data...")
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
            df["hr_mean"] = df["heart_rate"].apply(lambda x: process_hr_stat(x, 'mean'))
            df["hr_max"] = df["heart_rate"].apply(lambda x: process_hr_stat(x, 'max'))
            df["hr_min"] = df["heart_rate"].apply(lambda x: process_hr_stat(x, 'min'))
            df["hr_rmssd"] = df["heart_rate"].apply(process_hr_rmssd)
            agg_dict = {
                "hr_mean": ["mean", "std"],
                "hr_max": ["max"],
                "hr_min": ["min"],
                "hr_rmssd": ["mean", "max"]
            }
        elif sensor_name in ["ch2025_mLight", "ch2025_wLight"]:
            # Usually has lux or similar light intensity
            light_col = "light" if "light" in df.columns else "m_light" if "m_light" in df.columns else "w_light" if "w_light" in df.columns else None
            if light_col:
                # If it's a list
                df["light_mean"] = df[light_col].apply(lambda x: np.mean(x) if isinstance(x, (list, np.ndarray)) and len(x)>0 else x if pd.api.types.is_numeric_dtype(type(x)) else 0)
                df["light_max"] = df[light_col].apply(lambda x: np.max(x) if isinstance(x, (list, np.ndarray)) and len(x)>0 else x if pd.api.types.is_numeric_dtype(type(x)) else 0)
                agg_dict = {"light_mean": ["mean", "max"], "light_max": ["max"]}
            else:
                numeric_cols = [c for c in df.select_dtypes(include="number").columns.tolist() if c not in ["subject_id", "timestamp", "date"]]
                agg_dict = {col: ["mean", "max"] for col in numeric_cols}
        elif sensor_name == "ch2025_mUsageStats":
            df["total_usage_time"] = df["m_usage_stats"].apply(process_usage_array)
            df["num_apps"] = df["m_usage_stats"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            
            # Targeted App Extraction
            target_apps = ['youtube', 'kakao', 'instagram', 'naver', 'chrome']
            for app in target_apps:
                df[f"app_{app}"] = df["m_usage_stats"].apply(
                    lambda x: sum(i.get('total_time', 0) for i in x if isinstance(i, dict) and app in str(i.get('app_name', '')).lower()) 
                    if isinstance(x, (list, np.ndarray)) else 0
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
            agg_dict = {"m_screen_use": ["sum", "mean"]}
        elif sensor_name == "ch2025_mAmbience":
            df["ambience_changes"] = df["m_ambience"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            
            # Targeted Sound Extraction
            target_sounds = ['speech', 'music', 'vehicle', 'silence', 'outdoor']
            for snd in target_sounds:
                df[f"snd_{snd}"] = df["m_ambience"].apply(
                    lambda x: max([float(i[1]) for i in x if isinstance(i, (list, np.ndarray)) and len(i)==2 and snd in str(i[0]).lower()] + [0.0])
                    if isinstance(x, (list, np.ndarray)) else 0.0
                )
                
            agg_dict = {"ambience_changes": ["sum", "mean"]}
            for snd in target_sounds:
                agg_dict[f"snd_{snd}"] = ["max", "mean"]
        elif sensor_name == "ch2025_mBle":
            df["ble_devices"] = df["m_ble"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            agg_dict = {"ble_devices": ["sum", "mean", "max"]}
        elif sensor_name == "ch2025_mWifi":
            df["wifi_devices"] = df["m_wifi"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            agg_dict = {"wifi_devices": ["sum", "mean", "max"]}
        elif sensor_name == "ch2025_mGps":
            df["gps_records"] = df["m_gps"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
            agg_dict = {"gps_records": ["sum", "mean"]}
        else:
            # Default numeric aggregation
            numeric_cols = [c for c in df.select_dtypes(include="number").columns.tolist() if c not in ["subject_id", "timestamp", "date"]]
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

def build_training_dataset(metrics_df: pd.DataFrame, sensor_data_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    logger.info("Building training dataset...")

    if metrics_df is None or metrics_df.empty:
        raise ValueError("Metrics DataFrame is empty or None.")

    features_df = create_daily_features(sensor_data_dict)

    if features_df.empty:
        logger.warning("Features DataFrame is empty. Returning only metrics.")
        return metrics_df

    metrics_df = metrics_df.copy()

    if "user_id" in metrics_df.columns and "subject_id" not in metrics_df.columns:
        metrics_df = metrics_df.rename(columns={"user_id": "subject_id"})

    id_col = "subject_id"
    if id_col not in metrics_df.columns:
        return metrics_df

    if "lifelog_date" in metrics_df.columns:
        metrics_df["date"] = pd.to_datetime(metrics_df["lifelog_date"]).dt.date
        final_df = pd.merge(metrics_df, features_df, on=[id_col, "date"], how="left")
    elif "date" in metrics_df.columns:
        metrics_df["date"] = pd.to_datetime(metrics_df["date"]).dt.date
        final_df = pd.merge(metrics_df, features_df, on=[id_col, "date"], how="left")
    else:
        final_df = pd.merge(metrics_df, features_df, on=[id_col], how="left")

    return final_df
