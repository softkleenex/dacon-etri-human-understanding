import numpy as np
import pandas as pd


def aggregate_time_series(df: pd.DataFrame, time_col: str, agg_col: str, freq: str = "D") -> pd.DataFrame:
    """
    Aggregates time-series data at a given frequency.
    """
    df_copy = df.copy()
    if time_col in df_copy.columns:
        df_copy[time_col] = pd.to_datetime(df_copy[time_col])
        df_copy = df_copy.set_index(time_col)
        # Assuming numeric columns for simple aggregation
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        df_agg = df_copy[numeric_cols].resample(freq).mean().reset_index()
        return df_agg
    return df_copy


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values via interpolation, forward/backward filling,
    and adding missingness indicator features.
    """
    df_copy = df.copy()
    # Create missingness indicators
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            df_copy[f"{col}_is_missing"] = df_copy[col].isnull().astype(int)

    # Interpolate time-series like data if applicable, then ffill
    # Only apply to numeric columns. STRICT LEAKAGE PREVENTION: limit_direction="forward"
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    df_copy[numeric_cols] = df_copy[numeric_cols].interpolate(method="linear", limit_direction="forward")
    df_copy[numeric_cols] = df_copy[numeric_cols].ffill().fillna(0) # fillna(0) for leading NaNs
    return df_copy


def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction features between different sensors.
    """
    df_copy = df.copy()
    
    # 1. Heart Rate vs Activity (Arousal vs Exertion)
    if 'ch2025_wHr_hr_mean_mean' in df_copy.columns and 'ch2025_mActivity_m_activity_mean' in df_copy.columns:
        # High HR but low activity might indicate stress/arousal without physical exertion
        df_copy['stress_proxy_hr_act'] = df_copy['ch2025_wHr_hr_mean_mean'] / (df_copy['ch2025_mActivity_m_activity_mean'] + 1.0)
        
    # 2. Light vs Screen (Blue light exposure)
    light_col = next((c for c in df_copy.columns if 'light_mean' in c), None)
    if light_col and 'ch2025_mScreenStatus_m_screen_use_mean' in df_copy.columns:
        # Screen use when ambient light is low (nighttime screen use)
        df_copy['night_screen_proxy'] = df_copy['ch2025_mScreenStatus_m_screen_use_mean'] / (df_copy[light_col] + 1.0)
        
    return df_copy


def apply_personalized_normalization(df: pd.DataFrame, user_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Normalizes features per user (Z-score normalization).
    Uses expanding window to prevent data leakage from future days.
    Preserves raw features for objective targets (S1-S4) while providing
    Z-scores for subjective targets (Q1-Q3).
    """
    df_copy = df.copy()
    if user_col in df_copy.columns:
        # DO NOT normalize target columns
        exclude_cols = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4", "target", "date", "timestamp"]
        existing_cols = [c for c in feature_cols if c in df_copy.columns and c not in exclude_cols]
        
        if existing_cols:
            grouped = df_copy.groupby(user_col)[existing_cols]
            # Expanding window ensures day N is normalized only using days 0 to N
            means = grouped.expanding(min_periods=1).mean().reset_index(level=0, drop=True)
            stds = grouped.expanding(min_periods=1).std().reset_index(level=0, drop=True)
            
            # For the first day, stds will be NaN. We must fill it with 1.0 (or 0) so z-score doesn't become NaN
            stds = stds.fillna(1.0)
            
            # Avoid division by zero
            zscores = (df_copy[existing_cols] - means) / (stds + 1e-8)
            zscores.columns = [f"{col}_zscore" for col in existing_cols]
            
            # Concat z-scores alongside raw features
            df_copy = pd.concat([df_copy, zscores], axis=1)
    return df_copy


def add_rolling_features(df: pd.DataFrame, user_col: str, date_col: str, windows: list = [3, 7]) -> pd.DataFrame:
    """
    Adds rolling mean, std, and lag features per user to capture temporal dependencies.
    """
    df_copy = df.copy()
    if user_col not in df_copy.columns or date_col not in df_copy.columns:
        return df_copy

    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.sort_values(by=[user_col, date_col])

    # Select numerical columns that are not already rolling features or targets/IDs
    exclude_cols = [user_col, date_col, "target", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    numeric_cols = [c for c in df_copy.select_dtypes(include=[np.number]).columns if c not in exclude_cols and not c.endswith("_is_missing")]

    grouped = df_copy.groupby(user_col)[numeric_cols]

    # Calculate lag 1 (previous day's value)
    lag_df = grouped.shift(1)
    lag_df.columns = [f"{c}_lag1" for c in numeric_cols]
    df_copy = pd.concat([df_copy, lag_df], axis=1)
    
    # Calculate difference from previous day
    diff_df = grouped.diff(1)
    diff_df.columns = [f"{c}_diff1" for c in numeric_cols]
    df_copy = pd.concat([df_copy, diff_df], axis=1)

    # Calculate rolling statistics
    for w in windows:
        roll = grouped.rolling(window=w, min_periods=1)
        
        roll_mean = roll.mean().reset_index(level=0, drop=True)
        roll_mean.columns = [f"{c}_roll{w}_mean" for c in numeric_cols]
        
        roll_std = roll.std().reset_index(level=0, drop=True)
        # Fill NaN for first day std
        roll_std = roll_std.fillna(0)
        roll_std.columns = [f"{c}_roll{w}_std" for c in numeric_cols]
        
        df_copy = pd.concat([df_copy, roll_mean, roll_std], axis=1)

    return df_copy


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive feature engineering to the given DataFrame
    based on the master plan strategies.
    """
    import os
    
    df_copy = df.copy()

    if "date" in df_copy.columns:
        dt = pd.to_datetime(df_copy["date"])
        df_copy["dayofweek"] = dt.dt.dayofweek
        df_copy["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        df_copy["month"] = dt.dt.month
        df_copy["day"] = dt.dt.day
        
        # Cyclical encoding
        df_copy['dayofweek_sin'] = np.sin(2 * np.pi * df_copy['dayofweek'] / 7)
        df_copy['dayofweek_cos'] = np.cos(2 * np.pi * df_copy['dayofweek'] / 7)
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)

    df_copy = handle_missing_values(df_copy)

    # Add Cross-Sensor interactions
    df_copy = add_cross_sensor_features(df_copy)

    id_col = "subject_id" if "subject_id" in df_copy.columns else "user_id"
    date_col = "date"
    
    # Add rolling and lag features before user normalization
    if id_col in df_copy.columns and date_col in df_copy.columns:
        df_copy = add_rolling_features(df_copy, user_col=id_col, date_col=date_col)

    # Placeholder for numerical feature columns to normalize per user
    feature_cols = [c for c in df_copy.select_dtypes(include=[np.number]).columns if not c.endswith("_is_missing") and c not in [id_col]]
    df_copy = apply_personalized_normalization(df_copy, user_col=id_col, feature_cols=feature_cols)

    return df_copy
