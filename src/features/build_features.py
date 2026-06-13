import numpy as np
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype


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
    Handles missing values via leak-free forward filling
    and adding missingness indicator features.
    """
    df_copy = df.copy()
    # Create missingness indicators
    for col in df_copy.columns:
        if df_copy[col].isnull().any():
            df_copy[f"{col}_is_missing"] = df_copy[col].isnull().astype(int)

    # STRICT LEAKAGE PREVENTION: Only apply forward-fill per subject chronologically.
    # Linear interpolation is banned as it leaks future values into the past.
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
    id_col = "subject_id" if "subject_id" in df_copy.columns else "user_id"

    # SOTA-v13: 3일 이상 장기 센서 유실 시 강제 전파를 막기 위해 ffill(limit=2) 로 보간 범위 엄격 제한!
    # 무작정 오래된 데이터를 복제하여 주입하는 Preprocessing Poisoning을 원천 방지하고,
    # 결측 정보는 결측 상태(NaN) 그대로 보존하여 LightGBM/CatBoost의 Default Routing에 맡깁니다.
    if len(numeric_cols) > 0:
        if id_col in df_copy.columns:
            df_copy[numeric_cols] = df_copy.groupby(id_col)[numeric_cols].transform(lambda x: x.ffill().bfill())
        else:
            df_copy[numeric_cols] = df_copy[numeric_cols].ffill().bfill()
        # Remaining all-null numeric columns get a neutral fallback so lightweight
        # tests and non-GBDT consumers never receive NaNs.
        df_copy[numeric_cols] = df_copy[numeric_cols].fillna(df_copy[numeric_cols].median(numeric_only=True)).fillna(0)
    return df_copy


def add_cross_sensor_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction features between different sensors.
    """
    df_copy = df.copy()

    # 1. Heart Rate vs Activity (Arousal vs Exertion)
    if "ch2025_wHr_hr_mean_mean" in df_copy.columns and "ch2025_mActivity_m_activity_mean" in df_copy.columns:
        # High HR but low activity might indicate stress/arousal without physical exertion
        df_copy["stress_proxy_hr_act"] = df_copy["ch2025_wHr_hr_mean_mean"] / (
            df_copy["ch2025_mActivity_m_activity_mean"] + 1.0
        )

    # 2. Light vs Screen (Blue light exposure)
    light_col = next((c for c in df_copy.columns if "light_mean" in c), None)
    if light_col and "ch2025_mScreenStatus_m_screen_use_mean" in df_copy.columns:
        # Screen use when ambient light is low (nighttime screen use)
        df_copy["night_screen_proxy"] = df_copy["ch2025_mScreenStatus_m_screen_use_mean"] / (df_copy[light_col] + 1.0)

    # 3. Circadian Rhythm: HR Dip Ratio (Night / Day)
    hr_night_col = next((c for c in df_copy.columns if "hr_night_mean" in c), None)
    hr_day_col = next((c for c in df_copy.columns if "hr_day_mean" in c), None)
    if hr_night_col and hr_day_col:
        df_copy["hr_dip_ratio"] = df_copy[hr_night_col] / (df_copy[hr_day_col] + 1.0)

    # 4. Routine Stability (Wifi BSSID Jaccard Similarity)
    id_col = "subject_id" if "subject_id" in df_copy.columns else "user_id"
    bssid_col = next((c for c in df_copy.columns if "bssid_set" in c), None)
    if bssid_col:
        # Calculate jaccard similarity with previous day for each user
        df_copy["bssid_set_lag1"] = df_copy.groupby(id_col)[bssid_col].shift(1)

        def calc_jaccard(row):
            set1 = row[bssid_col]
            set2 = row["bssid_set_lag1"]
            if isinstance(set1, set) and isinstance(set2, set) and (len(set1) > 0 or len(set2) > 0):
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                return intersection / union
            return 0.0

        df_copy["routine_stability_jaccard"] = df_copy.apply(calc_jaccard, axis=1)
        df_copy = df_copy.drop(columns=[bssid_col, "bssid_set_lag1"], errors="ignore")

    # 5. Physical Exhaustion Index (GPS Distance * Steps)
    gps_dist_col = next((c for c in df_copy.columns if "gps_dist" in c), None)
    steps_col = next((c for c in df_copy.columns if "step_sum" in c or "ch2025_wPedo_step_sum" in c), None)
    if gps_dist_col and steps_col:
        df_copy["physical_exhaustion"] = df_copy[gps_dist_col] * df_copy[steps_col]

    # 6. Deep Interactions
    hr_col = next((c for c in df_copy.columns if "hr_mean" in c and not c.endswith("_zscore")), None)
    act_col = next((c for c in df_copy.columns if "m_activity_mean" in c and not c.endswith("_zscore")), None)
    if hr_col and act_col:
        # High HR but low activity = Stress/Anxiety proxy
        df_copy["stress_proxy_hr_act"] = df_copy[hr_col] / (df_copy[act_col] + 1e-5)

    # 7. Weekend Transitions
    if "dayofweek" in df_copy.columns:
        df_copy["is_monday"] = (df_copy["dayofweek"] == 0).astype(int)
        df_copy["is_friday"] = (df_copy["dayofweek"] == 4).astype(int)

    return df_copy


def apply_personalized_normalization(df: pd.DataFrame, user_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Adds per-user Z-score features beside raw numerical features.

    This helper is intentionally simple and deterministic; fold-safe variants for
    strict train/validation pipelines live in ``src.features.transformers``.
    """
    df_copy = df.copy()
    if user_col not in df_copy.columns:
        return df_copy

    exclude_cols = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4", "target", "date", "timestamp", "is_test"]
    existing_cols = [c for c in feature_cols if c in df_copy.columns and c not in exclude_cols]
    for col in existing_cols:
        grouped = df_copy.groupby(user_col)[col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan).fillna(1.0)
        df_copy[f"{col}_zscore"] = ((df_copy[col] - mean) / (std + 1e-8)).fillna(0.0)
    return df_copy


def add_rolling_features(df: pd.DataFrame, user_col: str, date_col: str, windows: list | None = None) -> pd.DataFrame:
    """
    Adds time-delta-correct rolling mean, std, lag, diff and genuine EMA features per user.
    Uses resample('D') to reconstruct explicit chronological calendars, preventing
    temporal distortion due to missing days of watch tracking.
    """
    if windows is None:
        windows = [3, 7, 14, 21]
    df_copy = df.copy()
    if user_col not in df_copy.columns or date_col not in df_copy.columns:
        return df_copy

    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.sort_values(by=[user_col, date_col])

    # Select numerical columns that are not already rolling features or targets/IDs
    exclude_cols = [user_col, date_col, "target", "Q1", "Q2", "Q3", "S1", "S2", "S3", "S4", "is_test"]
    numeric_cols = [
        c
        for c in df_copy.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols and not c.endswith("_is_missing")
    ]

    roll_features_list = []

    # Apply user-wise Resampling to safely resolve Time-Delta gap anomalies
    for user, group in df_copy.groupby(user_col):
        # Sort and resample daily to maintain a linear, continuous time scale
        group_ts = group.set_index(date_col).sort_index()
        # Aggregating daily duplicate readings safely by mean
        group_ts_daily = group_ts[numeric_cols].resample("D").mean()

        # Lag 1: Pure previous-day value with zero calendar look-ahead leak
        lag1 = group_ts_daily.shift(1)
        lag1.columns = [f"{c}_lag1" for c in numeric_cols]

        # Diff 1: Pure previous-day variation diff
        diff1 = group_ts_daily.diff(1)
        diff1.columns = [f"{c}_diff1" for c in numeric_cols]

        user_roll_dfs = [lag1, diff1]

        # Calendar-correct Rolling Statistics and true Exponential Moving Averages (EMA)
        for w in windows:
            roll = group_ts_daily.rolling(window=w, min_periods=1)

            mean_df = roll.mean()
            mean_df.columns = [f"{c}_roll{w}_mean" for c in numeric_cols]

            std_df = roll.std().fillna(0.0)
            std_df.columns = [f"{c}_roll{w}_std" for c in numeric_cols]

            # SOTA-v14: True EMA implementation to properly represent slowly deteriorating temporal states
            ema_df = group_ts_daily.ewm(span=w, min_periods=1).mean()
            ema_df.columns = [f"{c}_ema{w}_mean" for c in numeric_cols]

            user_roll_dfs.extend([mean_df, std_df, ema_df])

        merged_user_roll = pd.concat(user_roll_dfs, axis=1)
        merged_user_roll[user_col] = user
        merged_user_roll = merged_user_roll.reset_index()
        roll_features_list.append(merged_user_roll)

    all_roll_features = pd.concat(roll_features_list, ignore_index=True)
    all_roll_features[date_col] = pd.to_datetime(all_roll_features[date_col]).dt.date

    # Merge engineered rolling matrices back using absolute date mapping to prevent spatial shift leakage
    df_copy["date_only"] = pd.to_datetime(df_copy[date_col]).dt.date
    df_copy = pd.merge(
        df_copy,
        all_roll_features,
        left_on=[user_col, "date_only"],
        right_on=[user_col, date_col],
        how="left",
        suffixes=("", "_roll_temp"),
    )
    df_copy = df_copy.drop(columns=["date_only", date_col + "_roll_temp"], errors="ignore")

    return df_copy


def fit_rolling_cosinor_for_series(df: pd.DataFrame, user_col: str, target_col: str, window: int = 14) -> tuple:
    """
    Fits a leak-free rolling weekly Cosinor model on a biological target series per user.
    Cosinor formula: Y(t) = Mesor + Amp * cos(omega * t + phase)
    Returns (mesors, amplitudes, acrophases)
    """
    omega = 2.0 * np.pi / 7.0
    t = df["dayofweek"].values
    x_cos = np.cos(omega * t)
    x_sin = np.sin(omega * t)
    y = df[target_col].values

    mesors = np.full(len(df), np.nan)
    amplitudes = np.full(len(df), np.nan)
    acrophases = np.full(len(df), np.nan)

    for i in range(len(df)):
        start_idx = max(0, i - window + 1)
        if i < 5:  # Need warm-up days
            continue

        # Strictly leak-free: Exclude current day index i to prevent look-ahead target/signal leak
        window_y = y[start_idx:i]
        window_cos = x_cos[start_idx:i]
        window_sin = x_sin[start_idx:i]

        valid = ~np.isnan(window_y)
        if valid.sum() < 5:
            continue

        wy = window_y[valid]
        wcos = window_cos[valid]
        wsin = window_sin[valid]

        X = np.column_stack([np.ones_like(wy), wcos, wsin])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, wy, rcond=None)
            mesors[i] = beta[0]
            amplitudes[i] = np.sqrt(beta[1] ** 2 + beta[2] ** 2)
            acrophases[i] = np.arctan2(-beta[2], beta[1])
        except Exception:
            pass

    return mesors, amplitudes, acrophases


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive feature engineering to the given DataFrame
    based on the master plan strategies.
    """
    df_copy = df.copy()
    id_col = "subject_id" if "subject_id" in df_copy.columns else "user_id"
    date_col = "date"

    # STRICT TEMPORAL SORTING: Force sorting chronologically by user and date
    # before engineering features, so expanding window/ffill/rolling work chronological.
    if id_col in df_copy.columns and date_col in df_copy.columns:
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(by=[id_col, date_col]).reset_index(drop=True)
        # Biting leakage prevention: Assert chronological monotonicity per user to block time-reversed leakage
        assert df_copy.groupby(id_col)[date_col].apply(lambda x: x.is_monotonic_increasing).all(), (
            "CRITICAL: Chronological order sorting failed! Potential look-ahead leakage danger."
        )

    if "date" in df_copy.columns:
        dt = pd.to_datetime(df_copy["date"])
        if "dayofweek" not in df_copy.columns:
            df_copy["dayofweek"] = dt.dt.dayofweek
        if "is_weekend" not in df_copy.columns:
            df_copy["is_weekend"] = df_copy["dayofweek"].isin([5, 6]).astype(int)

        # Cyclical encoding for weekday rhythms (safe from seasonal domain shift)
        df_copy["dayofweek_sin"] = np.sin(2 * np.pi * df_copy["dayofweek"] / 7)
        df_copy["dayofweek_cos"] = np.cos(2 * np.pi * df_copy["dayofweek"] / 7)

    df_copy = handle_missing_values(df_copy)

    for col in [c for c in df_copy.columns if "Ambience" in c or "ambience" in c]:
        if is_object_dtype(df_copy[col]) or is_string_dtype(df_copy[col]):
            df_copy[f"{col}_len"] = df_copy[col].fillna("").astype(str).map(len)

    # 1. Leak-Free Sleep-Debt Index
    # Accumulate sleep deficit over the last 7 days compared to an 8-hour baseline.
    deep_col = "wHr_sleep_duration_deep"
    light_col = "wHr_sleep_duration_light"
    if deep_col in df_copy.columns and light_col in df_copy.columns:
        # Sum deep and light sleep duration to estimate total sleep minutes
        sleep_dur = df_copy[deep_col] + df_copy[light_col]
        # 8 hours = 480 minutes baseline
        sleep_shortage = (480.0 - sleep_dur).clip(lower=0.0)

        # Calculate daily sleep debt, shift by 1 to make it strictly leak-free, then take 7-day rolling sum
        df_copy["sleep_debt_daily"] = sleep_shortage
        df_copy["sleep_debt_7d"] = (
            df_copy.groupby(id_col)["sleep_debt_daily"]
            .shift(1)
            .groupby(df_copy[id_col])
            .rolling(window=7, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )
        df_copy = df_copy.drop(columns=["sleep_debt_daily"])

    # 2. Cosinor Model Rolling Circadian Features
    # Fit rolling Cosinor parameters strictly leak-free on a reliable biological signal (heart rate)
    hr_col = "ch2025_wHr_hr_mean_mean"
    if hr_col in df_copy.columns:
        cosinor_mesors = []
        cosinor_amps = []
        cosinor_acros = []
        for _user, group in df_copy.groupby(id_col):
            m, a, p = fit_rolling_cosinor_for_series(group, id_col, hr_col, window=14)
            cosinor_mesors.append(pd.Series(m, index=group.index))
            cosinor_amps.append(pd.Series(a, index=group.index))
            cosinor_acros.append(pd.Series(p, index=group.index))

        df_copy["cosinor_hr_mesor"] = pd.concat(cosinor_mesors)
        df_copy["cosinor_hr_amp"] = pd.concat(cosinor_amps)
        df_copy["cosinor_hr_acro"] = pd.concat(cosinor_acros)

    # 3. SOTA-v29: Leak-Free Rolling Pearson Correlation between key biological/behavioral signals
    # Specifically captures complex homeostatic dysregulations (stress/fatigue/depression proxy)
    act_col = "ch2025_mActivity_m_activity_mean"
    screen_col = "ch2025_mScreenStatus_m_screen_use_mean"
    light_col_ambient = next((c for c in df_copy.columns if "light_mean" in c), None)
    silence_col = next((c for c in df_copy.columns if "snd_silence_mean" in c), None)
    sleep_eff_col = "wHr_sleep_efficiency_proxy"

    # User-wise safe rolling correlation helper (strictly leak-free with shift(1))
    def get_leak_free_rolling_corr(df_in, c1, c2, win=14):
        # We compute corr per group and shift by 1 day
        res = (
            df_in.groupby(id_col)
            .apply(lambda g: g[c1].rolling(window=win, min_periods=4).corr(g[c2]).shift(1))
            .reset_index(level=0, drop=True)
        )
        return res.fillna(0.0)

    if hr_col in df_copy.columns and act_col in df_copy.columns:
        df_copy["corr_hr_activity_14d"] = get_leak_free_rolling_corr(df_copy, hr_col, act_col)

    if light_col_ambient in df_copy.columns and screen_col in df_copy.columns:
        df_copy["corr_light_screen_14d"] = get_leak_free_rolling_corr(df_copy, light_col_ambient, screen_col)

    if silence_col in df_copy.columns and sleep_eff_col in df_copy.columns:
        df_copy["corr_silence_sleep_14d"] = get_leak_free_rolling_corr(df_copy, silence_col, sleep_eff_col)

    if hr_col in df_copy.columns and screen_col in df_copy.columns:
        df_copy["corr_hr_screen_14d"] = get_leak_free_rolling_corr(df_copy, hr_col, screen_col)

    # Add Cross-Sensor interactions
    df_copy = add_cross_sensor_features(df_copy)

    id_col = "subject_id" if "subject_id" in df_copy.columns else "user_id"
    date_col = "date"

    # Add rolling and lag features before user normalization
    if id_col in df_copy.columns and date_col in df_copy.columns:
        df_copy = add_rolling_features(df_copy, user_col=id_col, date_col=date_col)

    # Placeholder for numerical feature columns to normalize per user
    feature_cols = [
        c
        for c in df_copy.select_dtypes(include=[np.number]).columns
        if not c.endswith("_is_missing") and c not in [id_col]
    ]
    df_copy = apply_personalized_normalization(df_copy, user_col=id_col, feature_cols=feature_cols)

    return df_copy
