import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import DataLoader
from src.features.transformers import FoldWisePCA, LeakFreeSubjectZScaler
from src.utils.logger import get_logger
from src.utils.validation import make_subject_hole_folds

logger = get_logger(__name__)

# List of 46 hand-curated core domain features
CORE_FEATURES = [
    # Sleep quality & durations
    'wHr_deep_sleep_ratio', 'wHr_sleep_efficiency_proxy',
    'wHr_sleep_duration_deep', 'wHr_sleep_duration_light', 'wHr_sleep_duration_awake',
    'wHr_sleep_fragmentation_index', 'wHr_sleep_hr_mean', 'wHr_sleep_hr_std',
    'wHr_slp_hr_deep_ratio', 'wHr_slp_hr_awake_ratio', 'wHr_slp_hr_light_ratio',
    'wHr_slp_hr_rmssd', 'wHr_slp_hr_spike_count',
    # GPS spatial behavior
    'mGps_homestay_ratio', 'mGps_radius_of_gyration', 'mGps_spatial_entropy', 'mGps_unique_locations',
    # Smartphone App usage/entropy
    'mUsage_app_entropy', 'mUsage_dopamine_ratio', 'mUsage_hourly_volatility',
    'mUsage_night_time_ratio', 'mUsage_sessions_count', 'mUsage_total_time_mins',
    # Physical activity levels
    'act_active_ratio', 'act_still_ratio', 'pedo_total_steps', 'pedo_total_calories',
    'pedo_mean_speed', 'pedo_run_ratio', 'pedo_active_hours',
    # Night screen time & locks
    'screen_on_ratio', 'screen_unlock_cnt', 'screen_night_ratio', 'screen_presleep_ratio',
    # Ambient light & silence
    'mlight_all_mean', 'wlight_all_mean', 'mlight_dark_ratio', 'wlight_dark_ratio',
    # Phone charging state
    'ac_charging_ratio', 'ac_night_charging',
    # Calendar & date
    'dayofweek', 'is_weekend', 'is_holiday', 'is_rest_day'
]

CIRCADIAN_SEASONAL_FEATURES = [
    'circadian_hr_rmssd_ratio',
    'circadian_light_ratio',
    'circadian_screen_night_ratio',
    'circadian_act_still_ratio',
    'sin_dayofyear',
    'cos_dayofyear',
    'sin_month',
    'cos_month',
    'sin_dayofweek',
    'cos_dayofweek'
]

# Best Laplace Hyperparameters discovered by AgentGA Optuna Evolver
BEST_LAPLACE_PARAMS = {
    'Q1': {'half_life': 1.7617920093818988, 'prior_weight': 0.5806134931769096, 'm_smooth': 32.13976283360438},
    'Q2': {'half_life': 6.764748507172517, 'prior_weight': 0.42638074353735894, 'm_smooth': 18.952477341412223},
    'Q3': {'half_life': 1.0132235989421328, 'prior_weight': 0.5189342034344457, 'm_smooth': 4.941469880043686},
    'S1': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
    'S2': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
    'S3': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0},
    'S4': {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0}
}

class LeakFreeSubjectTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='subject_id', m_smooth=10.0):
        self.id_col = id_col
        self.m_smooth = m_smooth
        self.global_mean_ = 0.5
        self.subject_te_ = None

    def fit(self, X, y):
        y_series = pd.Series(y, index=X.index)
        self.global_mean_ = y_series.fillna(0.5).mean()
        non_null_mask = y_series.notna()
        group_sums = y_series[non_null_mask].groupby(X.loc[non_null_mask, self.id_col]).sum()
        group_counts = y_series[non_null_mask].groupby(X.loc[non_null_mask, self.id_col]).count()
        self.subject_te_ = (group_sums + self.m_smooth * self.global_mean_) / (group_counts + self.m_smooth)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["subject_target_encoded"] = X_copy[self.id_col].map(self.subject_te_).fillna(self.global_mean_)
        return X_copy

def add_clean_base_features(df):
    df_copy = df.copy()
    id_col = "subject_id" if "subject_id" in df_copy.columns else "user_id"

    # 1. Fillna of CORE_FEATURES using subject-wise median
    for col in CORE_FEATURES:
        if col in df_copy.columns:
            df_copy[col] = df_copy.groupby(id_col)[col].transform(lambda x: x.fillna(x.median()))
            df_copy[col] = df_copy[col].fillna(df_copy[col].median()).fillna(0.0)

    # 2. Add custom circadian dip ratio if not present
    if 'hr_dip_ratio' not in df_copy.columns:
        if 'ch2025_wHr_hr_night_mean' in df_copy.columns and 'ch2025_wHr_hr_day_mean' in df_copy.columns:
            df_copy['hr_dip_ratio'] = df_copy['ch2025_wHr_hr_night_mean'] / (df_copy['ch2025_wHr_hr_day_mean'] + 1.0)
        else:
            df_copy['hr_dip_ratio'] = 0.8

    df_copy['hr_dip_ratio'] = df_copy['hr_dip_ratio'].fillna(0.8)

    # 3. Add Circadian Biomarkers
    df_copy['circadian_hr_rmssd_ratio'] = df_copy['wHr_slp_hr_rmssd'] / (df_copy['wHr_sleep_hr_mean'] + 1.0)
    df_copy['circadian_light_ratio'] = df_copy['wlight_dark_ratio'] / (df_copy['wlight_all_mean'] + 1e-5)
    df_copy['circadian_screen_night_ratio'] = df_copy['screen_night_ratio'] / (df_copy['screen_on_ratio'] + 1e-5)
    df_copy['circadian_act_still_ratio'] = df_copy['act_night_still'] / (df_copy['act_still_ratio'] + 1e-5)

    # Fillna circadian biomarkers
    for col in ['circadian_hr_rmssd_ratio', 'circadian_light_ratio', 'circadian_screen_night_ratio', 'circadian_act_still_ratio']:
        if col in df_copy.columns:
            df_copy[col] = df_copy.groupby(id_col)[col].transform(lambda x: x.fillna(x.median()))
            df_copy[col] = df_copy[col].fillna(df_copy[col].median()).fillna(0.0)

    # 4. Add Seasonal/Temporal Biomarkers (cyclical sine-cosine)
    df_copy['date_parsed'] = pd.to_datetime(df_copy['date'])
    df_copy['dayofyear'] = df_copy['date_parsed'].dt.dayofyear
    df_copy['month'] = df_copy['date_parsed'].dt.month

    df_copy['sin_dayofyear'] = np.sin(2 * np.pi * df_copy['dayofyear'] / 365.25)
    df_copy['cos_dayofyear'] = np.cos(2 * np.pi * df_copy['dayofyear'] / 365.25)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy['month'] / 12.0)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy['month'] / 12.0)
    df_copy['sin_dayofweek'] = np.sin(2 * np.pi * df_copy['dayofweek'] / 7.0)
    df_copy['cos_dayofweek'] = np.cos(2 * np.pi * df_copy['dayofweek'] / 7.0)

    # Fillna seasonal biomarkers
    for col in CIRCADIAN_SEASONAL_FEATURES:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0.0)

    return df_copy

def add_clean_rolling_features(df, id_col='subject_id', date_col='date'):
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.sort_values(by=[id_col, date_col])

    key_cols = [
        'wHr_sleep_duration_deep', 'wHr_sleep_efficiency_proxy',
        'pedo_total_steps', 'mUsage_total_time_mins', 'screen_on_ratio',
        'circadian_hr_rmssd_ratio', 'circadian_light_ratio'
    ]
    key_cols = [c for c in key_cols if c in df_copy.columns]

    roll_list = []
    for user, group in df_copy.groupby(id_col):
        group_ts = group.set_index(date_col).sort_index()
        group_ts_daily = group_ts[key_cols].resample('D').mean()

        user_dfs = []
        for w in [3, 7]:
            # Rolling Mean (strictly leak-free)
            roll_mean = group_ts_daily.shift(1).rolling(window=w, min_periods=1).mean()
            roll_mean.columns = [f"{c}_roll{w}_mean" for c in key_cols]

            # EMA (strictly leak-free)
            roll_ema = group_ts_daily.shift(1).ewm(span=w, min_periods=1).mean()
            roll_ema.columns = [f"{c}_ema{w}_mean" for c in key_cols]

            user_dfs.extend([roll_mean, roll_ema])

        merged_user = pd.concat(user_dfs, axis=1)
        merged_user[id_col] = user
        merged_user = merged_user.reset_index()
        roll_list.append(merged_user)

    all_roll = pd.concat(roll_list, ignore_index=True)
    all_roll[date_col] = pd.to_datetime(all_roll[date_col]).dt.date

    df_copy['date_only'] = pd.to_datetime(df_copy[date_col]).dt.date
    df_copy = pd.merge(
        df_copy,
        all_roll,
        left_on=[id_col, 'date_only'],
        right_on=[id_col, date_col],
        how='left',
        suffixes=('', '_roll_temp')
    )
    df_copy = df_copy.drop(columns=['date_only', date_col + '_roll_temp'], errors='ignore')

    return df_copy

def add_temporal_target_features(df, y, train_idx, target_col, id_col, half_life=3.0, prior_weight=0.2):
    df_copy = df.copy()
    df_copy['original_index'] = df_copy.index

    # Isolate training fold labels
    y_train = pd.Series(np.nan, index=df_copy.index)
    y_train.loc[train_idx] = y.loc[train_idx]
    df_copy['y_train_temp'] = y_train

    df_copy['date_temp'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values(by=[id_col, 'date_temp']).reset_index(drop=True)

    # Shift to prevent target self-leakage
    df_copy['date_with_target'] = df_copy['date_temp'].where(df_copy['y_train_temp'].notna())

    df_copy['y_train_prev_source'] = df_copy.groupby(id_col)['y_train_temp'].shift(1)
    df_copy['date_prev_source'] = df_copy.groupby(id_col)['date_with_target'].shift(1)

    df_copy['y_prev'] = df_copy.groupby(id_col)['y_train_prev_source'].ffill()
    df_copy['date_prev'] = df_copy.groupby(id_col)['date_prev_source'].ffill()
    df_copy['delta_prev'] = (df_copy['date_temp'] - pd.to_datetime(df_copy['date_prev'])).dt.days

    df_copy['y_train_next_source'] = df_copy.groupby(id_col)['y_train_temp'].shift(-1)
    df_copy['date_next_source'] = df_copy.groupby(id_col)['date_with_target'].shift(-1)

    df_copy['y_next'] = df_copy.groupby(id_col)['y_train_next_source'].bfill()
    df_copy['date_next'] = df_copy.groupby(id_col)['date_next_source'].bfill()
    df_copy['delta_next'] = (pd.to_datetime(df_copy['date_next']) - df_copy['date_temp']).dt.days

    subject_means = df_copy.groupby(id_col)['y_train_temp'].transform('mean').fillna(0.5)

    # Apply Exponential Decay Weighted Interpolation
    decay_lambda = np.log(2.0) / half_life
    w_prev = np.exp(-decay_lambda * df_copy['delta_prev'].fillna(999.0))
    w_next = np.exp(-decay_lambda * df_copy['delta_next'].fillna(999.0))

    denom = w_prev + w_next + prior_weight
    df_copy['y_interp'] = (w_prev * df_copy['y_prev'].fillna(0.5) +
                           w_next * df_copy['y_next'].fillna(0.5) +
                           prior_weight * subject_means) / denom

    df_copy['y_prev'] = df_copy['y_prev'].fillna(subject_means)
    df_copy['y_next'] = df_copy['y_next'].fillna(subject_means)
    df_copy['y_interp'] = df_copy['y_interp'].fillna(subject_means)
    df_copy['delta_prev'] = df_copy['delta_prev'].fillna(999.0)
    df_copy['delta_next'] = df_copy['delta_next'].fillna(999.0)

    df_copy = df_copy.sort_values(by='original_index').reset_index(drop=True)

    features = pd.DataFrame(index=df.index)
    features[f'{target_col}_prev'] = df_copy['y_prev'].values
    features[f'{target_col}_next'] = df_copy['y_next'].values
    features[f'{target_col}_interp'] = df_copy['y_interp'].values
    features[f'{target_col}_log_delta_prev'] = np.log1p(df_copy['delta_prev']).values
    features[f'{target_col}_log_delta_next'] = np.log1p(df_copy['delta_next']).values

    return features

def apply_robust_directives(df_train_fold, df_test_fold, tr_idx, target, id_col, numeric_cols, y_train_fold):
    df_train_fold = df_train_fold.copy()
    df_test_fold = df_test_fold.copy()

    special_cols = []

    if target in ["S2", "S3", "S4"]:
        candidate_cols = [c for c in df_train_fold.columns if c.endswith("_subject_zscore")]
        corrs = {}
        for col in candidate_cols:
            c = df_train_fold.iloc[tr_idx][col].corr(y_train_fold.iloc[tr_idx])
            if pd.notna(c):
                corrs[col] = abs(c)
        top_5 = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_5_cols = [col for col, _ in top_5]

        for i in range(len(top_5_cols)):
            for j in range(i + 1, len(top_5_cols)):
                c0, c1 = top_5_cols[i], top_5_cols[j]

                # Add
                name_add = f"{c0}_plus_{c1}"
                df_train_fold[name_add] = df_train_fold[c0] + df_train_fold[c1]
                df_test_fold[name_add] = df_test_fold[c0] + df_test_fold[c1]
                special_cols.append(name_add)

                # Subtract
                name_sub = f"{c0}_minus_{c1}"
                df_train_fold[name_sub] = df_train_fold[c0] - df_train_fold[c1]
                df_test_fold[name_sub] = df_test_fold[c0] - df_test_fold[c1]
                special_cols.append(name_sub)

                # Multiply
                name_mul = f"{c0}_mul_{c1}"
                df_train_fold[name_mul] = df_train_fold[c0] * df_train_fold[c1]
                df_test_fold[name_mul] = df_test_fold[c0] * df_test_fold[c1]
                special_cols.append(name_mul)

    return df_train_fold, df_test_fold, special_cols

def run_clean_ultimate_pipeline_v31():
    logger.info("Initializing SOTA-v31 Curated Feature Pipeline with Circadian/Seasonal Biomarkers & Optimal Laplace Params...")

    # 1. Load Data
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    sample_sub_df = data_loader.load_sample_submission()

    processed_train_path = Path("data/processed/df_train_raw.parquet")
    processed_test_path = Path("data/processed/df_test_raw.parquet")

    df_train_raw = pd.read_parquet(processed_train_path)
    df_test_raw = pd.read_parquet(processed_test_path)

    df_train_raw['is_test'] = 0
    df_test_raw['is_test'] = 1
    df_all_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)

    # 2. Curated Base Feature Clean Engineering with Circadian & Seasonal Biomarkers
    df_all = add_clean_base_features(df_all_raw)
    df_all = add_clean_rolling_features(df_all, id_col='subject_id', date_col='date')

    df_train = df_all[df_all['is_test'] == 0].drop(columns=['is_test']).copy()
    df_test = df_all[df_all['is_test'] == 1].drop(columns=['is_test']).copy()

    id_col = "subject_id"
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

    # 3. Anchor Stacking: Incorporate DL Predictions
    lstm_oof_path = 'submissions/oof_dl_v10.csv'
    lstm_sub_path = 'submissions/submission_dl_v10.csv'

    base_feature_cols = []
    df_train['date'] = pd.to_datetime(df_train['date']).dt.date
    df_test['date'] = pd.to_datetime(df_test['date']).dt.date

    if os.path.exists(lstm_oof_path) and os.path.exists(lstm_sub_path):
        logger.info("Found DL sequence anchors. Merging...")
        base_oof = pd.read_csv(lstm_oof_path)
        base_oof['date'] = pd.to_datetime(base_oof['date']).dt.date
        df_train = pd.merge(df_train, base_oof, on=[id_col, 'date'], how='left', suffixes=('', '_base'))

        base_sub = pd.read_csv(lstm_sub_path)
        if 'date' not in base_sub.columns:
            if 'lifelog_date' in base_sub.columns:
                base_sub['date'] = base_sub['lifelog_date']
            elif 'sleep_date' in base_sub.columns:
                base_sub['date'] = base_sub['sleep_date']
        base_sub['date'] = pd.to_datetime(base_sub['date']).dt.date
        df_test = pd.merge(df_test, base_sub, on=[id_col, 'date'], how='left', suffixes=('', '_base'))

        eps = 1e-6
        for t in targets:
            base_col = f"{t}_base"
            if base_col in df_train.columns:
                prob = df_train[base_col].fillna(0.5).clip(eps, 1-eps)
                df_train[f'base_{t}_prob'] = prob
                df_train[f'base_{t}_logit'] = np.log(prob / (1 - prob))
            if base_col in df_test.columns:
                prob_t = df_test[base_col].fillna(0.5).clip(eps, 1-eps)
                df_test[f'base_{t}_prob'] = prob_t
                df_test[f'base_{t}_logit'] = np.log(prob_t / (1 - prob_t))
            base_feature_cols.extend([f'base_{t}_prob', f'base_{t}_logit'])

    # Setup CV
    n_splits = 5
    df_train = df_train.sort_values(by=[id_col, 'date']).reset_index(drop=True)
    df_test = df_test.sort_values(by=[id_col, 'date']).reset_index(drop=True)

    cv_folds = make_subject_hole_folds(df_train, n_folds=n_splits)

    # Active Core columns to scale per fold
    active_base_features = CORE_FEATURES + CIRCADIAN_SEASONAL_FEATURES + ['hr_dip_ratio']
    roll_cols = [c for c in df_train.columns if '_roll' in c or '_ema' in c]

    # Prediction arrays
    oof_preds_cat = np.zeros((len(df_train), len(targets)))
    oof_preds_lgb = np.zeros((len(df_train), len(targets)))
    oof_preds_xgb = np.zeros((len(df_train), len(targets)))
    oof_preds_et = np.zeros((len(df_train), len(targets)))
    oof_preds_rf = np.zeros((len(df_train), len(targets)))
    oof_preds_lr = np.zeros((len(df_train), len(targets)))

    test_preds_cat = np.zeros((len(df_test), len(targets)))
    test_preds_lgb = np.zeros((len(df_test), len(targets)))
    test_preds_xgb = np.zeros((len(df_test), len(targets)))
    test_preds_et = np.zeros((len(df_test), len(targets)))
    test_preds_rf = np.zeros((len(df_test), len(targets)))
    test_preds_lr = np.zeros((len(df_test), len(targets)))

    chain_order = ["S3", "S2", "S1", "S4", "Q1", "Q2", "Q3"]

    logger.info("========== Starting 6-Model Sequential Classifier Chain (v13) ==========")
    for chain_step, target in enumerate(chain_order):
        t_global_idx = targets.index(target)
        logger.info(f"\n--- [Chain Step {chain_step+1}/7] Target: {target} ---")

        # Get optimal Laplace hyperparameters
        opt_params = BEST_LAPLACE_PARAMS.get(target, {'half_life': 3.0, 'prior_weight': 0.2, 'm_smooth': 10.0})
        h_life = opt_params['half_life']
        p_weight = opt_params['prior_weight']
        m_smooth_val = opt_params['m_smooth']

        # 1. Collect previous chain predictions
        prev_targets = chain_order[:chain_step]
        chain_feature_cols = []
        for prev_t in prev_targets:
            prev_global_idx = targets.index(prev_t)

            blend_prev_oof = (1/6.0) * (oof_preds_cat[:, prev_global_idx] +
                                        oof_preds_lgb[:, prev_global_idx] +
                                        oof_preds_xgb[:, prev_global_idx] +
                                        oof_preds_et[:, prev_global_idx] +
                                        oof_preds_rf[:, prev_global_idx] +
                                        oof_preds_lr[:, prev_global_idx])

            blend_prev_test = (1/6.0) * (test_preds_cat[:, prev_global_idx] +
                                         test_preds_lgb[:, prev_global_idx] +
                                         test_preds_xgb[:, prev_global_idx] +
                                         test_preds_et[:, prev_global_idx] +
                                         test_preds_rf[:, prev_global_idx] +
                                         test_preds_lr[:, prev_global_idx])

            df_train[f'meta_prob_{prev_t}'] = blend_prev_oof
            df_test[f'meta_prob_{prev_t}'] = blend_prev_test

            eps = 1e-6
            df_train[f'meta_logit_{prev_t}'] = np.log(df_train[f'meta_prob_{prev_t}'].clip(eps, 1-eps) / (1 - df_train[f'meta_prob_{prev_t}'].clip(eps, 1-eps)))
            df_test[f'meta_logit_{prev_t}'] = np.log(df_test[f'meta_prob_{prev_t}'].clip(eps, 1-eps) / (1 - df_test[f'meta_prob_{prev_t}'].clip(eps, 1-eps)))

            chain_feature_cols.extend([f'meta_prob_{prev_t}', f'meta_logit_{prev_t}'])

        y = df_train[target]
        has_label = y.notna()
        valid_indices = np.where(has_label)[0]
        valid_idx_set = set(valid_indices)

        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            tr_idx = np.array([i for i in train_idx if i in valid_idx_set])
            va_idx = np.array([i for i in val_idx if i in valid_idx_set])

            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue

            df_train_fold = df_train.copy()
            df_test_fold = df_test.copy()
            y_train_fold = y.copy()

            # Chronological target interpolation (using best opt parameters)
            temp_features = add_temporal_target_features(df_train_fold, y_train_fold, tr_idx, target, id_col, half_life=h_life, prior_weight=p_weight)
            temp_test = add_temporal_target_features(
                pd.concat([df_train_fold, df_test_fold], ignore_index=True),
                pd.concat([y_train_fold, pd.Series(index=df_test_fold.index)], ignore_index=True),
                np.arange(len(df_train_fold)),
                target,
                id_col,
                half_life=h_life,
                prior_weight=p_weight
            )
            temp_features_test = temp_test.iloc[len(df_train_fold):].reset_index(drop=True)

            df_train_fold = pd.concat([df_train_fold, temp_features], axis=1)
            df_test_fold = pd.concat([df_test_fold, temp_features_test], axis=1)

            # Subject z-score scaling (Strictly Fold-wise)
            z_scaler = LeakFreeSubjectZScaler(id_col=id_col, numeric_cols=active_base_features + roll_cols)
            z_scaler.fit(df_train_fold.iloc[tr_idx])
            df_train_fold = z_scaler.transform(df_train_fold)
            df_test_fold = z_scaler.transform(df_test_fold)

            # Target Encoding (using best opt Laplace smooth param)
            t_encoder = LeakFreeSubjectTargetEncoder(id_col=id_col, m_smooth=m_smooth_val)
            t_encoder.fit(df_train_fold.iloc[tr_idx], y_train_fold.iloc[tr_idx])
            df_train_fold = t_encoder.transform(df_train_fold)
            df_test_fold = t_encoder.transform(df_test_fold)

            # Robust pairwise interaction features
            df_train_fold, df_test_fold, special_cols = apply_robust_directives(
                df_train_fold, df_test_fold, tr_idx, target, id_col, active_base_features + roll_cols, y_train_fold
            )

            # PCA for physiological targets (S2-S4)
            pca_cols = []
            if target in ["S2", "S3", "S4"]:
                subject_zscore_names = [f"{col}_subject_zscore" for col in active_base_features + roll_cols]
                pca_transformer = FoldWisePCA(n_components=5, numeric_cols=subject_zscore_names)
                pca_transformer.fit(df_train_fold.iloc[tr_idx])
                df_train_fold = pca_transformer.transform(df_train_fold)
                df_test_fold = pca_transformer.transform(df_test_fold)
                pca_cols = [f"pca_comp_{idx}" for idx in range(5)]

            # Pearson correlation filtering strictly on training fold!
            zscore_features = [f"{col}_subject_zscore" for col in active_base_features + roll_cols]
            candidate_cols = zscore_features + special_cols

            corrs = {}
            for col in candidate_cols:
                c = df_train_fold.iloc[tr_idx][col].corr(y_train_fold.iloc[tr_idx])
                if pd.notna(c):
                    corrs[col] = abs(c)

            # Highly rigorous cap (top 45 features) to prevent GBDT overfitting on N=360 samples!
            sorted_candidates = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:45]
            selected_candidates = [col for col, _ in sorted_candidates]

            temporal_cols = [f'{target}_prev', f'{target}_next', f'{target}_interp', f'{target}_log_delta_prev', f'{target}_log_delta_next']

            # Combine selected candidates and meta-features
            fold_features = selected_candidates + base_feature_cols + chain_feature_cols + ["subject_target_encoded"] + pca_cols + temporal_cols

            X_tr = df_train_fold.iloc[tr_idx][fold_features].fillna(0.5)
            y_tr = y_train_fold.iloc[tr_idx]
            X_va = df_train_fold.iloc[va_idx][fold_features].fillna(0.5)
            X_te = df_test_fold[fold_features].fillna(0.5)

            # Model 1. CatBoost
            cat_model = CatBoostClassifier(iterations=250, depth=3, learning_rate=0.02, l2_leaf_reg=20.0, random_state=42, verbose=0, thread_count=2)
            cat_model.fit(X_tr, y_tr)
            oof_preds_cat[va_idx, t_global_idx] = cat_model.predict_proba(X_va)[:, 1]
            test_preds_cat[:, t_global_idx] += cat_model.predict_proba(X_te)[:, 1] / n_splits

            # Model 2. LightGBM
            lgb_model = lgb.LGBMClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, min_child_samples=10, reg_alpha=5.0, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=2)
            lgb_model.fit(X_tr, y_tr)
            oof_preds_lgb[va_idx, t_global_idx] = lgb_model.predict_proba(X_va)[:, 1]
            test_preds_lgb[:, t_global_idx] += lgb_model.predict_proba(X_te)[:, 1] / n_splits

            # Model 3. XGBoost
            xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.02, reg_alpha=5.0, reg_lambda=5.0, random_state=42, eval_metric="logloss", n_jobs=2)
            xgb_model.fit(X_tr, y_tr)
            oof_preds_xgb[va_idx, t_global_idx] = xgb_model.predict_proba(X_va)[:, 1]
            test_preds_xgb[:, t_global_idx] += xgb_model.predict_proba(X_te)[:, 1] / n_splits

            # Model 4. ExtraTrees
            et_model = ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2)
            et_model.fit(X_tr, y_tr)
            oof_preds_et[va_idx, t_global_idx] = et_model.predict_proba(X_va)[:, 1]
            test_preds_et[:, t_global_idx] += et_model.predict_proba(X_te)[:, 1] / n_splits

            # Model 5. RandomForest
            rf_model = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2)
            rf_model.fit(X_tr, y_tr)
            oof_preds_rf[va_idx, t_global_idx] = rf_model.predict_proba(X_va)[:, 1]
            test_preds_rf[:, t_global_idx] += rf_model.predict_proba(X_te)[:, 1] / n_splits

            # Model 6. LogisticRegression
            lr_model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
            lr_model.fit(X_tr, y_tr)
            oof_preds_lr[va_idx, t_global_idx] = lr_model.predict_proba(X_va)[:, 1]
            test_preds_lr[:, t_global_idx] += lr_model.predict_proba(X_te)[:, 1] / n_splits

        logger.info(f"Target {target} CV -> CatBoost: {log_loss(y[has_label], oof_preds_cat[has_label, t_global_idx]):.4f} | LightGBM: {log_loss(y[has_label], oof_preds_lgb[has_label, t_global_idx]):.4f} | LogisticRegression: {log_loss(y[has_label], oof_preds_lr[has_label, t_global_idx]):.4f} | ExtraTrees: {log_loss(y[has_label], oof_preds_et[has_label, t_global_idx]):.4f}")

    # Save Curated OOFs and Submissions as v13 files
    logger.info("Saving SOTA-v31 curated OOF predictions...")

    oof_template = pd.DataFrame({'date': df_train['date'], 'subject_id': df_train[id_col]})
    sub_template = pd.DataFrame({'date': df_test['date'], 'subject_id': df_test[id_col]})

    cat_oof = oof_template.copy(); cat_sub = sub_template.copy(); cat_oof[targets] = oof_preds_cat; cat_sub[targets] = test_preds_cat
    cat_oof.to_csv("submissions/oof_catboost_v12.csv", index=False); cat_sub.to_csv("submissions/submission_catboost_v12.csv", index=False)

    lgb_oof = oof_template.copy(); lgb_sub = sub_template.copy(); lgb_oof[targets] = oof_preds_lgb; lgb_sub[targets] = test_preds_lgb
    lgb_oof.to_csv("submissions/oof_lgbm_v12.csv", index=False); lgb_sub.to_csv("submissions/submission_lgbm_v12.csv", index=False)

    xgb_oof = oof_template.copy(); xgb_sub = sub_template.copy(); xgb_oof[targets] = oof_preds_xgb; xgb_sub[targets] = test_preds_xgb
    xgb_oof.to_csv("submissions/oof_xgb_v12.csv", index=False); xgb_sub.to_csv("submissions/submission_xgb_v12.csv", index=False)

    et_oof = oof_template.copy(); et_sub = sub_template.copy(); et_oof[targets] = oof_preds_et; et_sub[targets] = test_preds_et
    et_oof.to_csv("submissions/oof_et_v12.csv", index=False); et_sub.to_csv("submissions/submission_et_v12.csv", index=False)

    rf_oof = oof_template.copy(); rf_sub = sub_template.copy(); rf_oof[targets] = oof_preds_rf; rf_sub[targets] = test_preds_rf
    rf_oof.to_csv("submissions/oof_rf_v12.csv", index=False); rf_sub.to_csv("submissions/submission_rf_v12.csv", index=False)

    lr_oof = oof_template.copy(); lr_sub = sub_template.copy(); lr_oof[targets] = oof_preds_lr; lr_sub[targets] = test_preds_lr
    lr_oof.to_csv("submissions/oof_lr_v12.csv", index=False); lr_sub.to_csv("submissions/submission_lr_v12.csv", index=False)

    logger.info("SOTA-v31 curated pipeline successfully finished training and saved as v12!")

if __name__ == "__main__":
    run_clean_ultimate_pipeline_v31()
