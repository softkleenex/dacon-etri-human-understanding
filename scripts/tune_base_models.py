import os
import sys
import warnings
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss
from sklearn.feature_selection import mutual_info_classif
import optuna

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path("/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding")
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DataLoader
from src.features.transformers import FoldWisePCA, LeakFreeSubjectZScaler
from src.utils.logger import get_logger
from src.utils.validation import make_subject_hole_folds

logger = get_logger("BaseModelTuner")

# Hardcoded feature lists from train_ultimate_pipeline_clean_v32.py
CORE_FEATURES = [
    'wHr_deep_sleep_ratio', 'wHr_sleep_efficiency_proxy',
    'wHr_sleep_duration_deep', 'wHr_sleep_duration_light', 'wHr_sleep_duration_awake',
    'wHr_sleep_fragmentation_index', 'wHr_sleep_hr_mean', 'wHr_sleep_hr_std',
    'wHr_slp_hr_deep_ratio', 'wHr_slp_hr_awake_ratio', 'wHr_slp_hr_light_ratio',
    'wHr_slp_hr_rmssd', 'wHr_slp_hr_spike_count',
    'mGps_homestay_ratio', 'mGps_radius_of_gyration', 'mGps_spatial_entropy', 'mGps_unique_locations',
    'mUsage_app_entropy', 'mUsage_dopamine_ratio', 'mUsage_hourly_volatility',
    'mUsage_night_time_ratio', 'mUsage_sessions_count', 'mUsage_total_time_mins',
    'act_active_ratio', 'act_still_ratio', 'pedo_total_steps', 'pedo_total_calories',
    'pedo_mean_speed', 'pedo_run_ratio', 'pedo_active_hours',
    'screen_on_ratio', 'screen_unlock_cnt', 'screen_night_ratio', 'screen_presleep_ratio',
    'mlight_all_mean', 'wlight_all_mean', 'mlight_dark_ratio', 'wlight_dark_ratio',
    'ac_charging_ratio', 'ac_night_charging',
    'dayofweek', 'is_weekend', 'is_holiday', 'is_rest_day'
]

CIRCADIAN_SEASONAL_FEATURES = [
    'circadian_hr_rmssd_ratio', 'circadian_light_ratio', 'circadian_screen_night_ratio',
    'circadian_act_still_ratio', 'sin_dayofyear', 'cos_dayofyear',
    'sin_month', 'cos_month', 'sin_dayofweek', 'cos_dayofweek'
]

WINDOW_PAIR_FEATURES = [
    'wHr_sleep_to_day_mean_ratio', 'wHr_sleep_to_all_mean_ratio', 'wHr_night_to_day_mean_ratio',
    'wHr_night_to_all_mean_ratio', 'wHr_sleep_to_night_mean_ratio', 'wlight_night_to_all_mean_ratio',
    'wlight_eve_to_all_mean_ratio', 'wlight_morn_to_all_mean_ratio', 'mlight_night_to_all_mean_ratio',
    'mlight_eve_to_all_mean_ratio', 'mlight_morn_to_all_mean_ratio'
]

ALL_BASE_FEATURES = CORE_FEATURES + CIRCADIAN_SEASONAL_FEATURES + WINDOW_PAIR_FEATURES

# Best Laplace Hyperparameters discovered by evolver
BEST_LAPLACE_PARAMS = {
    'Q1': {'half_life': 1.85441978, 'prior_weight': 0.58061349, 'm_smooth': 34.67560606},
    'Q2': {'half_life': 6.53547913, 'prior_weight': 0.36399036, 'm_smooth': 19.62722876},
    'Q3': {'half_life': 1.01322360, 'prior_weight': 0.36254896, 'm_smooth': 4.94146988},
    'S1': {'half_life': 3.00000000, 'prior_weight': 0.28334039, 'm_smooth': 12.93186334},
    'S2': {'half_life': 3.00000000, 'prior_weight': 0.22395933, 'm_smooth': 10.00000000},
    'S3': {'half_life': 3.00000000, 'prior_weight': 0.21492520, 'm_smooth': 8.21466671},
    'S4': {'half_life': 3.31398694, 'prior_weight': 0.20914030, 'm_smooth': 14.75008804}
}

TARGET_SPECIFIC_FILTER_PARAMS = {
    'Q1': {'top_k': 55, 'collinear_threshold': 0.85},
    'Q2': {'top_k': 50, 'collinear_threshold': 0.85},
    'Q3': {'top_k': 45, 'collinear_threshold': 0.83},
    'S1': {'top_k': 60, 'collinear_threshold': 0.87},
    'S2': {'top_k': 65, 'collinear_threshold': 0.87},
    'S3': {'top_k': 60, 'collinear_threshold': 0.87},
    'S4': {'top_k': 65, 'collinear_threshold': 0.87}
}

class LeakFreeSubjectTargetEncoder:
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

    if 'hr_dip_ratio' not in df_copy.columns:
        if 'ch2025_wHr_hr_night_mean' in df_copy.columns and 'ch2025_wHr_hr_day_mean' in df_copy.columns:
            df_copy['hr_dip_ratio'] = df_copy['ch2025_wHr_hr_night_mean'] / (df_copy['ch2025_wHr_hr_day_mean'] + 1.0)
        else:
            df_copy['hr_dip_ratio'] = 0.8
    df_copy['hr_dip_ratio'] = df_copy['hr_dip_ratio'].fillna(0.8)

    df_copy['circadian_hr_rmssd_ratio'] = df_copy['wHr_slp_hr_rmssd'] / (df_copy['wHr_sleep_hr_mean'] + 1.0)
    df_copy['circadian_light_ratio'] = df_copy['wlight_dark_ratio'] / (df_copy['wlight_all_mean'] + 1e-5)
    df_copy['circadian_screen_night_ratio'] = df_copy['screen_night_ratio'] / (df_copy['screen_on_ratio'] + 1e-5)
    df_copy['circadian_act_still_ratio'] = df_copy['act_night_still'] / (df_copy['act_still_ratio'] + 1e-5)

    df_copy['wHr_sleep_to_day_mean_ratio'] = df_copy['wHr_sleep_hr_mean'] / (df_copy['ch2025_wHr_hr_day_mean'] + 1e-5)
    df_copy['wHr_sleep_to_all_mean_ratio'] = df_copy['wHr_sleep_hr_mean'] / (df_copy['ch2025_wHr_hr_mean_mean'] + 1e-5)
    df_copy['wHr_night_to_day_mean_ratio'] = df_copy['ch2025_wHr_hr_night_mean'] / (df_copy['ch2025_wHr_hr_day_mean'] + 1e-5)
    df_copy['wHr_night_to_all_mean_ratio'] = df_copy['ch2025_wHr_hr_night_mean'] / (df_copy['ch2025_wHr_hr_mean_mean'] + 1e-5)
    df_copy['wHr_sleep_to_night_mean_ratio'] = df_copy['wHr_sleep_hr_mean'] / (df_copy['ch2025_wHr_hr_night_mean'] + 1e-5)
    
    df_copy['wlight_night_to_all_mean_ratio'] = df_copy['wlight_night_mean'] / (df_copy['wlight_all_mean'] + 1e-5)
    df_copy['wlight_eve_to_all_mean_ratio'] = df_copy['wlight_eve_mean'] / (df_copy['wlight_all_mean'] + 1e-5)
    df_copy['wlight_morn_to_all_mean_ratio'] = df_copy['wlight_morn_mean'] / (df_copy['wlight_all_mean'] + 1e-5)
    
    df_copy['mlight_night_to_all_mean_ratio'] = df_copy['mlight_night_mean'] / (df_copy['mlight_all_mean'] + 1e-5)
    df_copy['mlight_eve_to_all_mean_ratio'] = df_copy['mlight_eve_mean'] / (df_copy['mlight_all_mean'] + 1e-5)
    df_copy['mlight_morn_to_all_mean_ratio'] = df_copy['mlight_morn_mean'] / (df_copy['mlight_all_mean'] + 1e-5)

    for col in ALL_BASE_FEATURES:
        if col in df_copy.columns:
            df_copy[col] = df_copy.groupby(id_col)[col].transform(lambda x: x.fillna(x.median()))
            df_copy[col] = df_copy[col].fillna(df_copy[col].median()).fillna(0.0)

    df_copy['date_parsed'] = pd.to_datetime(df_copy['date'])
    df_copy['dayofyear'] = df_copy['date_parsed'].dt.dayofyear
    df_copy['month'] = df_copy['date_parsed'].dt.month

    df_copy['sin_dayofyear'] = np.sin(2 * np.pi * df_copy['dayofyear'] / 365.25)
    df_copy['cos_dayofyear'] = np.cos(2 * np.pi * df_copy['dayofyear'] / 365.25)
    df_copy['sin_month'] = np.sin(2 * np.pi * df_copy['month'] / 12.0)
    df_copy['cos_month'] = np.cos(2 * np.pi * df_copy['month'] / 12.0)
    df_copy['sin_dayofweek'] = np.sin(2 * np.pi * df_copy['dayofweek'] / 7.0)
    df_copy['cos_dayofweek'] = np.cos(2 * np.pi * df_copy['dayofweek'] / 7.0)

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
        'circadian_hr_rmssd_ratio', 'circadian_light_ratio',
        'wHr_sleep_to_day_mean_ratio', 'wHr_night_to_day_mean_ratio'
    ]
    key_cols = [c for c in key_cols if c in df_copy.columns]

    roll_list = []
    for user, group in df_copy.groupby(id_col):
        group_ts = group.set_index(date_col).sort_index()
        group_ts_daily = group_ts[key_cols].resample('D').mean()

        user_dfs = []
        for w in [3, 7]:
            roll_mean = group_ts_daily.shift(1).rolling(window=w, min_periods=1).mean()
            roll_mean.columns = [f"{c}_roll{w}_mean" for c in key_cols]
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
        df_copy, all_roll,
        left_on=[id_col, 'date_only'], right_on=[id_col, date_col],
        how='left', suffixes=('', '_roll_temp')
    )
    df_copy = df_copy.drop(columns=['date_only', date_col + '_roll_temp'], errors='ignore')
    return df_copy

def add_temporal_target_features(df, y, train_idx, target_col, id_col, half_life=3.0, prior_weight=0.2):
    df_copy = df.copy()
    df_copy['original_index'] = df_copy.index

    y_train = pd.Series(np.nan, index=df_copy.index)
    y_train.loc[train_idx] = y.loc[train_idx]
    df_copy['y_train_temp'] = y_train

    df_copy['date_temp'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values(by=[id_col, 'date_temp']).reset_index(drop=True)

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
                name_add = f"{c0}_plus_{c1}"
                df_train_fold[name_add] = df_train_fold[c0] + df_train_fold[c1]
                df_test_fold[name_add] = df_test_fold[c0] + df_test_fold[c1]
                special_cols.append(name_add)

                name_sub = f"{c0}_minus_{c1}"
                df_train_fold[name_sub] = df_train_fold[c0] - df_train_fold[c1]
                df_test_fold[name_sub] = df_test_fold[c0] - df_test_fold[c1]
                special_cols.append(name_sub)

                name_mul = f"{c0}_mul_{c1}"
                df_train_fold[name_mul] = df_train_fold[c0] * df_train_fold[c1]
                df_test_fold[name_mul] = df_test_fold[c0] * df_test_fold[c1]
                special_cols.append(name_mul)
    return df_train_fold, df_test_fold, special_cols

def tune_target_base_models(target, df_train, df_test, cv_folds, base_feature_cols):
    y = df_train[target]
    has_label = y.notna()
    valid_indices = np.where(has_label)[0]
    valid_idx_set = set(valid_indices)

    # Fast cache feature pre-processing logic for Optuna loop efficiency
    opt_params = BEST_LAPLACE_PARAMS[target]
    h_life = opt_params['half_life']
    p_weight = opt_params['prior_weight']
    m_smooth_val = opt_params['m_smooth']

    id_col = "subject_id"
    active_base_features = ALL_BASE_FEATURES + ['hr_dip_ratio']
    roll_cols = [c for c in df_train.columns if '_roll' in c or '_ema' in c]

    # Pre-build fold-wise features to avoid rebuilding during Optuna trials
    fold_data = []
    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        tr_idx = np.array([i for i in train_idx if i in valid_idx_set])
        va_idx = np.array([i for i in val_idx if i in valid_idx_set])
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        df_train_fold = df_train.copy()
        df_test_fold = df_test.copy()
        y_train_fold = y.copy()

        temp_features = add_temporal_target_features(df_train_fold, y_train_fold, tr_idx, target, id_col, half_life=h_life, prior_weight=p_weight)
        df_train_fold = pd.concat([df_train_fold, temp_features], axis=1)

        z_scaler = LeakFreeSubjectZScaler(id_col=id_col, numeric_cols=active_base_features + roll_cols)
        z_scaler.fit(df_train_fold.iloc[tr_idx])
        df_train_fold = z_scaler.transform(df_train_fold)
        df_test_fold = z_scaler.transform(df_test_fold)

        t_encoder = LeakFreeSubjectTargetEncoder(id_col=id_col, m_smooth=m_smooth_val)
        t_encoder.fit(df_train_fold.iloc[tr_idx], y_train_fold.iloc[tr_idx])
        df_train_fold = t_encoder.transform(df_train_fold)
        df_test_fold = t_encoder.transform(df_test_fold)

        df_train_fold, _, special_cols = apply_robust_directives(
            df_train_fold, df_test_fold, tr_idx, target, id_col, active_base_features + roll_cols, y_train_fold
        )

        pca_cols = []
        if target in ["S2", "S3", "S4"]:
            subject_zscore_names = [f"{col}_subject_zscore" for col in active_base_features + roll_cols]
            pca_transformer = FoldWisePCA(n_components=5, numeric_cols=subject_zscore_names)
            pca_transformer.fit(df_train_fold.iloc[tr_idx])
            df_train_fold = pca_transformer.transform(df_train_fold)
            pca_cols = [f"pca_comp_{idx}" for idx in range(5)]

        zscore_features = [f"{col}_subject_zscore" for col in active_base_features + roll_cols]
        candidate_cols = zscore_features + special_cols

        corrs = {}
        for col in candidate_cols:
            c = df_train_fold.iloc[tr_idx][col].corr(y_train_fold.iloc[tr_idx])
            if pd.notna(c):
                corrs[col] = abs(c)

        filter_params = TARGET_SPECIFIC_FILTER_PARAMS[target]
        best_top_k = filter_params['top_k']
        best_collinear_thresh = filter_params['collinear_threshold']

        sorted_candidates = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:best_top_k]
        selected_candidates = [col for col, _ in sorted_candidates]

        temporal_cols = [f'{target}_prev', f'{target}_next', f'{target}_interp', f'{target}_log_delta_prev', f'{target}_log_delta_next']
        initial_features = selected_candidates + base_feature_cols + ["subject_target_encoded"] + pca_cols + temporal_cols
        initial_features = list(set(initial_features))

        X_tr_temp = df_train_fold.iloc[tr_idx][initial_features].fillna(0.5)
        y_tr_temp = y_train_fold.iloc[tr_idx]

        corr_matrix = X_tr_temp.corr(method='pearson').abs()
        mi_scores = mutual_info_classif(X_tr_temp, y_tr_temp, random_state=42)
        mi_series = pd.Series(mi_scores, index=X_tr_temp.columns)

        cols = corr_matrix.columns
        to_drop = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if corr_matrix.iloc[i, j] >= best_collinear_thresh:
                    f1, f2 = cols[i], cols[j]
                    if f1 in to_drop or f2 in to_drop:
                        continue
                    mi1 = mi_series.get(f1, 0.0)
                    mi2 = mi_series.get(f2, 0.0)
                    if mi1 < mi2:
                        to_drop.add(f1)
                    else:
                        to_drop.add(f2)

        mi_filtered_cols = [c for c in X_tr_temp.columns if c not in to_drop]
        mi_scores_pruned = mi_series[mi_filtered_cols].sort_values(ascending=False)
        num_mi_keep = max(1, int(len(mi_scores_pruned) * 0.8))
        mi_keep_cols = list(mi_scores_pruned.index[:num_mi_keep])

        X_tr_mi = X_tr_temp[mi_keep_cols]
        selector_clf = lgb.LGBMClassifier(n_estimators=35, max_depth=3, learning_rate=0.08, verbosity=-1, random_state=42)
        selector_clf.fit(X_tr_mi, y_tr_temp)
        feat_imp = pd.Series(selector_clf.feature_importances_, index=mi_keep_cols).sort_values(ascending=False)

        target_top_n_map = {'Q3': 40, 'Q1': 70, 'Q2': 70, 'S1': 90, 'S2': 90, 'S3': 90, 'S4': 90}
        top_n = target_top_n_map.get(target, 100)
        active_cols = sorted(list(feat_imp.index[:top_n]))

        X_tr = df_train_fold.iloc[tr_idx][active_cols].fillna(0.5).values
        y_tr = y_train_fold.iloc[tr_idx].values
        X_va = df_train_fold.iloc[va_idx][active_cols].fillna(0.5).values
        y_va = y_train_fold.iloc[va_idx].values

        fold_data.append((X_tr, y_tr, X_va, y_va))

    logger.info(f"Target {target} pre-processing complete. Initiating Optuna hyperparameter sweeps...")

    # Optuna study for LightGBM, XGBoost, CatBoost
    def objective(trial):
        model_name = trial.suggest_categorical("model_type", ["lgbm", "xgb", "catboost"])
        losses = []

        if model_name == "lgbm":
            max_depth = trial.suggest_int("lgb_max_depth", 3, 7)
            num_leaves = trial.suggest_int("lgb_num_leaves", 7, 63)
            learning_rate = trial.suggest_float("lgb_learning_rate", 0.01, 0.1, log=True)
            reg_alpha = trial.suggest_float("lgb_reg_alpha", 0.1, 20.0, log=True)
            reg_lambda = trial.suggest_float("lgb_reg_lambda", 0.1, 20.0, log=True)
            min_child_samples = trial.suggest_int("lgb_min_child_samples", 5, 40)
            n_estimators = trial.suggest_int("lgb_n_estimators", 50, 300)

            for X_tr, y_tr, X_va, y_va in fold_data:
                clf = lgb.LGBMClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves,
                    learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    min_child_samples=min_child_samples, random_state=42, verbosity=-1, n_jobs=2
                )
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_va)[:, 1]
                losses.append(log_loss(y_va, np.clip(p, 1e-15, 1-1e-15)))

        elif model_name == "xgb":
            max_depth = trial.suggest_int("xgb_max_depth", 3, 7)
            learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.1, log=True)
            reg_alpha = trial.suggest_float("xgb_reg_alpha", 0.1, 20.0, log=True)
            reg_lambda = trial.suggest_float("xgb_reg_lambda", 0.1, 20.0, log=True)
            subsample = trial.suggest_float("xgb_subsample", 0.6, 0.95)
            colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.6, 0.95)
            n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)

            for X_tr, y_tr, X_va, y_va in fold_data:
                clf = xgb.XGBClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                    reg_alpha=reg_alpha, reg_lambda=reg_lambda, subsample=subsample,
                    colsample_bytree=colsample_bytree, random_state=42, eval_metric="logloss", n_jobs=2
                )
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_va)[:, 1]
                losses.append(log_loss(y_va, np.clip(p, 1e-15, 1-1e-15)))

        elif model_name == "catboost":
            depth = trial.suggest_int("cat_depth", 3, 7)
            learning_rate = trial.suggest_float("cat_learning_rate", 0.01, 0.1, log=True)
            l2_leaf_reg = trial.suggest_float("cat_l2_leaf_reg", 1.0, 30.0, log=True)
            iterations = trial.suggest_int("cat_iterations", 50, 300)

            for X_tr, y_tr, X_va, y_va in fold_data:
                clf = CatBoostClassifier(
                    iterations=iterations, depth=depth, learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg, random_state=42, verbose=0, thread_count=2
                )
                clf.fit(X_tr, y_tr)
                p = clf.predict_proba(X_va)[:, 1]
                losses.append(log_loss(y_va, np.clip(p, 1e-15, 1-1e-15)))

        return np.mean(losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=35)  # 35 trials per model-target combination to optimize fast
    logger.info(f"Target {target} Best Trial Value: {study.best_value:.6f}")
    return study.best_params, study.best_value

def main():
    logger.info("Starting Hyperparameter Tuning Study for GBDT models (Target-specific)...")
    
    # Load raw data and build folds
    data_loader = DataLoader(data_dir="data")
    metrics_df = data_loader.load_metrics()
    
    df_train_raw = pd.read_parquet("data/processed/df_train_raw.parquet")
    df_test_raw = pd.read_parquet("data/processed/df_test_raw.parquet")
    df_train_raw['is_test'] = 0
    df_test_raw['is_test'] = 1
    df_all_raw = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    
    df_all = add_clean_base_features(df_all_raw)
    df_all = add_clean_rolling_features(df_all, id_col='subject_id', date_col='date')
    
    df_train = df_all[df_all['is_test'] == 0].drop(columns=['is_test']).copy()
    df_test = df_all[df_all['is_test'] == 1].drop(columns=['is_test']).copy()
    
    id_col = "subject_id"
    df_train = df_train.sort_values(by=[id_col, 'date']).reset_index(drop=True)
    df_test = df_test.sort_values(by=[id_col, 'date']).reset_index(drop=True)
    
    cv_folds = make_subject_hole_folds(df_train, n_folds=5)
    
    # Anchor check
    lstm_oof_path = 'submissions/oof_dl_v10.csv'
    base_feature_cols = []
    df_train['date'] = pd.to_datetime(df_train['date']).dt.date
    df_test['date'] = pd.to_datetime(df_test['date']).dt.date
    
    if os.path.exists(lstm_oof_path):
        base_oof = pd.read_csv(lstm_oof_path)
        base_oof['date'] = pd.to_datetime(base_oof['date']).dt.date
        df_train = pd.merge(df_train, base_oof, on=[id_col, 'date'], how='left', suffixes=('', '_base'))
        
        eps = 1e-6
        for t in ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]:
            base_col = f"{t}_base"
            if base_col in df_train.columns:
                prob = df_train[base_col].fillna(0.5).clip(eps, 1-eps)
                df_train[f'base_{t}_prob'] = prob
                df_train[f'base_{t}_logit'] = np.log(prob / (1 - prob))
            base_feature_cols.extend([f'base_{t}_prob', f'base_{t}_logit'])
            
    # Target parameter dictionary mapping
    tuning_results = {}
    
    targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
    for target in targets:
        best_params, best_loss = tune_target_base_models(target, df_train, df_test, cv_folds, base_feature_cols)
        tuning_results[target] = {
            'best_params': best_params,
            'best_loss': best_loss
        }
        
    # Save the optimal parameters as a json directive
    directive_path = "checks/base_model_opt_hyperparams.json"
    with open(directive_path, 'w') as f:
        json.dump(tuning_results, f, indent=4)
        
    logger.info(f"Target-Specific GBDT model tuning complete. Directive saved to: {directive_path}")

if __name__ == "__main__":
    main()
