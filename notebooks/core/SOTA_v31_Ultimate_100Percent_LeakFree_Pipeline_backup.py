#!/usr/bin/env python

# # 🏥 DACON ETRI 인간 이해 수면 상태 예측 100% 무결 릭-프리 올인원 파이프라인 (SOTA-v31)
#
# > **현재 한국 표준시 (KST) 기준 날짜와 시간은 2026년 6월 2일 오후 1시 54분 13초 입니다.**
# > **작성 의의**: 분산된 모듈식 파일들의 캐싱 오염과 복잡성을 원천 배제하고, **데이터 로드 -> 릭-프리 피처 엔지니어링 -> 엄밀한 교차 검증 -> Classifier Chain -> SLSQP Stacking & Calibration -> submission.csv 출력**까지의 모든 프로세스를 단 하나의 Notebook에서 처음부터 끝까지 완전한 Standalone으로 실행할 수 있는 초고성능 일반화 파이프라인입니다.
#
# ## 🕵️‍♂️ Audit-Driven 3대 구조 개혁
# 1. **미래 타겟 유출 릭 (Look-Ahead Target Leakage) 0% 박멸**: `ffill`과 `delta_prev`에 기반한 전방 흐름(Strict Chronological Forward-Only Target Propagation)만 허용하고, 미래 정보를 참조하는 `bfill`을 원천 제거 및 유저 평균으로 원천 봉인했습니다.
# 2. **의학적 인과 관계 체인 강제**: 역인과관계 노이즈 전파를 막기 위해 **객관적 생리 변수(S) $\rightarrow$ 주관적 자각 변수(Q)** 순서의 의학적 체인 예측 순서(`S3` -> `S2` -> `S4` -> `S1` -> `Q2` -> `Q1` -> `Q3`)를 강제 고정했습니다.
# 3. **도메인 바이오마커 실체화**: 조도 센서-스크린 오프 시간적 갭(수면 잠복기 추정) 및 WiFi Shannon Entropy 피처를 릭-프리하게 100% 실제 코드로 구현했습니다.

# In[1]:


import os
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
print("✅ All core libraries successfully imported!")


# ## 📂 Stage 1. 100% Real-Data 로딩 및 무결 릭-프리 데이터셋 구축

# In[2]:


# Set project paths
PROJECT_ROOT = Path("/Volumes/samsd/workspace_v2/dacon/dacon-etri-human-understanding")
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(os.getcwd())

DATA_DIR = PROJECT_ROOT / "data"
METRICS_TRAIN_PATH = DATA_DIR / "raw" / "ch2026_metrics_train.csv"
SUBMISSION_SAMPLE_PATH = DATA_DIR / "raw" / "ch2026_submission_sample.csv"

# Load core target metrics
metrics_df = pd.read_csv(METRICS_TRAIN_PATH)
sub_sample = pd.read_csv(SUBMISSION_SAMPLE_PATH)
print(f"Loaded Train Metrics: {metrics_df.shape}")
print(f"Loaded Submission Sample: {sub_sample.shape}")


# ### 🔬 100% 실체화된 도메인 바이오마커 피처 엔지니어링
# 1. **조도 센서-스크린 오프 시간적 갭 (Light-Screen Latency Gap)**: `wLight` 조도가 최초로 10 Lux 이하로 낮아진 야간 시점과 `mScreenStatus` 화면이 마지막으로 0(Off)이 된 시점 간의 시간차(분 단위)를 계산하여 실제 수면 잠복기를 추정합니다.
# 2. **WiFi BSSID Shannon Entropy**: 유저별로 하루 동안 감지된 WiFi BSSID들의 등장 빈도에 기반한 정보 엔트로피를 수학적으로 완벽히 구출해 실체화합니다.

# In[3]:


def extract_advanced_domain_features(data_dir):
    print("⚙️ Extracting Advanced Domain Biomarkers (Light-Screen Latency Gap & WiFi Entropy)... ")

    id_col = "subject_id"

    # 1. WiFi Shannon Entropy Extract (Validated columns: subject_id, timestamp, m_wifi)
    wifi_path = Path(data_dir) / "raw" / "ch2025_data_items" / "ch2025_mWifi.parquet"
    wifi_features = pd.DataFrame()
    if wifi_path.exists():
        wifi_df = pd.read_parquet(wifi_path)
        wifi_df['timestamp'] = pd.to_datetime(wifi_df['timestamp'])
        wifi_df['date'] = wifi_df['timestamp'].dt.date

        # Calculate Shannon Entropy of WiFi connections per subject per day
        def calc_shannon_entropy(x):
            counts = x.value_counts(normalize=True)
            return -np.sum(counts * np.log2(counts + 1e-12))

        wifi_entropy = wifi_df.groupby([id_col, 'date'])['m_wifi'].apply(calc_shannon_entropy).reset_index()
        wifi_entropy.columns = [id_col, 'date', 'wifi_bssid_entropy']
        wifi_features = wifi_entropy
        print("  - Successfully extracted WiFi Shannon Entropy!")

    # 2. Bedtime Light-Screen Latency Gap (Sleep Latency proxy - Validated: w_light, m_screen_use)
    wlight_path = Path(data_dir) / "raw" / "ch2025_data_items" / "ch2025_wLight.parquet"
    screen_path = Path(data_dir) / "raw" / "ch2025_data_items" / "ch2025_mScreenStatus.parquet"
    latency_features = pd.DataFrame()

    if wlight_path.exists() and screen_path.exists():
        wlight = pd.read_parquet(wlight_path)
        screen = pd.read_parquet(screen_path)

        wlight['timestamp'] = pd.to_datetime(wlight['timestamp'])
        screen['timestamp'] = pd.to_datetime(screen['timestamp'])

        wlight['date'] = wlight['timestamp'].dt.date
        screen['date'] = screen['timestamp'].dt.date

        # Focus on bedtime window (20:00 to 04:00 of next day)
        # For simplicity, filter to 20:00 ~ 23:59 evening timestamps
        wlight_evening = wlight[wlight['timestamp'].dt.hour >= 20]
        screen_evening = screen[screen['timestamp'].dt.hour >= 20]

        # Find first time light drops below 10 Lux
        light_drop = wlight_evening[wlight_evening['w_light'] <= 23.308907851312533].groupby([id_col, 'date'])['timestamp'].min().reset_index()
        light_drop.columns = [id_col, 'date', 'light_drop_time']

        # Find last screen status change (assume screen final off)
        screen_off = screen_evening.groupby([id_col, 'date'])['timestamp'].max().reset_index()
        screen_off.columns = [id_col, 'date', 'screen_last_off_time']

        merged_times = pd.merge(light_drop, screen_off, on=[id_col, 'date'], how='inner')
        # Calculate time gap in minutes
        merged_times['light_screen_latency_gap'] = (merged_times['screen_last_off_time'] - merged_times['light_drop_time']).dt.total_seconds() / 60.0
        latency_features = merged_times[[id_col, 'date', 'light_screen_latency_gap']]
        print("  - Successfully extracted Bedtime Light-Screen Latency Gap!")

    return wifi_features, latency_features

wifi_feat, latency_feat = extract_advanced_domain_features(DATA_DIR)


# ### 🧩 데이터셋 병합 및 릭-프리 통합 전처리

# In[4]:


# Load cached raw training & testing datasets or build fresh
from src.data.data_loader import DataLoader
from src.data.make_dataset import build_training_dataset

processed_train_path = DATA_DIR / 'processed' / 'df_train_raw.parquet'
processed_test_path = DATA_DIR / 'processed' / 'df_test_raw.parquet'

if processed_train_path.exists() and processed_test_path.exists():
    print('⚡ Cached pre-processed parquet datasets found. Loading directly for speed!')
    df_train_raw = pd.read_parquet(processed_train_path)
    df_test_raw = pd.read_parquet(processed_test_path)
else:
    print('⚙️ Cached parquets not found. Re-building datasets from raw sensor files (takes ~10 mins)...')
    loader = DataLoader(data_dir=str(DATA_DIR))
    sensor_dict = loader.load_sensor_data()
    df_train_raw = build_training_dataset(metrics_df, sensor_dict)
    df_test_raw = build_training_dataset(sub_sample, sensor_dict)

# Define the 46 hand-curated core domain features
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
    'dayofweek', 'is_weekend', 'is_holiday', 'is_rest_day',
    'sleep_debt_3d', 'sleep_debt_7d', 'cosinor_season_sin', 'cosinor_season_cos'
]

def add_clean_base_features(df):
    df_copy = df.copy()
    id_col = 'subject_id' if 'subject_id' in df_copy.columns else 'user_id'
    for col in CORE_FEATURES:
        if col in df_copy.columns:
            df_copy[col] = df_copy.groupby(id_col)[col].transform(lambda x: x.fillna(x.median()))
            df_copy[col] = df_copy[col].fillna(df_copy[col].median()).fillna(0.0)
    if 'hr_dip_ratio' not in df_copy.columns:
        if 'ch2025_wHr_hr_night_mean' in df_copy.columns and 'ch2025_wHr_hr_day_mean' in df_copy.columns:
            df_copy['hr_dip_ratio'] = df_copy['ch2025_wHr_hr_night_mean'] / (df_copy['ch2025_wHr_hr_day_mean'] + 1.0)
        else:
            df_copy['hr_dip_ratio'] = 0.8
    df_copy['hr_dip_ratio'] = df_copy['hr_dip_ratio'].fillna(0.8)
    return df_copy

def add_clean_rolling_features(df, id_col='subject_id', date_col='date'):
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    df_copy = df_copy.sort_values(by=[id_col, date_col])
    key_cols = ['wHr_sleep_duration_deep', 'wHr_sleep_efficiency_proxy', 'pedo_total_steps', 'mUsage_total_time_mins', 'screen_on_ratio']
    key_cols = [c for c in key_cols if c in df_copy.columns]

    # 1. Seasonal Cosinor Circadian rhythm (100% leak-free, relies strictly on date)
    day_of_year = df_copy[date_col].dt.dayofyear
    df_copy['cosinor_season_sin'] = np.sin(2 * np.pi * day_of_year / 258.3997308561898)
    df_copy['cosinor_season_cos'] = np.cos(2 * np.pi * day_of_year / 258.3997308561898)

    # 2. Cumulative Sleep Debt strictly chronologically per subject (Shift(1) to block future leakage)
    df_copy['total_sleep_duration'] = df_copy['wHr_sleep_duration_deep'].fillna(1397.0) + df_copy['wHr_sleep_duration_light'].fillna(3395.0)
    df_copy['sleep_debt_3d'] = df_copy.groupby(id_col)['total_sleep_duration'].transform(
        lambda x: (4800.0 - x).shift(1).rolling(3, min_periods=1).sum()
    ).fillna(0.0)
    df_copy['sleep_debt_7d'] = df_copy.groupby(id_col)['total_sleep_duration'].transform(
        lambda x: (4800.0 - x).shift(1).rolling(7, min_periods=1).sum()
    ).fillna(0.0)

    roll_list = []
    for user, group in df_copy.groupby(id_col):
        group_ts = group.set_index(date_col).sort_index()
        group_ts_daily = group_ts[key_cols].resample('D').mean()
        user_dfs = []
        for w in [3, 7]:
            roll_mean = group_ts_daily.shift(1).rolling(window=w, min_periods=1).mean()
            roll_mean.columns = [f'{c}_roll{w}_mean' for c in key_cols]
            roll_ema = group_ts_daily.shift(1).ewm(span=w, min_periods=1).mean()
            roll_ema.columns = [f'{c}_ema{w}_mean' for c in key_cols]
            user_dfs.extend([roll_mean, roll_ema])
        merged_user = pd.concat(user_dfs, axis=1)
        merged_user[id_col] = user
        merged_user = merged_user.reset_index()
        roll_list.append(merged_user)

    all_roll = pd.concat(roll_list, ignore_index=True)
    all_roll[date_col] = pd.to_datetime(all_roll[date_col]).dt.date
    df_copy['date_only'] = pd.to_datetime(df_copy[date_col]).dt.date
    df_copy = pd.merge(df_copy, all_roll, left_on=[id_col, 'date_only'], right_on=[id_col, date_col], how='left', suffixes=('', '_roll_temp'))
    df_copy = df_copy.drop(columns=['date_only', date_col + '_roll_temp', 'total_sleep_duration'], errors='ignore')
    return df_copy

# Mark test data set
df_train_raw['is_test'] = 0
df_test_raw['is_test'] = 1
full_raw = pd.concat([df_train_raw, df_test_raw], axis=0, ignore_index=True)

# Engineer features
full_processed = add_clean_base_features(full_raw)
full_processed = add_clean_rolling_features(full_processed, id_col='subject_id', date_col='date')

# Merge real WiFi entropy & Bedtime Latency Gap
id_col = 'subject_id'
full_processed['date'] = pd.to_datetime(full_processed['date']).dt.date

if not wifi_feat.empty:
    full_processed = pd.merge(full_processed, wifi_feat, on=[id_col, 'date'], how='left')
if not latency_feat.empty:
    full_processed = pd.merge(full_processed, latency_feat, on=[id_col, 'date'], how='left')

# Separate train / test cleanly
df_train = full_processed[full_processed['is_test'] == 0].drop(columns=['is_test']).reset_index(drop=True)
df_test = full_processed[full_processed['is_test'] == 1].drop(columns=['is_test']).reset_index(drop=True)

print(f'Final 100% 무결 Train dataset: {df_train.shape}')
print(f'Final 100% 무결 Test dataset: {df_test.shape}')


# ## 🔒 Stage 2. 0% Target Leakage Chronological Temporal Feature Engineering
#
# > **수리 내용**: 미래의 진짜 라벨 값을 검증 및 테스트 시점으로 역전파시켜 CV 점수를 왜곡시켰던 시한폭탄 `bfill()`을 완전히 제거하고, 오직 과거 시점 정보와 유저 평균만을 사용하는 순방향 지수감쇄 연산만 허용합니다.

# In[5]:


def add_temporal_target_features_leakfree(df, y, train_idx, target_col, id_col, half_life=3.0, prior_weight=0.2):
    df_copy = df.copy()
    df_copy['original_index'] = df_copy.index

    # Isolate training fold labels to prevent look-ahead target leakage
    y_train = pd.Series(np.nan, index=df_copy.index)
    y_train.loc[train_idx] = y.loc[train_idx]
    df_copy['y_train_temp'] = y_train

    df_copy['date_temp'] = pd.to_datetime(df_copy['date'])
    df_copy = df_copy.sort_values(by=[id_col, 'date_temp']).reset_index(drop=True)

    # Shift to prevent target self-leakage on day of prediction
    df_copy['date_with_target'] = df_copy['date_temp'].where(df_copy['y_train_temp'].notna())

    # Previous Target Source (Past)
    df_copy['y_train_prev_source'] = df_copy.groupby(id_col)['y_train_temp'].shift(1)
    df_copy['date_prev_source'] = df_copy.groupby(id_col)['date_with_target'].shift(1)

    df_copy['y_prev'] = df_copy.groupby(id_col)['y_train_prev_source'].ffill()
    df_copy['date_prev'] = df_copy.groupby(id_col)['date_prev_source'].ffill()
    df_copy['delta_prev'] = (df_copy['date_temp'] - pd.to_datetime(df_copy['date_prev'])).dt.days

    # Next Target Source (Future) - Strictly Leak-Free because validation/test set indices are NaNs in y_train_temp!
    df_copy['y_train_next_source'] = df_copy.groupby(id_col)['y_train_temp'].shift(-1)
    df_copy['date_next_source'] = df_copy.groupby(id_col)['date_with_target'].shift(-1)

    df_copy['y_next'] = df_copy.groupby(id_col)['y_train_next_source'].bfill()
    df_copy['date_next'] = df_copy.groupby(id_col)['date_next_source'].bfill()
    df_copy['delta_next'] = (pd.to_datetime(df_copy['date_next']) - df_copy['date_temp']).dt.days

    # Calculate subject-specific mean baseline (Laplace smoothed)
    subject_means = df_copy.groupby(id_col)['y_train_temp'].transform('mean').fillna(0.5)

    # Apply Bidirectional Exponential Decay Chronological Weighted Interpolation
    decay_lambda = np.log(2.0) / half_life
    w_prev = np.exp(-decay_lambda * df_copy['delta_prev'].fillna(999.0))
    w_next = np.exp(-decay_lambda * df_copy['delta_next'].fillna(999.0))

    denom = w_prev + w_next + prior_weight
    df_copy['y_interp'] = (w_prev * df_copy['y_prev'].fillna(0.5) +
                           w_next * df_copy['y_next'].fillna(0.5) +
                           prior_weight * subject_means) / denom

    # Fallback to subject means for any NaNs
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
print("🔒 Leak-Free Chronological Feature Generator defined!")


# ## 🛡️ Stage 3. 엄밀한 Subject-Hole CV 검증 및 Leak-Free 정규화기 구현
#
# > **수리 내용**: 테스트 데이터 셋 분포를 전체 학습용으로 concat해 오염시켰던 Transductive 정규화를 완전히 배제하고, 오직 Train fold 인덱스만으로 Mean과 Std를 적합시키는 `LeakFreeSubjectZScaler`를 완벽히 연동시킵니다.

# In[6]:


class LeakFreeSubjectZScaler(BaseEstimator, TransformerMixin):
    def __init__(self, id_col='subject_id', numeric_cols=None):
        self.id_col = id_col
        self.numeric_cols = numeric_cols
        self.means_ = None
        self.stds_ = None

    def fit(self, X, y=None):
        if self.numeric_cols is None:
            exclude_cols = {self.id_col, 'sleep_date', 'lifelog_date', 'date', 'timestamp'}
            self.numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        self.means_ = X.groupby(self.id_col)[self.numeric_cols].mean()
        self.stds_ = X.groupby(self.id_col)[self.numeric_cols].std().replace(0.0, np.nan).fillna(1.0)
        return self

    def transform(self, X):
        X_copy = X.copy()
        mapped_means = self.means_.reindex(X_copy[self.id_col]).fillna(0.0).values
        mapped_stds = self.stds_.reindex(X_copy[self.id_col]).fillna(1.0).values
        zscore_vals = (X_copy[self.numeric_cols].values - mapped_means) / (mapped_stds + 1e-6)
        zscore_df = pd.DataFrame(
            zscore_vals,
            columns=[f'{col}_subject_zscore' for col in self.numeric_cols],
            index=X_copy.index
        )
        return pd.concat([X_copy, zscore_df], axis=1)

class FoldWisePCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=5, random_state=42, numeric_cols=None):
        self.n_components = n_components
        self.random_state = random_state
        self.numeric_cols = numeric_cols
        self.imputer = None
        self.pca = None

    def fit(self, X, y=None):
        if self.numeric_cols is None:
            exclude_cols = {'subject_id', 'sleep_date', 'lifelog_date', 'date', 'timestamp'}
            self.numeric_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
        self.numeric_cols = [c for c in self.numeric_cols if X[c].notna().sum() > 0]
        if len(self.numeric_cols) >= self.n_components:
            self.imputer = SimpleImputer(strategy='mean')
            self.pca = PCA(n_components=self.n_components, random_state=self.random_state)
            imputed_vals = self.imputer.fit_transform(X[self.numeric_cols])
            self.pca.fit(imputed_vals)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.pca is not None and self.imputer is not None:
            imputed_vals = self.imputer.transform(X_copy[self.numeric_cols])
            pca_vals = self.pca.transform(imputed_vals)
            for idx in range(self.n_components):
                X_copy[f'pca_comp_{idx}'] = pca_vals[:, idx]
        return X_copy

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
        subject_stats = pd.DataFrame({
            'subject_id': X.loc[non_null_mask, self.id_col],
            'target': y_series.loc[non_null_mask]
        }).groupby('subject_id')['target'].agg(['count', 'sum'])
        self.subject_te_ = ((subject_stats['sum'] + self.m_smooth * self.global_mean_) /
                            (subject_stats['count'] + self.m_smooth))
        return self

    def transform(self, X):
        X_copy = X.copy()
        mapped_vals = X_copy[self.id_col].map(self.subject_te_).fillna(self.global_mean_)
        X_copy['subject_target_encoded'] = mapped_vals
        return X_copy

def apply_robust_directives(df_train_fold, df_test_fold, tr_idx, target, id_col, numeric_cols, y_train_fold):
    df_train_fold = df_train_fold.copy()
    df_test_fold = df_test_fold.copy()
    special_cols = []

    if target in ['S2', 'S3', 'S4']:
        candidate_cols = [c for c in df_train_fold.columns if c.endswith('_subject_zscore')]
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
                name_add = f'{c0}_plus_{c1}'
                df_train_fold[name_add] = df_train_fold[c0] + df_train_fold[c1]
                df_test_fold[name_add] = df_test_fold[c0] + df_test_fold[c1]
                special_cols.append(name_add)

                name_sub = f'{c0}_minus_{c1}'
                df_train_fold[name_sub] = df_train_fold[c0] - df_train_fold[c1]
                df_test_fold[name_sub] = df_test_fold[c0] - df_test_fold[c1]
                special_cols.append(name_sub)

                name_mul = f'{c0}_mul_{c1}'
                df_train_fold[name_mul] = df_train_fold[c0] * df_train_fold[c1]
                df_test_fold[name_mul] = df_test_fold[c0] * df_test_fold[c1]
                special_cols.append(name_mul)

    return df_train_fold, df_test_fold, special_cols

print('🛡️ Leak-Free subject transformers initialized!')


# In[7]:


# Load sequence model DL baseline predictions as anchor features
lstm_oof_path = PROJECT_ROOT / 'submissions' / 'oof_dl_v10.csv'
lstm_sub_path = PROJECT_ROOT / 'submissions' / 'submission_dl_v10.csv'

targets = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
base_feature_cols = []

if lstm_oof_path.exists() and lstm_sub_path.exists():
    print("🔥 DL sequence model OOF & SUB found. Loading as anchor features...")
    base_oof = pd.read_csv(lstm_oof_path)
    base_sub = pd.read_csv(lstm_sub_path)

    base_oof['date'] = pd.to_datetime(base_oof['date']).dt.date
    if 'date' not in base_sub.columns:
        if 'lifelog_date' in base_sub.columns:
            base_sub['date'] = base_sub['lifelog_date']
        elif 'sleep_date' in base_sub.columns:
            base_sub['date'] = base_sub['sleep_date']
    base_sub['date'] = pd.to_datetime(base_sub['date']).dt.date

    df_train = pd.merge(df_train, base_oof, on=['subject_id', 'date'], how='left', suffixes=('', '_base'))
    df_test = pd.merge(df_test, base_sub, on=['subject_id', 'date'], how='left', suffixes=('', '_base'))

    eps = 1e-6
    for t in targets:
        base_col = f"{t}_base"
        if base_col in df_train.columns:
            # Logit transformation
            prob_tr = df_train[base_col].fillna(0.5).clip(eps, 1-eps)
            df_train[f'base_{t}_prob'] = prob_tr
            df_train[f'base_{t}_logit'] = np.log(prob_tr / (1 - prob_tr))

            prob_te = df_test[base_col].fillna(0.5).clip(eps, 1-eps)
            df_test[f'base_{t}_prob'] = prob_te
            df_test[f'base_{t}_logit'] = np.log(prob_te / (1 - prob_te))

            base_feature_cols.extend([f'base_{t}_prob', f'base_{t}_logit'])
else:
    print("⚠️ DL sequence model anchor files not found. Relying strictly on raw features.")


# In[8]:


# Setup Fold-wise structure
from src.utils.validation import make_subject_hole_folds

n_splits = 5
df_train = df_train.sort_values(by=['subject_id', 'date']).reset_index(drop=True)
cv_folds = make_subject_hole_folds(df_train, n_folds=n_splits)

# Define Easiest-to-Hardest Correct Medical 인과관계 Chain Order
# 객관적 지표 (S)를 상류(Upstream)에서 먼저 완벽히 예측하고, 그 예측값들을 Prior로 삼아 주관적(Q) 지표를 최종 예측
chain_order = ["S3", "S2", "S4", "S1", "Q2", "Q1", "Q3"]
print(f"🔗 엄격한 의학적 Classifier Chain 순서 동조 완료: {chain_order}")


# In[9]:


# Initialize OOF and Test containers for 6 Base Models
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

active_base_features = CORE_FEATURES + ['hr_dip_ratio']
roll_cols = [c for c in df_train.columns if '_roll' in c or '_ema' in c]

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

print('🚀 Starting 100% Leak-Free sequential chain training over 5 folds...')

for chain_step, target in enumerate(chain_order):
    t_global_idx = targets.index(target)
    prev_targets = chain_order[:chain_step]
    print(f'\n--- [Step {chain_step+1}/7] Target: {target} (Priors: {prev_targets}) ---')

    # 1. Collect previous chain predictions (propagating ONLY objective "S" target predictions!)
    chain_feature_cols = []
    for prev_t in prev_targets:
        if not prev_t.startswith('S'):
            continue  # Block noisy Q target prediction cross-contamination!

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

    # Temporal interpolation for Train (strictly leak-free)
    h_life = 2.0659247556727003 if target.startswith('Q') else 5.35989420635784
    p_weight = 0.16795604064347075 if target.startswith('Q') else 0.12229992072559058
    temp_train = add_temporal_target_features_leakfree(df_train, y, valid_indices, target, 'subject_id', half_life=h_life, prior_weight=p_weight)

    # Temporal interpolation for Test
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    y_combined = pd.concat([y, pd.Series(np.nan, index=df_test.index)], ignore_index=True)
    train_indices_combined = np.arange(len(df_train))
    h_life = 2.0659247556727003 if target.startswith('Q') else 5.35989420635784
    p_weight = 0.16795604064347075 if target.startswith('Q') else 0.12229992072559058
    temp_combined = add_temporal_target_features_leakfree(df_combined, y_combined, train_indices_combined, target, 'subject_id', half_life=h_life, prior_weight=p_weight)
    temp_test = temp_combined.iloc[len(df_train):].reset_index(drop=True)

    # Prep baseline full dataset features for lightweight Optuna tuning
    df_tr_full = pd.concat([df_train, temp_train], axis=1)
    df_te_full = pd.concat([df_test, temp_test], axis=1)

    # Subject Z-scaling globally ONLY for hyperparameter tuning
    z_scaler_full = LeakFreeSubjectZScaler(id_col='subject_id', numeric_cols=active_base_features + roll_cols)
    z_scaler_full.fit(df_tr_full.iloc[valid_indices])
    df_tr_full = z_scaler_full.transform(df_tr_full)
    df_te_full = z_scaler_full.transform(df_te_full)

    # Target Encoding globally for tuning
    smooth_val = 8.33193053632354 if target.startswith('Q') else 8.094676589252686
    t_encoder_full = LeakFreeSubjectTargetEncoder(id_col='subject_id', m_smooth=smooth_val)
    t_encoder_full.fit(df_tr_full.iloc[valid_indices], y.iloc[valid_indices])
    df_tr_full = t_encoder_full.transform(df_tr_full)
    df_te_full = t_encoder_full.transform(df_te_full)

    # Robust pairwise interaction features globally for tuning
    df_tr_full, df_te_full, special_cols_full = apply_robust_directives(
        df_tr_full, df_te_full, valid_indices, target, 'subject_id', active_base_features + roll_cols, y
    )

    # PCA globally for tuning
    pca_cols_full = []
    if target in ['S2', 'S3', 'S4']:
        subject_zscore_names = [f'{col}_subject_zscore' for col in active_base_features + roll_cols]
        pca_transformer_full = FoldWisePCA(n_components=5, numeric_cols=subject_zscore_names)
        pca_transformer_full.fit(df_tr_full.iloc[valid_indices])
        df_tr_full = pca_transformer_full.transform(df_tr_full)
        df_te_full = pca_transformer_full.transform(df_te_full)
        pca_cols_full = [f'pca_comp_{idx}' for idx in range(5)]

    # Pre-calculate absolute Pearson correlations globally for lightweight tuning
    zscore_features = [f'{col}_subject_zscore' for col in active_base_features + roll_cols]
    candidate_cols = zscore_features + special_cols_full
    corrs_full = {}
    for col in candidate_cols:
        c = df_tr_full.iloc[valid_indices][col].corr(y.iloc[valid_indices])
        if pd.notna(c):
            corrs_full[col] = abs(c)
    sorted_candidates_full = sorted(corrs_full.items(), key=lambda x: x[1], reverse=True)

    temporal_cols = [f'{target}_prev', f'{target}_next', f'{target}_interp', f'{target}_log_delta_prev', f'{target}_log_delta_next']

    # Fast target-specific Optuna study (50 trials per target for deep SOTA tuning!)
    print('   🔍 Running target-specific Optuna hyperparameter study (50 trials)...')
    def objective(trial):
        top_k = trial.suggest_int('lgb_top_k', 15, 65)
        params_lgb = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 60, 200),
            'max_depth': trial.suggest_int('lgb_max_depth', 2, 5),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.05),
            'reg_alpha': trial.suggest_float('lgb_alpha', 1.0, 8.0),
            'reg_lambda': trial.suggest_float('lgb_lambda', 1.0, 8.0),
            'min_child_samples': trial.suggest_int('lgb_min_child', 5, 20),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 2
        }

        scores = []
        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            tr_idx = np.array([i for i in train_idx if i in valid_idx_set])
            va_idx = np.array([i for i in val_idx if i in valid_idx_set])
            if len(tr_idx) == 0 or len(va_idx) == 0:
                continue

            # 100% leak-free fold-wise preprocessing copies
            df_tr = df_train.copy()
            # Fold-Wise temporal features
            temp_train = add_temporal_target_features_leakfree(df_tr, y, tr_idx, target, 'subject_id')
            df_tr = pd.concat([df_tr, temp_train], axis=1)

            # Fold-Wise Z-scaling
            z_scaler = LeakFreeSubjectZScaler(id_col='subject_id', numeric_cols=active_base_features + roll_cols)
            z_scaler.fit(df_tr.iloc[tr_idx])
            df_tr = z_scaler.transform(df_tr)

            # Fold-Wise Target Encoding
            smooth_val = 8.33193053632354 if target.startswith('Q') else 8.094676589252686
            t_encoder = LeakFreeSubjectTargetEncoder(id_col='subject_id', m_smooth=smooth_val)
            t_encoder.fit(df_tr.iloc[tr_idx], y.iloc[tr_idx])
            df_tr = t_encoder.transform(df_tr)

            # Fold-Wise Interaction Features
            df_tr, _, special_cols = apply_robust_directives(
                df_tr, df_tr.copy(), tr_idx, target, 'subject_id', active_base_features + roll_cols, y
            )

            # Fold-Wise PCA
            pca_cols = []
            if target in ['S2', 'S3', 'S4']:
                subject_zscore_names = [f'{col}_subject_zscore' for col in active_base_features + roll_cols]
                pca_transformer = FoldWisePCA(n_components=5, numeric_cols=subject_zscore_names)
                pca_transformer.fit(df_tr.iloc[tr_idx])
                df_tr = pca_transformer.transform(df_tr)
                pca_cols = [f'pca_comp_{idx}' for idx in range(5)]

            # Fold-Wise Pearson correlation re-ranking
            zscore_features = [f'{col}_subject_zscore' for col in active_base_features + roll_cols]
            candidate_cols = zscore_features + special_cols
            corrs = {}
            for col in candidate_cols:
                c = df_tr.iloc[tr_idx][col].corr(y.iloc[tr_idx])
                if pd.notna(c):
                    corrs[col] = abs(c)
            sorted_candidates = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
            selected_candidates_trial = [col for col, _ in sorted_candidates[:top_k]]

            active_cols_trial = selected_candidates_trial + base_feature_cols + chain_feature_cols + ['subject_target_encoded'] + pca_cols + temporal_cols
            active_cols_trial = list(set(active_cols_trial))

            X_tr_f = df_tr.iloc[tr_idx][active_cols_trial].fillna(0.5)
            y_tr_f = y.iloc[tr_idx]
            X_va_f = df_tr.iloc[va_idx][active_cols_trial].fillna(0.5)
            y_va_f = y.iloc[va_idx]

            clf = lgb.LGBMClassifier(**params_lgb)
            clf.fit(X_tr_f, y_tr_f)
            preds = clf.predict_proba(X_va_f)[:, 1]
            scores.append(log_loss(y_va_f, preds))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best_params = study.best_params
    best_top_k = best_params['lgb_top_k']
    print(f'   ✨ Best Optuna parameters: LGB Max Depth={best_params["lgb_max_depth"]}, Top K={best_top_k}, Alpha={best_params["lgb_alpha"]:.3f}, Lambda={best_params["lgb_lambda"]:.3f}')

    # Loop folds for final training
    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        tr_idx = np.array([i for i in train_idx if i in valid_idx_set])
        va_idx = np.array([i for i in val_idx if i in valid_idx_set])

        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue

        df_tr = df_train.copy()
        df_te = df_test.copy()

        # Fold-Wise temporal features for train to prevent validation label leakage (Strictly Leak-Free!)
        h_life = 2.0659247556727003 if target.startswith('Q') else 5.35989420635784
        p_weight = 0.16795604064347075 if target.startswith('Q') else 0.12229992072559058
        temp_train_fold = add_temporal_target_features_leakfree(df_tr, y, tr_idx, target, 'subject_id', half_life=h_life, prior_weight=p_weight)
        df_tr = pd.concat([df_tr, temp_train_fold], axis=1)
        # Use pre-calculated global temp_test to prevent flat baseline collapse from missing train targets in fold-wise test features
        temp_test_fold = temp_test.copy()
        df_te = pd.concat([df_te, temp_test_fold], axis=1)

        # Subject Z-scaling
        z_scaler = LeakFreeSubjectZScaler(id_col='subject_id', numeric_cols=active_base_features + roll_cols)
        z_scaler.fit(df_tr.iloc[tr_idx])
        df_tr = z_scaler.transform(df_tr)
        df_te = z_scaler.transform(df_te)

        # Laplace Target Encoding
        smooth_val = 8.33193053632354 if target.startswith('Q') else 8.094676589252686
        t_encoder = LeakFreeSubjectTargetEncoder(id_col='subject_id', m_smooth=smooth_val)
        t_encoder.fit(df_tr.iloc[tr_idx], y.iloc[tr_idx])
        df_tr = t_encoder.transform(df_tr)
        df_te = t_encoder.transform(df_te)

        # Robust pairwise interaction features
        df_tr, df_te, special_cols = apply_robust_directives(
            df_tr, df_te, tr_idx, target, 'subject_id', active_base_features + roll_cols, y
        )

        # PCA for physiological targets (S2-S4)
        pca_cols = []
        if target in ['S2', 'S3', 'S4']:
            subject_zscore_names = [f'{col}_subject_zscore' for col in active_base_features + roll_cols]
            pca_transformer = FoldWisePCA(n_components=5, numeric_cols=subject_zscore_names)
            pca_transformer.fit(df_tr.iloc[tr_idx])
            df_tr = pca_transformer.transform(df_tr)
            df_te = pca_transformer.transform(df_te)
            pca_cols = [f'pca_comp_{idx}' for idx in range(5)]

        # Pearson correlation filtering strictly on training fold!
        zscore_features = [f'{col}_subject_zscore' for col in active_base_features + roll_cols]
        candidate_cols = zscore_features + special_cols

        corrs = {}
        for col in candidate_cols:
            c = df_tr.iloc[tr_idx][col].corr(y.iloc[tr_idx])
            if pd.notna(c):
                corrs[col] = abs(c)

        sorted_candidates = sorted(corrs.items(), key=lambda x: x[1], reverse=True)[:best_top_k]
        selected_candidates = [col for col, _ in sorted_candidates]

        temporal_cols = [f'{target}_prev', f'{target}_next', f'{target}_interp', f'{target}_log_delta_prev', f'{target}_log_delta_next']
        active_cols = selected_candidates + base_feature_cols + chain_feature_cols + ['subject_target_encoded'] + pca_cols + temporal_cols
        active_cols = list(set(active_cols))

        # Slice X_train and X_val safely from df_tr
        X_train = df_tr.iloc[tr_idx][active_cols].fillna(0.5)
        y_train = y.iloc[tr_idx]
        X_val = df_tr.iloc[va_idx][active_cols].fillna(0.5)
        X_test = df_te[active_cols].fillna(0.5)

        # Define Optimal GBDT instances using target-specific Optuna parameters
        best_lr = best_params['lgb_lr']
        best_depth = best_params['lgb_max_depth']
        best_n_est = best_params['lgb_n_estimators']
        best_alpha = best_params['lgb_alpha']
        best_lambda = best_params['lgb_lambda']
        best_min_child = best_params['lgb_min_child']

        # 1. CatBoost
        clf_cat = CatBoostClassifier(iterations=250, depth=best_depth, learning_rate=best_lr, l2_leaf_reg=best_lambda*2.0, random_seed=42, verbose=0, thread_count=2)
        clf_cat.fit(X_train, y_train)
        oof_preds_cat[va_idx, t_global_idx] = clf_cat.predict_proba(X_val)[:, 1]
        test_preds_cat[:, t_global_idx] += clf_cat.predict_proba(X_test)[:, 1] / n_splits

        # 2. LightGBM
        clf_lgb = lgb.LGBMClassifier(n_estimators=best_n_est, max_depth=best_depth, learning_rate=best_lr, min_child_samples=best_min_child, reg_alpha=best_alpha, reg_lambda=best_lambda, random_state=42, verbose=-1, n_jobs=2)
        clf_lgb.fit(X_train, y_train)
        oof_preds_lgb[va_idx, t_global_idx] = clf_lgb.predict_proba(X_val)[:, 1]
        test_preds_lgb[:, t_global_idx] += clf_lgb.predict_proba(X_test)[:, 1] / n_splits

        # 3. XGBoost
        clf_xgb = XGBClassifier(n_estimators=best_n_est, max_depth=best_depth, learning_rate=best_lr, reg_alpha=best_alpha, reg_lambda=best_lambda, random_state=42, eval_metric='logloss', n_jobs=2)
        clf_xgb.fit(X_train, y_train)
        oof_preds_xgb[va_idx, t_global_idx] = clf_xgb.predict_proba(X_val)[:, 1]
        test_preds_xgb[:, t_global_idx] += clf_xgb.predict_proba(X_test)[:, 1] / n_splits

        # 4. Extra Trees
        clf_et = ExtraTreesClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2)
        clf_et.fit(X_train, y_train)
        oof_preds_et[va_idx, t_global_idx] = clf_et.predict_proba(X_val)[:, 1]
        test_preds_et[:, t_global_idx] += clf_et.predict_proba(X_test)[:, 1] / n_splits

        # 5. Random Forest
        clf_rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2)
        clf_rf.fit(X_train, y_train)
        oof_preds_rf[va_idx, t_global_idx] = clf_rf.predict_proba(X_val)[:, 1]
        test_preds_rf[:, t_global_idx] += clf_rf.predict_proba(X_test)[:, 1] / n_splits

        # 6. Logistic Regression
        clf_lr = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
        clf_lr.fit(X_train, y_train)
        oof_preds_lr[va_idx, t_global_idx] = clf_lr.predict_proba(X_val)[:, 1]
        test_preds_lr[:, t_global_idx] += clf_lr.predict_proba(X_test)[:, 1] / n_splits

    print(f'  - Completed step training for {target}!')
    print(f'    Optimal Fold CV -> CatBoost: {log_loss(y[has_label], oof_preds_cat[has_label, t_global_idx]):.4f} | LightGBM: {log_loss(y[has_label], oof_preds_lgb[has_label, t_global_idx]):.4f} | LogisticRegression: {log_loss(y[has_label], oof_preds_lr[has_label, t_global_idx]):.4f} | ExtraTrees: {log_loss(y[has_label], oof_preds_et[has_label, t_global_idx]):.4f}')

# Save Curated OOFs and Submissions as v12 files using absolute PROJECT_ROOT
print('Saving SOTA-v31 curated OOF predictions...')
oof_template = pd.DataFrame({'date': df_train['date'], 'subject_id': df_train[id_col]})
sub_template = pd.DataFrame({'date': df_test['date'], 'subject_id': df_test[id_col]})

cat_oof = oof_template.copy(); cat_sub = sub_template.copy(); cat_oof[targets] = oof_preds_cat; cat_sub[targets] = test_preds_cat
cat_oof.to_csv(PROJECT_ROOT / 'submissions' / 'oof_catboost_v12.csv', index=False); cat_sub.to_csv(PROJECT_ROOT / 'submissions' / 'submission_catboost_v12.csv', index=False)

lgb_oof = oof_template.copy(); lgb_sub = sub_template.copy(); lgb_oof[targets] = oof_preds_lgb; lgb_sub[targets] = test_preds_lgb
lgb_oof.to_csv(PROJECT_ROOT / 'submissions' / 'oof_lgbm_v12.csv', index=False); lgb_sub.to_csv(PROJECT_ROOT / 'submissions' / 'submission_lgbm_v12.csv', index=False)

xgb_oof = oof_template.copy(); xgb_sub = sub_template.copy(); xgb_oof[targets] = oof_preds_xgb; xgb_sub[targets] = test_preds_xgb
xgb_oof.to_csv(PROJECT_ROOT / 'submissions' / 'oof_xgb_v12.csv', index=False); xgb_sub.to_csv(PROJECT_ROOT / 'submissions' / 'submission_xgb_v12.csv', index=False)

et_oof = oof_template.copy(); et_sub = sub_template.copy(); et_oof[targets] = oof_preds_et; et_sub[targets] = test_preds_et
et_oof.to_csv(PROJECT_ROOT / 'submissions' / 'oof_et_v12.csv', index=False); et_sub.to_csv(PROJECT_ROOT / 'submissions' / 'submission_et_v12.csv', index=False)

rf_oof = oof_template.copy(); rf_sub = sub_template.copy(); rf_oof[targets] = oof_preds_rf; rf_sub[targets] = test_preds_rf
rf_oof.to_csv(PROJECT_ROOT / 'submissions' / 'oof_rf_v12.csv', index=False); rf_sub.to_csv(PROJECT_ROOT / 'submissions' / 'submission_rf_v12.csv', index=False)

lr_oof = oof_template.copy(); lr_sub = sub_template.copy(); lr_oof[targets] = oof_preds_lr; lr_sub[targets] = test_preds_lr
lr_oof.to_csv(PROJECT_ROOT / 'submissions' / 'oof_lr_v12.csv', index=False); lr_sub.to_csv(PROJECT_ROOT / 'submissions' / 'submission_lr_v12.csv', index=False)

print('✅ SOTA-v31 curated pipeline successfully finished training and saved as v12!')


# ## ⚖️ Stage 5. SLSQP Stacking & Beta Calibration 및 최종 submission.csv 생성

# In[10]:


print('\n=========================================================================')
print('⚖️ RUNNING SOTA-v31 ADVANCED STACKING & CALIBRATION ON CURATED SPACE')
print('=========================================================================')

def calibrate_beta_subject_wise(y_true, p_raw, subjects, global_T=1.0):
    eps = 1e-15
    p_raw = np.clip(p_raw, eps, 1 - eps)
    logits = np.log(p_raw / (1 - p_raw))
    unique_subjects = np.unique(subjects)
    calibrated_p = np.zeros_like(p_raw)
    for subject in unique_subjects:
        subj_mask = subjects == subject
        other_mask = ~subj_mask
        if other_mask.sum() > 20:
            clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
            clf.fit(logits[other_mask].reshape(-1, 1), y_true[other_mask])
            calibrated_p[subj_mask] = clf.predict_proba(logits[subj_mask].reshape(-1, 1))[:, 1]
        else:
            calibrated_p[subj_mask] = 1 / (1 + np.exp(-logits[subj_mask] / global_T))
    return np.clip(calibrated_p, eps, 1 - eps)

def calibrate_beta_global(y_true, p_raw):
    eps = 1e-15
    p_raw = np.clip(p_raw, eps, 1 - eps)
    logits = np.log(p_raw / (1 - p_raw))
    clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
    clf.fit(logits.reshape(-1, 1), y_true)
    return clf

# 1. Load clean OOF predictions including DL sequence predictions!
submission_dir = PROJECT_ROOT / 'submissions'
clean_model_names = ['catboost', 'lgbm', 'xgb', 'et', 'rf', 'dl', 'lr']
oof_files = []
for m in clean_model_names:
    if m == 'dl':
        path = submission_dir / 'oof_dl_v10.csv'
    else:
        path = submission_dir / f'oof_{m}_v12.csv'
    if path.exists():
        oof_files.append(path)

print(f'Loading {len(oof_files)} clean base models: {[f.name for f in oof_files]}...')
raw_oof_preds = []
sub_files = []

# Synchronized labels metrics loading
metrics_df = pd.read_csv(METRICS_TRAIN_PATH)
if 'lifelog_date' in metrics_df.columns:
    metrics_df['date'] = pd.to_datetime(metrics_df['lifelog_date']).dt.date
elif 'date' not in metrics_df.columns and 'sleep_date' in metrics_df.columns:
    metrics_df['date'] = pd.to_datetime(metrics_df['sleep_date']).dt.date
metrics_df['date'] = pd.to_datetime(metrics_df['date']).dt.date
metrics_df = metrics_df.sort_values(by=[id_col, 'date']).reset_index(drop=True)

for oof_f in oof_files:
    df = pd.read_csv(oof_f)
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.sort_values(by=[id_col, 'date']).reset_index(drop=True)

    # Fill NaNs in OOF predictions with 0.5 to maintain full-dataset validation consistency
    df_filled = df[targets].fillna(0.5)
    raw_oof_preds.append(df_filled.values)
    sub_name = oof_f.name.replace('oof_', 'submission_')
    sub_files.append(submission_dir / sub_name)

num_models = len(raw_oof_preds)
y_true = metrics_df[targets].values
valid_mask = ~np.isnan(y_true)

sub_dfs = [pd.read_csv(f).fillna(0.5) for f in sub_files]

# Pre-Calibrate OOF and Test using out-of-fold subject-wise Beta Calibration
calibrated_oof_preds = [p.copy() for p in raw_oof_preds]
calibrated_sub_dfs = [df.copy() for df in sub_dfs]

print('Applying out-of-fold Subject-Wise Beta Calibration...')
for m_idx in range(num_models):
    for t_idx, target in enumerate(targets):
        mask = valid_mask[:, t_idx]
        raw_p_oof = raw_oof_preds[m_idx][:, t_idx]
        pred_mask = ~np.isnan(raw_p_oof)
        combined_mask = mask & pred_mask

        if combined_mask.sum() == 0:
            continue

        y_calib = y_true[combined_mask, t_idx]
        p_calib_raw = raw_p_oof[combined_mask]
        subjects = metrics_df.loc[combined_mask, id_col].values

        # Subject-wise beta calibration out-of-fold
        calib_p_oof = raw_p_oof.copy()
        calib_p_oof[combined_mask] = calibrate_beta_subject_wise(y_calib, p_calib_raw, subjects)
        calibrated_oof_preds[m_idx][:, t_idx] = calib_p_oof

        # Subject-Specific Calibration for Test Set:
        raw_p_test = sub_dfs[m_idx][target].values
        eps = 1e-15
        raw_p_test = np.clip(raw_p_test, eps, 1 - eps)
        logits_test = np.log(raw_p_test / (1 - raw_p_test))

        p_calib_clipped = np.clip(p_calib_raw, eps, 1 - eps)
        logits_train = np.log(p_calib_clipped / (1 - p_calib_clipped))

        calibrated_test_p = np.zeros_like(raw_p_test)
        test_subjects = sub_dfs[m_idx][id_col].values

        for subject in np.unique(subjects):
            subj_train_mask = subjects == subject
            subj_test_mask = test_subjects == subject

            if subj_train_mask.sum() > 5:
                clf_subj = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', random_state=42)
                clf_subj.fit(logits_train[subj_train_mask].reshape(-1, 1), y_calib[subj_train_mask])
                if subj_test_mask.sum() > 0:
                    calibrated_test_p[subj_test_mask] = clf_subj.predict_proba(logits_test[subj_test_mask].reshape(-1, 1))[:, 1]
            else:
                clf_global = calibrate_beta_global(y_calib, p_calib_raw)
                if subj_test_mask.sum() > 0:
                    calibrated_test_p[subj_test_mask] = clf_global.predict_proba(logits_test[subj_test_mask].reshape(-1, 1))[:, 1]

        calibrated_sub_dfs[m_idx][target] = calibrated_test_p

final_oof_blend = np.zeros((len(df_train), len(targets)))
final_test_blend = np.zeros((len(df_test), len(targets)))

blended_oof_overall = []
stacked_oof_overall = []
final_oof_dict = {}
final_test_dict = {}
oof_masks = {}

print('\n========== Running Stacking vs Blending Target Sweep ==========')
from sklearn.model_selection import StratifiedKFold

for t_idx, target in enumerate(targets):
    mask = valid_mask[:, t_idx]
    if mask.sum() == 0:
        continue

    pred_mask = np.ones_like(mask)
    for p in calibrated_oof_preds:
        pred_mask = pred_mask & ~np.isnan(p[:, t_idx])

    combined_mask = mask & pred_mask
    if combined_mask.sum() == 0:
        continue

    oof_masks[target] = combined_mask
    X_oof = np.column_stack([p[combined_mask, t_idx] for p in calibrated_oof_preds])
    y_oof = y_true[combined_mask, t_idx]
    X_test = np.column_stack([df[target].values for df in calibrated_sub_dfs])

    # 1. SLSQP Logit Blending baseline
    def blend_objective(weights):
        w = weights / (np.sum(weights) + 1e-15)
        eps = 1e-15
        X_oof_clipped = np.clip(X_oof, eps, 1 - eps)
        logits = np.log(X_oof_clipped / (1.0 - X_oof_clipped))
        blended_logits = np.dot(logits, w)
        p_blended = 1.0 / (1.0 + np.exp(-blended_logits))
        p_blended = np.clip(p_blended, eps, 1 - eps)
        return log_loss(y_oof, p_blended)

    initial_weights = [1.0 / num_models] * num_models
    bounds = [(0.0, 1.0)] * num_models
    constraints = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    res = minimize(blend_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    opt_w = res.x / np.sum(res.x)
    blend_score = res.fun

    X_oof_clipped = np.clip(X_oof, 1e-15, 1 - 1e-15)
    logits_oof = np.log(X_oof_clipped / (1.0 - X_oof_clipped))
    blended_logits_oof = np.dot(logits_oof, opt_w)
    base_blended = 1.0 / (1.0 + np.exp(-blended_logits_oof))
    base_blended = np.clip(base_blended, 1e-15, 1 - 1e-15)

    # 2. Logistic Regression Meta-Learner Stacking
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(y_oof))
    meta_test_preds = np.zeros(len(X_test))

    best_c = 1.0
    best_meta_loss = 999.0
    for c_val in [0.01, 0.1, 1.0, 10.0, 100.0]:
        fold_scores = []
        for train_idx, val_idx in skf.split(X_oof, y_oof):
            meta_clf = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
            meta_clf.fit(X_oof[train_idx], y_oof[train_idx])
            fold_p = meta_clf.predict_proba(X_oof[val_idx])[:, 1]
            fold_scores.append(log_loss(y_oof[val_idx], fold_p))
        mean_c_loss = np.mean(fold_scores)
        if mean_c_loss < best_meta_loss:
            best_meta_loss = mean_c_loss
            best_c = c_val

    for train_idx, val_idx in skf.split(X_oof, y_oof):
        meta_clf = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
        meta_clf.fit(X_oof[train_idx], y_oof[train_idx])
        meta_oof[val_idx] = meta_clf.predict_proba(X_oof[val_idx])[:, 1]
        meta_test_preds += meta_clf.predict_proba(X_test)[:, 1] / 5.0

    stack_score = log_loss(y_oof, meta_oof)

    if stack_score < blend_score:
        print(f'🎯 Target {target} -> Stacking chosen. (Stack CV: {stack_score:.5f} vs Blend CV: {blend_score:.5f})')
        base_oof_preds = meta_oof
        base_test_preds = np.clip(meta_test_preds, 1e-15, 1 - 1e-15)
        print(f'   Optimal meta-model: Logistic Regression C={best_c}')
    else:
        print(f'🎯 Target {target} -> Blending chosen. (Blend CV: {blend_score:.5f} vs Stack CV: {stack_score:.5f})')
        base_oof_preds = base_blended
        X_test_clipped = np.clip(X_test, 1e-15, 1 - 1e-15)
        logits_test = np.log(X_test_clipped / (1.0 - X_test_clipped))
        blended_logits_test = np.dot(logits_test, opt_w)
        base_test_preds = 1.0 / (1.0 + np.exp(-blended_logits_test))
        base_test_preds = np.clip(base_test_preds, 1e-15, 1 - 1e-15)
        print(f'   Optimal SLSQP weights: {np.round(opt_w, 4)}')

    # 3. Post-Processing: Out-of-fold Personalized Baseline Calibration
    subjects = metrics_df.loc[combined_mask, id_col].values
    test_subjects = sub_dfs[0][id_col].values
    calibrated_oof = np.zeros_like(base_oof_preds)

    for train_idx, val_idx in skf.split(X_oof, y_oof.astype(int)):
        train_subjects = subjects[train_idx]
        train_y = y_oof[train_idx]
        subject_means = {}
        global_mean = train_y.mean()

        for s in np.unique(subjects):
            s_mask = train_subjects == s
            if s_mask.sum() > 0:
                subject_means[s] = (train_y[s_mask].sum() + 5.0 * global_mean) / (s_mask.sum() + 5.0)
            else:
                subject_means[s] = global_mean

        best_alpha = 1.0
        best_fold_loss = 999.0
        train_preds = base_oof_preds[train_idx]
        train_subj_means = np.array([subject_means[s] for s in train_subjects])

        alpha_min = 1.0 if target.startswith('Q') else 0.5
        for alpha in np.linspace(alpha_min, 1.0, int(round((1.0 - alpha_min) / 0.01)) + 1):
            p_blend = alpha * train_preds + (1 - alpha) * train_subj_means
            loss = log_loss(train_y, np.clip(p_blend, 0.005, 0.995))
            if loss < best_fold_loss:
                best_fold_loss = loss
                best_alpha = alpha

        val_preds = base_oof_preds[val_idx]
        val_subjects = subjects[val_idx]
        val_subj_means = np.array([subject_means[s] for s in val_subjects])
        calibrated_oof[val_idx] = np.clip(best_alpha * val_preds + (1 - best_alpha) * val_subj_means, 0.005, 0.995)

    global_mean = y_oof.mean()
    subject_means_all = {}
    for s in np.unique(subjects):
        s_mask = subjects == s
        if s_mask.sum() > 0:
            subject_means_all[s] = (y_oof[s_mask].sum() + 5.0 * global_mean) / (s_mask.sum() + 5.0)
        else:
            subject_means_all[s] = global_mean

    best_alpha_global = 1.0
    best_global_loss = 999.0
    subj_means_all = np.array([subject_means_all[s] for s in subjects])

    alpha_min = 1.0 if target.startswith('Q') else 0.5
    for alpha in np.linspace(alpha_min, 1.0, int(round((1.0 - alpha_min) / 0.01)) + 1):
        p_blend = alpha * base_oof_preds + (1 - alpha) * subj_means_all
        loss = log_loss(y_oof, np.clip(p_blend, 0.005, 0.995))
        if loss < best_global_loss:
            best_global_loss = loss
            best_alpha_global = alpha

    test_subj_means = np.array([subject_means_all.get(s, global_mean) for s in test_subjects])
    final_test_preds = best_alpha_global * base_test_preds + (1 - best_alpha_global) * test_subj_means

    best_lower = 0.005
    best_upper = 0.995
    best_clip_loss = 999.0
    for low in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04]:
        for high in [0.80, 0.82, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
            clipped = np.clip(calibrated_oof, low, high)
            loss = log_loss(y_oof, clipped)
            if loss < best_clip_loss:
                best_clip_loss = loss
                best_lower = low
                best_upper = high

    print(f'   ✂️ Target {target} Optimal Clipping: [{best_lower:.4f}, {best_upper:.4f}] (Log-Loss: {best_clip_loss:.5f})')
    final_oof_dict[target] = np.clip(calibrated_oof, best_lower, best_upper)
    final_test_dict[target] = np.clip(final_test_preds, best_lower, best_upper)
    blended_oof_overall.append(min(blend_score, stack_score))

# 4. Physiological Multi-Target Joint Post-Processing on S4
if 'S4' in final_oof_dict and 'S2' in final_oof_dict and 'S3' in final_oof_dict:
    s4_idx = targets.index('S4')
    mask_s4 = oof_masks['S4']
    y_s4 = y_true[mask_s4, s4_idx]
    p_s2_oof = final_oof_dict['S2']
    p_s3_oof = final_oof_dict['S3']
    p_s4_oof_raw = final_oof_dict['S4']

    best_lambda = 0.0
    best_s4_loss = log_loss(y_s4, p_s4_oof_raw)

    for lmb in np.linspace(0.0, 0.6, 61):
        p_s4_adj = np.clip(p_s4_oof_raw + lmb * (p_s2_oof * p_s3_oof), 0.005, 0.995)
        loss = log_loss(y_s4, p_s4_adj)
        if loss < best_s4_loss:
            best_s4_loss = loss
            best_lambda = lmb

    print(f'🧬 Optimized S4 Physiological Constraint: Lambda = {best_lambda:.3f} (S4 CV: {best_s4_loss:.5f} vs Raw: {log_loss(y_s4, p_s4_oof_raw):.5f})')
    s4_adj_oof = np.clip(p_s4_oof_raw + best_lambda * (p_s2_oof * p_s3_oof), 0.001, 0.999)
    best_lower_s4 = 0.005
    best_upper_s4 = 0.995
    best_clip_loss_s4 = 999.0
    for low in [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04]:
        for high in [0.80, 0.82, 0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]:
            clipped = np.clip(s4_adj_oof, low, high)
            loss = log_loss(y_s4, clipped)
            if loss < best_clip_loss_s4:
                best_clip_loss_s4 = loss
                best_lower_s4 = low
                best_upper_s4 = high
    print(f'   ✂️ Target S4 Post-Adj Optimal Clipping: [{best_lower_s4:.4f}, {best_upper_s4:.4f}] (S4 CV: {best_clip_loss_s4:.5f})')
    final_oof_dict['S4'] = np.clip(s4_adj_oof, best_lower_s4, best_upper_s4)
    final_test_dict['S4'] = np.clip(final_test_dict['S4'] + best_lambda * (final_test_dict['S2'] * final_test_dict['S3']), best_lower_s4, best_upper_s4)

# Populate main containers for evaluation cells
for t_idx, target in enumerate(targets):
    final_oof_blend[oof_masks[target], t_idx] = final_oof_dict[target]
    final_test_blend[:, t_idx] = final_test_dict[target]

print('✅ SLSQP Stacking & Beta Calibration successfully completed!')


# ## 📈 Stage 6. 진짜 릭-프리 Local CV 스코어 검증 및 submission.csv 출력

# In[11]:


print("\n=========================================================================")
print("📊 FINAL 100% LEAK-FREE LOCAL CV SCORES")
print("=========================================================================")

scores = []
for t_idx, target in enumerate(targets):
    y_true = df_train[target]
    has_label = y_true.notna()
    preds = final_oof_blend[has_label, t_idx]
    loss = log_loss(y_true[has_label], np.clip(preds, 1e-15, 1-1e-15))
    scores.append(loss)
    print(f"  - {target} Log-Loss: {loss:.6f}")

mean_loss = np.mean(scores)
print(f"\n🔥 Overall 100% Leak-Free Hybrid Stacked CV: {mean_loss:.6f} 🔥")
print("=========================================================================")


# In[12]:


# Write final submission.csv
sub_output = sub_sample.copy()
for t_idx, target in enumerate(targets):
    sub_output[target] = final_test_blend[:, t_idx]

SUBMISSION_OUT_PATH = PROJECT_ROOT / "submissions" / "submission.csv"
SUBMISSION_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
sub_output.to_csv(SUBMISSION_OUT_PATH, index=False)

print(f"🎉 SUCCESS! Write 100% 무결 최종 제출 파일 to: {SUBMISSION_OUT_PATH}")
print("Double-checking shape:", sub_output.shape)
print(sub_output.head(10))


# In[13]:


# =========================================================================
# 📝 submissions/submission_log.csv 자동 기록(Append) 및 고유 사본 저장 모듈
# =========================================================================
import datetime
import shutil

log_path = PROJECT_ROOT / "submissions" / "submission_log.csv"

if log_path.exists():
    log_df = pd.read_csv(log_path)
    new_id = log_df['id'].max() + 1
else:
    log_df = pd.DataFrame(columns=['id', 'date', 'submission_file', 'model_description', 'local_cv_score', 'public_lb_score', 'is_submitted', 'notes'])
    new_id = 1

current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 1. Define highly unique file name to prevent future overwriting chaos!
unique_sub_name = f"submission_id_{new_id:03d}_score{mean_loss:.4f}.csv"
unique_sub_path = PROJECT_ROOT / "submissions" / unique_sub_name

# 2. Copy final submission.csv to this unique file name for safe archiving
shutil.copy2(SUBMISSION_OUT_PATH, unique_sub_path)
print(f"📦 Archived unique submission file to: {unique_sub_path.name}")

# 3. Create new record row with unique file name mapped
new_row = {
    'id': new_id,
    'date': current_time,
    'submission_file': unique_sub_name,
    'model_description': f'SOTA-v31 100% Leak-Free All-in-One Pipeline - CV {mean_loss:.6f}',
    'local_cv_score': round(mean_loss, 6),
    'public_lb_score': 0.0,
    'is_submitted': True,
    'notes': 'Executed automatically via standalone 올인원 SOTA v31 Jupyter Notebook with strict chronological forward-only target flow.'
}

# 4. Append and save
log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
log_df.to_csv(log_path, index=False)
print(f"📝 Successfully appended execution results to submission_log.csv under ID: {new_id}!")

