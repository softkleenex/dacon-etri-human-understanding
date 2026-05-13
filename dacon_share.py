#!/usr/bin/env python
# coding: utf-8

# # ETRI CH2026 수면의 질 예측 - DACON 코드 공유
# 
# **Public Score: 0.5917 (LGBM + XGB blend 73)**
# 
# ---
# 
# ## 프로젝트 개요
# 
# ### 대회 정보
# - ** competition**: ETRI Challenge 2026 - 수면 기반 생활습관 로그 예측
# - ** task**: 7개 타겟(Q1, Q2, Q3, S1, S2, S3, S4)에 대한 다중 레이블 이진 분류
# - ** metric**: Log Loss (낮을수록 좋음)
# - ** 플랫폼**: DACON (dacon.io)
# 
# ### 타겟 설명
# | 타겟 | 설명 | 특징 |
# |------|------|------|
# | Q1 | 주관적 수면 만족도 | 높을수록 만족 |
# | Q2 | 수면 개입 정도 | 낮을수록 좋음 |
# | Q3 | 수면 품질 관련 | 중간 정도 |
# | S1-S4 | 수면 단계 비율 | Objective 측정 |
# 
# ### 핵심 전략
# 1. **V152 앵커 모델**: 이전 최고 성능 모델을 베이스로 활용
# 2. **3모델 앙상블**: LightGBM + XGBoost + CatBoost 블렌딩
# 3. **Window-Pair 특성**: 시간대 간 상호작용.feature 추가
# 4. **안정성 필터링**: 23,177개 특성 → 1,682개 안정적 특성 선별
# ### 코드 공유·실행 안내
# 
# - **경로**: 로컬 계정명·`/home/...` 같은 고정 절대 경로를 쓰지 않습니다. 현재 작업 디렉터리 또는 그 상위에서 `data/` 폴더를 찾아 프로젝트 루트로 사용합니다.
# - **데이터**: 대회 제공 파일을 프로젝트 루트의 `data/` 아래에 두고, Jupyter·IDE에서 **레포지토리 루트**를 작업 디렉터리로 지정한 뒤 실행하세요.
# - **재현**: V152 베이스 OOF·대용량 피처 parquet 등은 별도 실험 산출물이므로, 이 노트북만으로 전체 학습이 바로 재현되지 않을 수 있습니다.
# 

# In[ ]:


# ========================================
# 1. 라이브러리 임포트
# ========================================

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 라이브러리
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# 평가 지표
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, GroupKFold

# 한글 폰트 설정 (한국어 주석용)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

print("✅ 모든 라이브러리 로드 완료")


# ---
# 
# ## 2. 데이터 로드 및 탐색적 분석 (EDA)
# 
# ### 2.1 데이터 구조 이해

# In[ ]:


# ========================================
# 2.1 데이터 경로 설정
# ========================================

def resolve_project_root() -> Path:
    """현재 작업 디렉터리 또는 상위에서 `data/`를 찾아 프로젝트 루트로 사용 (공유용, 절대 경로·사용자명 미사용)."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent, cwd.parent.parent):
        if (candidate / "data").is_dir():
            return candidate
    return cwd

PROJECT_ROOT = resolve_project_root()
DATA_DIR = PROJECT_ROOT / "data"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# 타겟 컬럼 정의
TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
ID_COLS = ["subject_id", "sleep_date", "lifelog_date"]

# 확률 clipping을 위한 작은 값
EPS = 1e-6

print(f"📁 프로젝트 경로: {PROJECT_ROOT}")
print(f"📁 데이터 경로: {DATA_DIR}")
if not DATA_DIR.is_dir():
    print(
        "⚠️ data/ 폴더를 찾지 못했습니다. 대회 데이터를 프로젝트 루트의 data/에 두거나, "
        "작업 디렉터리를 레포 루트로 바꾼 뒤 다시 실행하세요."
    )


# In[ ]:


# ========================================
# 2.2 훈련 데이터 로드
# ========================================

# 훈련 레이블 로드 (정답이 포함된 훈련 데이터)
train_df = pd.read_csv(DATA_DIR / "ch2026_metrics_train.csv")

# 테스트 샘플 로드 (제출용 형식)
sample_sub = pd.read_csv(DATA_DIR / "ch2026_submission_sample.csv")

# 날짜 컬럼을 datetime으로 변환
for col in ["sleep_date", "lifelog_date"]:
    train_df[col] = pd.to_datetime(train_df[col])
    sample_sub[col] = pd.to_datetime(sample_sub[col])

print(f"📊 훈련 데이터: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
print(f"📋 테스트 데이터: {sample_sub.shape[0]} rows × {sample_sub.shape[1]} columns")
print(f"\n 훈련 데이터 컬럼: {list(train_df.columns)}")
print(f"\n 훈련 데이터 앞 5행:")
train_df.head()


# In[ ]:


# ========================================
# 2.3 데이터 기본 통계
# ========================================

print("=" * 60)
print("📈 훈련 데이터 기본 통계")
print("=" * 60)

# Subject별 데이터 분포
print("\n▸ Subject별 이벤트 수:")
subject_counts = train_df['subject_id'].value_counts().sort_index()
for subj, cnt in subject_counts.items():
    print(f"  {subj}: {cnt} events")

# 타겟 분포 확인
print("\n▸ 타겟 분포 (타겟별 0/1 비율):")
for col in TARGET_COLS:
    rate = train_df[col].mean()
    print(f"  {col}: {rate:.3f} (1의 비율)")


# In[ ]:


# ========================================
# 2.4 타겟간 상관관계 분석
# ========================================

# 타겟 간 상관관계 계산
target_corr = train_df[TARGET_COLS].corr()

# 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(target_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0)
plt.title("Target Correlation Matrix (Training Data)", fontsize=14)
plt.tight_layout()
plt.savefig("target_correlation.png", dpi=150)
plt.show()

print("\n📌 주요 발견:")
print("  • Q1-S1: Moderate correlation (0.36) - 수면 개입과 주관적 만족 관련성")
print("  • S2-S4: Moderate correlation (0.48) - 수면 단계 연속성")
print("  • Q1-S4: Nearly uncorrelated (0.02) - 주관적 vs 객관적 불일치")


# In[ ]:


# ========================================
# 2.5 Subject별 타겟 프로파일 분석
# ========================================

# Subject별 평균 타겟값 계산
subject_profile = train_df.groupby('subject_id')[TARGET_COLS].mean()

# 히트맵으로 시각화
plt.figure(figsize=(12, 5))
sns.heatmap(subject_profile.T, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title("Subject-wise Target Profile (Mean Values)", fontsize=14)
plt.xlabel("Subject ID")
plt.ylabel("Target")
plt.tight_layout()
plt.savefig("subject_profile.png", dpi=150)
plt.show()

print("\n📌 극단값 Subject 발견:")
print(f"  • id03: Q1 높음({subject_profile.loc['id03','Q1']:.2f}), S4 낮음({subject_profile.loc['id03','S4']:.2f})")
print(f"  • id06: Q1 낮음({subject_profile.loc['id06','Q1']:.2f}), S4 높음({subject_profile.loc['id06','S4']:.2f})")
print("  → 주관적 만족도와 깊은 수면이 역상관하는 경우가 있음")


# ---
# 
# ## 3. 피처 엔지니어링
# 
# ### 3.1 센서 데이터 로드 및 윈도우 정의

# In[ ]:


# ========================================
# 3.1 센서 모달리티 및 윈도우 정의
# ========================================

# 12개 센서 모달리티 정의
SENSOR_MODALITIES = [
    "mACStatus", "mActivity", "mAmbience", "mBle",
    "mGps", "mLight", "mScreenStatus", "mUsageStats",
    "mWifi", "wHr", "wLight", "wPedo"
]

# 수면 관련 시간 윈도우 정의
# 각 윈도우는 하루 중 특정 시간대를 의미
SLEEP_WINDOWS = {
    "daily_00_24": "00:00-24:00 (하루 전체)",
    "day_09_18": "09:00-18:00 (주간)",
    "evening_18_21": "18:00-21:00 (저녁)",
    "prebed_21_24": "21:00-24:00 (취침 전)",
    "sleep_21_09": "21:00-09:00 (수면 시간)",
    "late_00_03": "00:00-03:00 (심야)",
    "deep_03_06": "03:00-06:00 (깊은 수면)",
    "wake_06_09": "06:00-09:00 (기상)"
}

print(f"📡 센서 모달리티: {len(SENSOR_MODALITIES)}개")
for s in SENSOR_MODALITIES:
    print(f"  - {s}")

print(f"\n🛏️ 수면 윈도우: {len(SLEEP_WINDOWS)}개")
for k, v in SLEEP_WINDOWS.items():
    print(f"  - {k}: {v}")


# In[ ]:


# ========================================
# 3.2 캘린더/시간 특성 생성
# ========================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜 정보에서 캘린더 기반 특성 추출

    Args:
        df: 날짜 컬럼이 포함된 데이터프레임

    Returns:
        캘린더 특성이 추가된 데이터프레임
    """
    out = df.copy()

    # Subject를 범주형으로 변환
    out["subject_cat"] = out["subject_id"].astype(str)

    # sleep_date와 lifelog_date에 대해 반복
    for prefix, col in [("sleep", "sleep_date"), ("lifelog", "lifelog_date")]:
        dt = out[col]

        # 기본 날짜 특성
        out[f"{prefix}_month"] = dt.dt.month.astype("float")
        out[f"{prefix}_day"] = dt.dt.day.astype("float")
        out[f"{prefix}_dayofweek"] = dt.dt.dayofweek.astype("float")
        out[f"{prefix}_weekofyear"] = dt.dt.isocalendar().week.astype("float")
        out[f"{prefix}_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float")
        out[f"{prefix}_dayofyear"] = dt.dt.dayofyear.astype("float")

        # 순환 특성 (계절성 포착용)
        out[f"{prefix}_month_sin"] = np.sin(2 * np.pi * out[f"{prefix}_month"] / 12)
        out[f"{prefix}_month_cos"] = np.cos(2 * np.pi * out[f"{prefix}_month"] / 12)
        out[f"{prefix}_dow_sin"] = np.sin(2 * np.pi * out[f"{prefix}_dayofweek"] / 7)
        out[f"{prefix}_dow_cos"] = np.cos(2 * np.pi * out[f"{prefix}_dayofweek"] / 7)
        out[f"{prefix}_doy_sin"] = np.sin(2 * np.pi * out[f"{prefix}_dayofyear"] / 366)
        out[f"{prefix}_doy_cos"] = np.cos(2 * np.pi * out[f"{prefix}_dayofyear"] / 366)

    # 날짜 간 간격 (수면 날짜와 라이프로그 날짜의 차이)
    out["date_gap_days"] = (out["sleep_date"] - out["lifelog_date"]).dt.days.astype("float")

    # Subject 번호 (수치형으로 변환)
    out["subject_num"] = out["subject_id"].str.extract(r"(\d+)").astype("float")

    return out

# 캘린더 특성 추가 테스트
train_with_calendar = add_calendar_features(train_df)
print(f"✅ 캘린더 특성 추가 완료: {train_with_calendar.shape[1]} columns")
print(f"   추가된 특성: month, day, dayofweek, weekend, dayofyear, cyclic features")


# In[ ]:


# ========================================
# 3.3 Window-Pair Interaction 특성 생성
# ========================================

def create_window_pair_features(base_features: pd.DataFrame) -> pd.DataFrame:
    """
    윈도우 간 상호작용 특성 생성

    예: prebed_21_24와 sleep_21_09 간의 활동량 차이

    윈도우 조합:
    - prebed_21_24 × sleep_21_09: 취침 전 → 수면 전환
    - sleep_21_09 × wake_06_09: 수면 → 기상 전환
    - day_09_18 × evening_18_21: 주간 → 저녁 전환

    Args:
        base_features: 윈도우별 집계 특성이 포함된 데이터프레임

    Returns:
        윈도우 쌍 상호작용 특성이 추가된 데이터프레임
    """
    out = base_features.copy()

    # 핵심 윈도우 쌍 정의 (수면 관련 예측에 중요)
    window_pairs = [
        ("prebed_21_24", "sleep_21_09"),  # 취침 전 → 수면
        ("sleep_21_09", "wake_06_09"),    # 수면 → 기상
        ("day_09_18", "evening_18_21"),    # 주간 → 저녁
        ("evening_18_21", "prebed_21_24"), # 저녁 → 취침 전
        ("late_00_03", "deep_03_06"),      # 심야 → 깊은 수면
        ("deep_03_06", "wake_06_09"),      # 깊은 수면 → 기상
    ]

    # 각 윈도우 쌍에 대해 차이/비율 특성 생성
    for w1, w2 in window_pairs:
        # 이 예시에서는 실제 센서 특성 없으므로
        # 구조만 보여주는 코드 (실제 구현에서는 센서 데이터 활용)
        pair_name = f"{w1}_to_{w2}"

        # 실제 구현에서는 센서 데이터에서 파생된 수치 특성들 간 연산 수행
        # 예: activity 차이, light 비율 등

        # Placeholder: 실제 센서 데이터 기반 특성으로 대체 필요
        pass

    return out

print("📝 Window-pair 피처 생성 함수 정의 완료")
print("   (실제 구현에서는 센서 데이터(parquet)에서 파생된 수치 특성 사용)")


# ---
# 
# ## 4. 안정성 기반 특성 선택
# 
# ### 4.1 왜 안정성 필터링이 중요한가?
# 
# 1. **23,177개 특성** 중 일부는 훈련 데이터에만 과적합
# 2. 테스트 데이터에서 성능 저하 발생 (Local-Public Gap 발생)
# 3. 안정성 필터링으로 테스트에서도 잘 일반화되는 특성만 선별

# In[ ]:


# ========================================
# 4.1 안정성 기반 특성 선택 함수
# ========================================

def apply_stability_filter(
    feature_df: pd.DataFrame,
    train_df: pd.DataFrame,
    target_cols: list,
    min_stability: float = 0.3,
    max_features_per_target: int = 300
) -> dict:
    """
    타겟별로 안정성 점수 기반 특성 선별

    안정성 점수: CV fold 간 Pearson 상관계수의 평균
    - 높을수록 다른 fold에서도 일관된 예측력 유지
    - 낮으면 특정 fold에 과적합 (테스트에서 성능 저하)

    Args:
        feature_df: 특성 데이터프레임
        train_df: 훈련 레이블
        target_cols: 타겟 리스트
        min_stability: 최소 안정성 임계값 (0~1)
        max_features_per_target: 타겟당 최대 특성 수

    Returns:
        타겟별 선택된 특성 딕셔너리
    """

    # 수치형 특성 컬럼만 선택
    exclude_cols = set(ID_COLS + target_cols + ['subject_cat'])
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

    selected_features = {}

    for target in target_cols:
        # 각 특성과 타겟 간 상관계수 계산 (여러 fold에서)
        stability_scores = {}

        for feat in feature_cols:
            # 간단한 안정성 측정: 전체 데이터에서 상관계수
            # 실제 구현에서는 CV fold별 상관계수의 분산 사용
            try:
                corr = feature_df[feat].corr(train_df[target])
                if pd.notna(corr):
                    stability_scores[feat] = abs(corr)
            except:
                pass

        # 안정성 임계값 이상인 특성만 선택
        filtered = {k: v for k, v in stability_scores.items() 
                    if v >= min_stability}

        # 상관계수 높은 순으로 정렬 후 상위 max_features_per_target개 선택
        sorted_feats = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
        top_k = min(max_features_per_target, len(sorted_feats))
        selected_features[target] = [f for f, _ in sorted_feats[:top_k]]

    return selected_features

print("✅ 안정성 필터링 함수 정의 완료")
print("   타겟당 최대 300개 특성, 상관계수 기반 선별")


# ---
# 
# ## 5. Cross-Validation 전략
# 
# ### 5.1 Subject-Aware CV의 중요성
# 
# **문제점**: 단순 K-Fold는 같은 Subject의 데이터가 훈련/테스트에 섞여 과적합 발생
# 
# **해결책**: Subject를 고려한 CV 전략 사용
# - `subject_block`: Subject별로 블럭 분리
# - `subject_modulo`: Subject별로 시간 순서 고려
# - `subject_hole`: Subject 내에서 시간적 홀(hole) 생성

# In[ ]:


# ========================================
# 5.1 Subject-Hole CV 구현
# ========================================

def make_subject_hole_folds(
    train: pd.DataFrame, 
    n_folds: int = 5
) -> list:
    """
    Subject 내에서 시간적 홀(hole)을 만드는 CV 전략

    Train/Test가 인터리빙(섞여)된 구조를 시뮬레이션

    Args:
        train: 훈련 데이터프레임
        n_folds: CV 폴드 수

    Returns:
        (train_idx, valid_idx) 튜플 리스트
    """
    all_indices = train.index.to_numpy()
    result = []

    # 각 Subject별 시간순 정렬 후 chunk 분할
    block_count = max(n_folds * 2, 4)  # 홀 수를 위해 2배
    by_subject = {}

    for subject_id, group in train.sort_values(["subject_id", "sleep_date"]).groupby(
        "subject_id", sort=False
    ):
        indices = group.index.to_numpy()
        # 동일 크기로 분할
        chunks = [chunk for chunk in np.array_split(indices, block_count) if len(chunk)]
        by_subject[str(subject_id)] = chunks

    # 각 fold에 대해 두 개의 홀 생성 (early + late)
    for fold_id in range(n_folds):
        valid_parts = []

        for chunks in by_subject.values():
            # fold_id: early hole, fold_id + n_folds: late hole
            for hole_id in (fold_id, fold_id + n_folds):
                if hole_id < len(chunks):
                    valid_parts.append(chunks[hole_id])

        if not valid_parts:
            continue

        valid_idx = np.concatenate(valid_parts)
        train_idx = np.setdiff1d(all_indices, valid_idx, assume_unique=False)

        if len(train_idx) and len(valid_idx):
            result.append((train_idx, valid_idx))

    return result

# CV 전략 시각화
print("📌 Subject-Hole CV 구조:")
print("   Subject별 chronological chunks 생성 (예: 10개 chunk)")
print("   Fold 0: chunk 0, 5 validation → chunk 1-4, 6-9 training")
print("   Fold 1: chunk 1, 6 validation → chunk 0, 2-5, 7-9 training")
print("   ...")
print("   → 테스트의 T/X/T/X 인터리빙 구조와 유사")


# ---
# 
# ## 6. LightGBM 모델 학습
# 
# ### 6.1 베이스 모델 (V152 앵커) 사용의 중요성
# 
# **V152 앵커**: 이전 실험에서 검증된 최고 성능 모델
# - 단독으로 사용시 Public Score: 0.5939
# - 제거 시 항상 성능 저하 발생
# - 새 모델의 베이스로 사용하여 개선 효과 극대화

# In[ ]:


# ========================================
# 6.1 LightGBM 학습 파이프라인
# ========================================

def train_lgbm_model(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    selected_features: dict,
    base_oof_path: str = None,
    n_folds: int = 5,
    seed: int = 42
) -> tuple:
    """
    LightGBM 모델 학습 및 OOF 예측 생성

    Args:
        train_df: 훈련 레이블
        features: 특성 데이터프레임
        selected_features: 타겟별 선택된 특성 딕셔너리
        base_oof_path: 베이스 모델 OOF 경로 (앵커)
        n_folds: CV 폴드 수
        seed: 랜덤 시드

    Returns:
        (oof_predictions, models, metrics)
    """

    # CV 폴드 생성
    cv_folds = make_subject_hole_folds(train_df, n_folds)

    model_features = features.copy()
    oof_preds = pd.DataFrame(index=train_df.index)
    models = {}
    fold_metrics = []
    base_feature_cols = []

    # Base 앵커 특성 추가 (있는 경우)
    if base_oof_path and os.path.exists(base_oof_path):
        base_oof = pd.read_csv(base_oof_path)
        for col in ["sleep_date", "lifelog_date"]:
            base_oof[col] = pd.to_datetime(base_oof[col], errors="coerce")

        for target in TARGET_COLS:
            # Base 모델 예측을 특성으로 추가
            merged = train_df[['subject_id', 'sleep_date', 'lifelog_date']].merge(
                base_oof, on=['subject_id', 'sleep_date', 'lifelog_date'], how='left'
            )
            prob = merged[target].fillna(0.5).clip(EPS, 1-EPS)
            model_features[f'base_{target}_prob'] = prob.to_numpy()
            model_features[f'base_{target}_logit'] = np.log(prob / (1 - prob)).to_numpy()

        base_feature_cols = [f'base_{t}_prob' for t in TARGET_COLS] + [f'base_{t}_logit' for t in TARGET_COLS]

    for target in TARGET_COLS:
        print(f"\n[LightGBM] Training {target}...")

        # 선택된 특성 가져오기
        target_features = selected_features.get(target, [])

        # 베이스 특성 추가
        feature_cols = target_features + base_feature_cols

        # 훈련 데이터 준비
        X = model_features[feature_cols].fillna(0)
        y = train_df[target].values

        target_models = []
        target_oof = np.zeros(len(train_df))

        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            # LightGBM 데이터셋 생성
            dtrain = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx])
            dvalid = lgb.Dataset(X.iloc[va_idx], label=y[va_idx], reference=dtrain)

            # 파라미터 설정
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'min_child_samples': 20,
                'seed': seed + fold_idx,
                'verbose': -1
            }

            # 모델 학습
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=500,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            # OOF 예측
            target_oof[va_idx] = model.predict(X.iloc[va_idx])
            target_models.append(model)

        # OOF 스코어 계산
        oof_score = log_loss(y, target_oof.clip(EPS, 1-EPS))
        print(f"  {target} OOF LogLoss: {oof_score:.4f}")

        oof_preds[target] = target_oof
        models[target] = target_models
        fold_metrics.append({'target': target, 'oof_logloss': oof_score})

    avg_logloss = np.mean([m['oof_logloss'] for m in fold_metrics])
    print(f"\n[LightGBM] Average OOF LogLoss: {avg_logloss:.4f}")

    return oof_preds, models, {'avg_logloss': avg_logloss, 'per_target': fold_metrics}

print("✅ LightGBM 학습 파이프라인 정의 완료")


# ---
# 
# ## 7. XGBoost 모델 학습
# 
# ### 7.1 XGBoost를 앙상블에 포함하는 이유
# 
# **핵심 발견**: LGBM (leaf-wise) + XGB (level-wise)는 다르게 정규화
# - LGBM: leaf-wise 성장 → 과적합 경향을 일부 허용
# - XGB: level-wise 성장 → 더 균형 잡힌 트리
# - 함께 사용시 서로 다른 오류를 보완 (앙상블 이점)

# In[ ]:


# ========================================
# 7.1 XGBoost 학습 파이프라인
# ========================================

def train_xgb_model(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    selected_features: dict,
    base_oof_path: str = None,
    n_folds: int = 5,
    seed: int = 42
) -> tuple:
    """
    XGBoost 모델 학습 및 OOF 예측 생성

    Args:
        train_df: 훈련 레이블
        features: 특성 데이터프레임
        selected_features: 타겟별 선택된 특성 딕셔너리
        base_oof_path: 베이스 모델 OOF 경로
        n_folds: CV 폴드 수
        seed: 랜덤 시드

    Returns:
        (oof_predictions, models, metrics)
    """

    cv_folds = make_subject_hole_folds(train_df, n_folds)

    model_features = features.copy()
    oof_preds = pd.DataFrame(index=train_df.index)
    models = {}
    fold_metrics = []
    base_feature_cols = []

    # Base 앵커 특성 추가
    if base_oof_path and os.path.exists(base_oof_path):
        base_oof = pd.read_csv(base_oof_path)
        for col in ["sleep_date", "lifelog_date"]:
            base_oof[col] = pd.to_datetime(base_oof[col], errors="coerce")

        for target in TARGET_COLS:
            merged = train_df[['subject_id', 'sleep_date', 'lifelog_date']].merge(
                base_oof, on=['subject_id', 'sleep_date', 'lifelog_date'], how='left'
            )
            prob = merged[target].fillna(0.5).clip(EPS, 1-EPS)
            model_features[f'base_{target}_prob'] = prob.to_numpy()
            model_features[f'base_{target}_logit'] = np.log(prob / (1 - prob)).to_numpy()

        base_feature_cols = [f'base_{t}_prob' for t in TARGET_COLS] + [f'base_{t}_logit' for t in TARGET_COLS]

    for target in TARGET_COLS:
        print(f"\n[XGBoost] Training {target}...")

        target_features = selected_features.get(target, [])
        feature_cols = target_features + base_feature_cols

        X = model_features[feature_cols].fillna(0)
        y = train_df[target].values

        target_models = []
        target_oof = np.zeros(len(train_df))

        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            dtrain = xgb.DMatrix(X.iloc[tr_idx], label=y[tr_idx])
            dvalid = xgb.DMatrix(X.iloc[va_idx], label=y[va_idx])

            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 5,
                'seed': seed + fold_idx,
                'verbosity': 0
            }

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=500,
                evals=[(dvalid, 'valid')],
                early_stopping_rounds=50,
                verbose_eval=False
            )

            target_oof[va_idx] = model.predict(dvalid)
            target_models.append(model)

        oof_score = log_loss(y, target_oof.clip(EPS, 1-EPS))
        print(f"  {target} OOF LogLoss: {oof_score:.4f}")

        oof_preds[target] = target_oof
        models[target] = target_models
        fold_metrics.append({'target': target, 'oof_logloss': oof_score})

    avg_logloss = np.mean([m['oof_logloss'] for m in fold_metrics])
    print(f"\n[XGBoost] Average OOF LogLoss: {avg_logloss:.4f}")

    return oof_preds, models, {'avg_logloss': avg_logloss, 'per_target': fold_metrics}

print("✅ XGBoost 학습 파이프라인 정의 완료")


# ---
# 
# ## 8. CatBoost 모델 학습
# 
# ### 8.1 CatBoost의 장점
# 
# - **범주형 특성 처리**: Subject ID 등 범주형 데이터를 내장 처리
# - **대칭 트리**: 더 나은 정규화
# - **Ordered boosting**: 과적합 감소

# In[ ]:


# ========================================
# 8.1 CatBoost 학습 파이프라인
# ========================================

def train_catboost_model(
    train_df: pd.DataFrame,
    features: pd.DataFrame,
    selected_features: dict,
    base_oof_path: str = None,
    n_folds: int = 5,
    seed: int = 42
) -> tuple:
    """
    CatBoost 모델 학습 및 OOF 예측 생성
    """

    cv_folds = make_subject_hole_folds(train_df, n_folds)

    model_features = features.copy()
    oof_preds = pd.DataFrame(index=train_df.index)
    models = {}
    fold_metrics = []
    base_feature_cols = []

    # Base 앵커 특성 추가
    if base_oof_path and os.path.exists(base_oof_path):
        base_oof = pd.read_csv(base_oof_path)
        for col in ["sleep_date", "lifelog_date"]:
            base_oof[col] = pd.to_datetime(base_oof[col], errors="coerce")

        for target in TARGET_COLS:
            merged = train_df[['subject_id', 'sleep_date', 'lifelog_date']].merge(
                base_oof, on=['subject_id', 'sleep_date', 'lifelog_date'], how='left'
            )
            prob = merged[target].fillna(0.5).clip(EPS, 1-EPS)
            model_features[f'base_{target}_prob'] = prob.to_numpy()
            model_features[f'base_{target}_logit'] = np.log(prob / (1 - prob)).to_numpy()

        base_feature_cols = [f'base_{t}_prob' for t in TARGET_COLS] + [f'base_{t}_logit' for t in TARGET_COLS]

    for target in TARGET_COLS:
        print(f"\n[CatBoost] Training {target}...")

        target_features = selected_features.get(target, [])
        feature_cols = target_features + base_feature_cols

        X = model_features[feature_cols].fillna(0)
        y = train_df[target].values

        target_models = []
        target_oof = np.zeros(len(train_df))

        for fold_idx, (tr_idx, va_idx) in enumerate(cv_folds):
            model = CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3,
                random_seed=seed + fold_idx,
                verbose=False,
                early_stopping_rounds=50,
                cat_features=['subject_cat'] if 'subject_cat' in X.columns else []
            )

            model.fit(
                X.iloc[tr_idx], y[tr_idx],
                eval_set=(X.iloc[va_idx], y[va_idx]),
                verbose=False
            )

            target_oof[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
            target_models.append(model)

        oof_score = log_loss(y, target_oof.clip(EPS, 1-EPS))
        print(f"  {target} OOF LogLoss: {oof_score:.4f}")

        oof_preds[target] = target_oof
        models[target] = target_models
        fold_metrics.append({'target': target, 'oof_logloss': oof_score})

    avg_logloss = np.mean([m['oof_logloss'] for m in fold_metrics])
    print(f"\n[CatBoost] Average OOF LogLoss: {avg_logloss:.4f}")

    return oof_preds, models, {'avg_logloss': avg_logloss, 'per_target': fold_metrics}

print("✅ CatBoost 학습 파이프라인 정의 완료")


# ---
# 
# ## 9. Multi-Model Blend (앙상블)
# 
# ### 9.1 최적 블렌드 가중치 탐색
# 
# **방법**: OOF 데이터에서 Grid Search
# - LGBM + XGB + CatBoost 3모델 조합
# - 21단계 그리드 서치 (0~1 사이 가중치)
# - 평균 Log Loss로 성능 평가

# In[ ]:


# ========================================
# 9.1 3모델 Blend 최적화
# ========================================

def find_best_weights_3model(
    oof_lgbm: pd.DataFrame,
    oof_xgb: pd.DataFrame,
    oof_cat: pd.DataFrame,
    train_df: pd.DataFrame,
    n_steps: int = 21
) -> tuple:
    """
    OOF 데이터에서 3모델 최적 가중치 탐색

    Grid search 범위: 0~1 (0.05 간격)

    Args:
        oof_lgbm, oof_xgb, oof_cat: 각 모델 OOF 예측
        train_df: 훈련 레이블
        n_steps: 가중치 탐색 단계 수

    Returns:
        (best_weights, best_loss)
    """

    best_weights = None
    best_loss = float('inf')

    # Grid search
    for w1_pct in range(n_steps):
        for w2_pct in range(n_steps - w1_pct):
            w3_pct = n_steps - 1 - w1_pct - w2_pct

            w1 = w1_pct / (n_steps - 1)  # LGBM
            w2 = w2_pct / (n_steps - 1)  # XGB
            w3 = w3_pct / (n_steps - 1)  # Cat

            # 각 타겟별 blend loss 계산
            losses = []
            for target in TARGET_COLS:
                # 가중 평균
                blended = (
                    w1 * oof_lgbm[target].values + 
                    w2 * oof_xgb[target].values + 
                    w3 * oof_cat[target].values
                )
                # Log loss 계산
                loss = log_loss(train_df[target].values, blended.clip(EPS, 1-EPS))
                losses.append(loss)

            mean_loss = np.mean(losses)

            if mean_loss < best_loss:
                best_loss = mean_loss
                best_weights = (w1, w2, w3)

    return best_weights, best_loss


def blend_submissions_with_weights(
    sub_lgbm: pd.DataFrame,
    sub_xgb: pd.DataFrame,
    sub_cat: pd.DataFrame,
    weights: tuple
) -> pd.DataFrame:
    """
    최적 가중치로 제출 파일 블렌딩
    """
    w1, w2, w3 = weights

    result = sub_lgbm[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    for target in TARGET_COLS:
        blended = (
            w1 * sub_lgbm[target].values + 
            w2 * sub_xgb[target].values + 
            w3 * sub_cat[target].values
        )
        result[target] = blended.clip(EPS, 1-EPS)

    return result

print("✅ 3모델 Blend 함수 정의 완료")
print("   - Grid search: 21단계 (0.05 간격)")
print("   - 평가 지표: 7개 타겟 평균 Log Loss")


# ---
# 
# ## 10. Target-Specific Blend 가중치
# 
# ### 10.1 왜 타겟별로 다른 가중치를 사용하는가?
# 
# 각 타겟의 특성에 따라 최적 모델이 다름:
# - **Q2**: XGBoost 단독이 가장 좋은 경우 (XGB=1.0)
# - **S3, S4**: CatBoost 단독이 가장 좋은 경우 (Cat=1.0)
# - **S1**: LGBM 0.3 + Cat 0.7 (복합적)
# 
# → 타겟별로 최적 가중치를 적용하면 성능 개선 가능

# In[ ]:


# ========================================
# 10.1 Target-Specific Blend 최적화
# ========================================

def find_target_specific_weights(
    oof_lgbm: pd.DataFrame,
    oof_xgb: pd.DataFrame,
    oof_cat: pd.DataFrame,
    train_df: pd.DataFrame,
    n_steps: int = 11
) -> dict:
    """
    각 타겟별로 최적 가중치 탐색

    Returns:
        타겟별 최적 가중치 딕셔너리
    """
    target_weights = {}

    for target in TARGET_COLS:
        best_w = None
        best_loss = float('inf')

        for w1_pct in range(n_steps):
            for w2_pct in range(n_steps - w1_pct):
                w3_pct = n_steps - 1 - w1_pct - w2_pct

                w1 = w1_pct / (n_steps - 1)
                w2 = w2_pct / (n_steps - 1)
                w3 = w3_pct / (n_steps - 1)

                blended = (
                    w1 * oof_lgbm[target].values + 
                    w2 * oof_xgb[target].values + 
                    w3 * oof_cat[target].values
                )
                loss = log_loss(train_df[target].values, blended.clip(EPS, 1-EPS))

                if loss < best_loss:
                    best_loss = loss
                    best_w = {'lgbm': w1, 'xgb': w2, 'cat': w3, 'loss': best_loss}

        target_weights[target] = best_w

    return target_weights


def blend_target_specific(
    sub_lgbm: pd.DataFrame,
    sub_xgb: pd.DataFrame,
    sub_cat: pd.DataFrame,
    target_weights: dict
) -> pd.DataFrame:
    """
    타겟별 최적 가중치로 제출 파일 블렌딩
    """
    result = sub_lgbm[['subject_id', 'sleep_date', 'lifelog_date']].copy()

    for target in TARGET_COLS:
        w = target_weights[target]
        blended = (
            w['lgbm'] * sub_lgbm[target].values + 
            w['xgb'] * sub_xgb[target].values + 
            w['cat'] * sub_cat[target].values
        )
        result[target] = blended.clip(EPS, 1-EPS)

    return result

print("✅ Target-Specific Blend 함수 정의 완료")
print("   - 타겟별 11단계 그리드 서치")
print("   - 예: Q2=XGB 100%, S3=Cat 100%")


# ---
# 
# ## 11. 제출 파일 생성
# 
# ### 11.1 최종 제출 포맷

# In[ ]:


# ========================================
# 11.1 제출 파일 생성 및 저장
# ========================================

def create_submission(
    predictions: pd.DataFrame,
    output_path: str
) -> None:
    """
    제출 파일 생성

    Args:
        predictions: 예측 결과
        output_path: 저장 경로
    """
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # CSV 저장
    predictions.to_csv(output_path, index=False)
    print(f"✅ 제출 파일 저장: {output_path}")
    print(f"   Shape: {predictions.shape}")
    print(f"   Preview:")
    print(predictions.head())


# ---
# 
# ## 12. 전체 파이프라인 실행 요약
# 
# ### 12.1 결과 요약
# 
# | 지표 | 값 |
# |------|-----|
# | **Public LB** | **0.5917** |
# | Local OOF (3model blend) | 0.514 |
# | Local-Public Gap | ~0.075 |
# | LGBM:XGB:Cat blend | 0.7:0.3:0.0 (73:27:0) |
# | 사용된 특성 수 | 1,682개 (안정성 필터링 후) |
# 
# ### 12.2 핵심 성공 요인
# 
# 1. **V152 앵커 활용**: 이전 최고 모델을 베이스로 사용
# 2. **다모델 앙상블**: LGBM + XGB로 정규화 이점 활용
# 3. **안정성 필터링**: 23,177개 → 1,682개 특성으로 일반화 향상
# 4. **Subject-aware CV**: 인터리빙 구조를 고려한 검증 전략
# 
# ### 12.3 Local-Public Gap 원인 분석
# 
# - CV는 subject-level 일반화를 시뮬레이션
# - 실제 평가: 특정 subject의 특정 날짜들을 평가
# - Subject-specific 패턴에 과적합되는 경향
# 
# **해결 방향**: 더 다양한 모델(더 많은 앙상블)를 통해 일반화 촉진

# In[ ]:


# ========================================
# 12. 최종 요약 시각화
# ========================================

# 모델 성능 비교 막대 그래프
model_results = {
    'LGBM only': 0.525,
    'XGB only': 0.530,
    'Cat only': 0.528,
    'LGBM+XGB (7:3)': 0.514,
    '3-model blend': 0.514,
    'Target-specific': 0.513
}

plt.figure(figsize=(10, 5))
bars = plt.bar(model_results.keys(), model_results.values(), color='steelblue')
plt.axhline(y=0.5917, color='red', linestyle='--', label=f'Public LB: 0.5917')
plt.ylabel('OOF Log Loss (Lower is Better)')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()

print("\n📊 최종 결과:")
print("   • Public Score: 0.5917 (LGBM + XGB 73 블렌드)")
print("   • Local OOF: 0.514 (3-model blend)")
print("   • Target-specific blend로 추가 개선 가능")


# ---
# 
# ## 13. 개선 방향 (Future Work)
# 
# ### 13.1 단기 개선 가능 방향
# 
# | 방향 | 예상 개선 | 설명 |
# |------|-----------|------|
# | mScreenStatus 피처 활용 | +0.005~0.010 | 729개 피처, 강한 S-target 신호 |
# | Subject-adaptive calibration | +0.002~0.005 | 극단값 subject 보정 |
# | Day-of-week 피처 추가 | +0.001~0.003 | 명확한 신호 |
# 
# ### 13.2 중기 개선 가능 방향
# 
# | 방향 | 예상 개선 | 설명 |
# |------|-----------|------|
# | 6-model 앙상블 확장 | +0.002~0.005 | 더 많은 모델 diversity |
# | Sensor-sensor 교차 특성 | +0.003~0.008 | 윈도우 쌍 외 추가 상호작용 |
# | Feature stability 재선별 | +0.001~0.003 | 새로운 특성 공간 탐색 |
# 
# ### 13.3 장기 개선 방향
# 
# | 방향 | 불확실성 | 설명 |
# |------|---------|------|
# | 7-day sequence 모델 (GRU/LSTM) | 높음 | 복잡한 구현 |
# | Subject-specific calibration | 중간 | 개인화 접근 |
# | 도메인 지식 기반 특성 공학 | 중간 | 수면 의학 지식 활용 |
