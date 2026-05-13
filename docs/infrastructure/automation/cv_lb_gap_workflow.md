# 🕵️‍♂️ CV-LB Gap Monitoring & Resolution Workflow

DACON 대회나 캐글에서 로컬 교차검증(CV) 점수와 Public Leaderboard(LB) 점수 간에 격차가 벌어지는 현상(CV-LB Gap)은 모델이 과적합(Overfitting)되었거나 검증 전략에 데이터 누수(Leakage)가 있음을 의미합니다.

현재 우리의 CV-LB 상태:
- LSTM + CatBoost: CV `0.6277` -> LB `0.6360` (Gap: `0.0083`)
- Optuna CNN-Transformer + CatBoost 45-Model: CV `0.5925` -> LB `0.6116` (Gap: `0.0191`)

점수가 0.611로 올랐지만, 갭이 커진 것은 명백한 **"앙상블/검증 과적합 신호"**입니다. 이를 해결하기 위한 워크플로우를 정의합니다.

## 🚨 핵심 원인 분석 (Root Cause Analysis)

1. **검증 세트 누수 (Validation Leakage)**
   - **문제**: `Subject-Hole CV`는 유저 타임라인 중간에 구멍(Hole)을 뚫습니다. 하지만 실제 테스트 셋은 항상 "미래"의 데이터입니다. 구멍을 뚫으면 양쪽 끝(과거와 미래) 데이터를 보고 중간을 맞추는 **Look-ahead Bias(미래 참조 누수)**가 발생합니다.
   - **해결책**: `make_subject_timeseries_folds` 함수를 도입하여, 유저별로 엄격한 시간 기반 분할(`TimeSeriesSplit`)을 수행합니다. 과거 데이터로 미래를 예측하는 훈련만 허용합니다.

2. **통계적 결측치 보간 누수 (Statistical Leakage)**
   - **문제**: `handle_missing_values`에서 `.interpolate()`를 `subject_id` 그룹화 없이 전체 데이터프레임에 적용했습니다. 다른 유저의 데이터나, 검증 셋의 미래 데이터가 보간에 섞여 들어갑니다.
   - **해결책**: 반드시 `df.groupby('subject_id')[numeric_cols].transform(lambda x: x.interpolate(limit_direction='forward'))` 구조를 사용하여 철저히 격리합니다.

3. **앙상블 가중치 과적합 (Ensemble Overfitting)**
   - **문제**: `optimize_ensemble.py`가 타겟별로 45개의 모델 중 최적의 가중치를 찾기 위해 OOF 전체를 보고 `scipy.optimize`를 돌립니다. 이는 Holdout 세트 없이 검증 데이터 자체에 가중치를 과적합시키는 전형적인 오류입니다.
   - **해결책**: 타겟별 최적화의 자유도(Degrees of freedom)를 낮추거나, 단순 평균(Simple Average) 점수를 함께 추적하여 실제 모델 성능 향상인지 노이즈 피팅인지 모니터링합니다.

## 🛠 해결 워크플로우 (Resolution Steps)

1. **데이터 전처리 롤백 및 격리**: `build_features.py`의 보간 및 통계 처리 로직에 `groupby('subject_id')`가 제대로 걸려 있는지 2중 체크합니다.
2. **검증 전략 롤백**: 모든 훈련 스크립트(`dl_trainer.py`, `train_simple_ml.py`)의 분할 방식을 `make_subject_timeseries_folds`로 교체합니다.
3. **가짜 점수 경계**: 로컬 CV 점수가 너무 낮아지면(예: 0.55 이하), 반드시 OOF 예측이 훈련 데이터에 누수된 것이 아닌지 의심합니다.
4. **LB 리포트 대조**: 모델 파이프라인 변경 후 LB 점수를 제출하고, 갭이 줄어드는 방향(예: 0.008 이하 유지)으로 최적화 방향을 수정합니다.
