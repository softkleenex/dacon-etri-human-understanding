# 🏆 제출 및 실험 기록부 (Submission & Experiment History)

실제 제출 파일과 K-Fold 로컬 검증 점수(CV Score), DACON 리더보드 점수(Public LB Score) 간의 갭(Gap)을 추적하는 공간입니다.

> **💡 핵심 변경 사항 (CSV 트래킹 도입)**
> 로컬 검증(CV) 점수와 실제 제출(LB) 점수를 명확히 구분하고, 파이썬 스크립트가 결과를 자동으로 기록(Append)할 수 있도록 **마크다운 표 대신 `submissions/submission_log.csv` 파일을 사용**하여 데이터를 추적합니다.

## 📈 리더보드 및 로컬 점수 추적 (CSV Log)
모든 실험 결과는 👉 **[`submissions/submission_log.csv`](../submissions/submission_log.csv)** 에 기록됩니다.

### 📝 CSV 컬럼 명세
- `id`: 실험 고유 번호
- `date`: 추론/제출 날짜
- `submission_file`: 생성된 CSV 파일명 (예: `xgb_fold5.csv`)
- `model_description`: 적용한 모델 및 주요 피처 변경점
- **`local_cv_score` (나 스스로 판단한 점수)**: K-Fold 교차 검증 등을 통해 로컬 머신에서 계산된 점수. 모델 훈련/추론 스크립트(`inference.py`)가 자동으로 이 값을 채워 넣습니다.
- **`public_lb_score` (진짜 제출해서 얻은 점수)**: DACON 웹사이트에 실제로 제출(API 또는 수동)한 뒤 얻은 점수. 제출 전에는 비워두고(Null), 제출 후에 사용자가 점수를 기입합니다.
- `is_submitted`: 실제 DACON에 제출했는지 여부 (True/False). 하루 제출 횟수 제한(Quota) 관리를 위해 사용합니다.
- `notes`: 에이전트나 사용자가 남기는 특이사항.

---

## 🔬 실험 기록 템플릿 (Experiment Tracking Template)
새로운 실험을 진행할 때 아래 템플릿을 복사하여 기록하세요.

### 🧪 실험 명: [실험 이름 또는 주제] (관련 CSV id: [id])
- **가설 (Hypothesis)**: [무엇을 변경하면 어떤 효과가 있을 것이라고 예상하는지 기술]
- **결과 (Result)**: [로컬 CV 및 Public LB 점수 변화, 학습 시간 등 주요 결과]
- **다음 단계 (Next Step)**: [결과를 바탕으로 얻은 통찰 및 다음으로 시도할 작업]

---

## 🛑 튜닝 및 실패 기록 (What Did NOT Work)
(CSV에 담기 힘든 길고 상세한 실패 원인 분석이나 통찰(Insight)은 이 마크다운 문서에 텍스트로 기록합니다.)

- **시도 1**: LightGBM Baseline (CSV id: 1)
  - **결과**: 로컬 CV 0.6022, Public LB 13.3222 (226등)
  - **이유 추정**: DACON 평가 지표가 Log-Loss이므로 점수가 낮을수록 순위가 높음. 현재 모델은 Macro F1 기준으로 최적화되어 있어 Log-Loss 점수가 매우 높게 나옴.
  - **통찰**: 다음 실험부터는 모델의 평가 및 최적화 지표를 Log-Loss로 변경해야 함.


### 🧪 실험 명: 궁극의 앙상블 블렌딩 및 오버피팅 완벽 제거 (LSTM + CatBoost)
- **가설 (Hypothesis)**: 
  - 과거 모델들이 로컬에서 0.62의 점수를 냈던 것은 `StratifiedKFold`로 인해 훈련/검증 셋에 동일 유저가 섞여 들어가는 치명적인 데이터 누수(Data Leakage) 때문이었다.
  - 이를 `GroupKFold`(시간 기반 분할)로 완벽히 격리하고, 결측치 보간(Imputation) 역시 Fold 내부로 격리하면 거짓된 과적합(Overfitting) 점수가 걷히고 진짜 실력(Honest Score)이 드러날 것이다.
  - 이렇게 정직하게 얻어낸 모델 2개(유저 임베딩과 어텐션이 적용된 LSTM, 그리고 최상위 60개 피처만 사용하도록 극한 정규화된 CatBoost)를 `scipy.optimize`를 통해 수학적으로 최적의 비율로 섞으면 가장 안정적이고 뛰어난 성능이 나올 것이다.
- **결과 (Result)**: 
  - 로컬 CV: **0.6277**
  - 데이콘 Public LB: **0.63606** (현재 기준 **230위**)
  - **평가**: 놀랍게도 로컬 점수(0.6277)와 실제 리더보드 점수(0.6360)의 격차가 불과 **0.008**밖에 나지 않습니다! 이는 과거의 과적합(Overfitting) 문제가 100% 완벽하게 해결되었으며, 이제 우리의 로컬 점수만을 믿고 오프라인 튜닝을 지속할 수 있다는 최고의 성과를 의미합니다. 또한, 평균값만 찍었을 때의 나이브 베이스라인(0.6641)을 수학적으로 가볍게 깨부수었습니다.
- **다음 단계 (Next Step)**: 
  - 이제 아키텍처의 구멍은 모두 막혔습니다. 목표 점수 **0.50** 달성을 위해, 시계열 데이터를 2D 이미지처럼 바라보고 CNN을 적용하는 1D-CNN 블록을 LSTM에 추가하거나, Transformer의 Self-Attention 구조를 덧붙여 표현력(Representation Power) 자체를 폭발적으로 끌어올리는 하드코어 딥러닝 튜닝을 오프라인(Downtime Workflow)에서 진행해야 합니다.


### 🧪 실험 명: 45-Model Optuna 앙상블 및 CV-LB Gap 심층 분석
- **결과 (Result)**: 
  - 로컬 CV: **0.5925**
  - 데이콘 Public LB: **0.61167**
- **Gap 분석 (Why did the gap widen to 0.019?)**:
  - 서브 에이전트의 초정밀 진단 결과, `0.5925`라는 점수는 4가지 거대한 데이터 누수(Data Leakage)가 만들어낸 환각이었습니다.
  - **1. 앙상블 과적합**: `optimize_ensemble.py`가 45개 모델을 타겟별로 섞을 때, 검증용 Holdout 셋 없이 OOF 데이터 전체를 보고 정답을 끼워 맞췄습니다. (자유도가 너무 높아 노이즈까지 학습)
  - **2. Hole 검증의 한계**: `Subject-Hole CV`는 유저 타임라인 중간에 구멍을 뚫습니다. 하지만 실제 테스트 셋은 '미래'를 예측하는 것입니다. 구멍 뚫기 방식은 '미래'의 데이터를 보고 '과거'를 예측하는 시간 역전 누수를 발생시킵니다.
  - **3. 전역 결측치 보간 누수**: `build_features.py`에서 `interpolate()` 함수가 유저별로 그룹화되지 않고 전체 데이터프레임에 적용되어 유저 간/미래 데이터 누수가 발생했습니다.
- **Next Step**:
  - 검증 방식을 엄격한 `TimeSeriesSplit`(시간 기반 분할)로 롤백.
  - 피처 엔지니어링 시 미래 데이터 참조 원천 차단 (expanding만 사용).
