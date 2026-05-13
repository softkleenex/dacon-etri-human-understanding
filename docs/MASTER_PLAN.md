# 🗺️ Master Plan & Work Instructions

본 문서는 ETRI 휴먼이해 인공지능 논문경진대회(DACON)에서의 모델링 전략, 작업 과정에서의 메모리(Context) 관리 지침, 그리고 활용 가능한 인프라를 명확히 정의하는 마스터 플랜입니다.

---

## 1. 🏗️ Infrastructure Specification (인프라 명세 및 활용 전략)

인프라 구성, 도구 연동 및 로컬/클라우드 환경 활용에 대한 목적과 실전 파이프라인 매뉴얼은 별도의 문서를 참고하세요.
👉 **[인프라 매뉴얼 가이드 (Infrastructure README)](infrastructure/README.md)**

---

## 2. 🚀 Advanced Modeling Strategies (도약을 위한 고도화 전략)

본 대회는 멀티모달 라이프로그 데이터를 활용하여 사용자의 **설문 지표(Q1~Q3)**와 **수면 지표(S1~S4)**를 이진 분류(0 또는 1)하는 **다중 태스크 분류(Multi-task Classification)** 문제입니다.

### 2.1. Feature Engineering (피처 엔지니어링)
- **Time-series Aggregation (시계열 집계):** 1~10분 간격으로 수집된 스마트폰/스마트워치 데이터(mACStatus, mActivity, mGps, wHr, wPedo 등)를 일별(Daily) 또는 특정 시간대별(e.g., 수면 전, 기상 후)로 집계하여 통계적 특성(Mean, Std, Max, Min, Skewness 등)을 추출합니다.
- **Missing Values & Noise Handling (결측치 및 노이즈 처리):** 기기 미착용 및 배터리 방전 등으로 인한 결측치가 다수 존재합니다. 보간법(Interpolation), Forward/Backward Fill을 적용하거나, 결측 여부 자체를 새로운 피처(Missingness Indicator)로 활용합니다.
- **Unstructured/List Data Processing (비정형 데이터 처리):** `mAmbience`(소리 확률), `mUsageStats`(앱 사용 시간)과 같이 리스트/JSON 형태로 주어지는 데이터는 임베딩(Embedding) 처리하거나 상위 N개의 주요 카테고리로 수치화합니다.

### 2.2. Model Architecture (모델 아키텍처)
- **Baseline Models:** CatBoost 기반의 트리 앙상블 모델과 Optuna를 활용한 베이지안 최적화를 적용합니다.
- **Deep Learning Approaches:** 센서 시계열의 순차적 특성을 반영하기 위해 **Temporal Attention이 포함된 MultiTaskLSTM**을 주력으로 사용합니다. 7일 단위의 슬라이딩 윈도우 시퀀스를 통해 시간적 흐름(누적 피로도 등)을 신경망이 직접 학습하도록 구성합니다.
- **Multi-task Learning (멀티태스크 러닝):** Q1~Q3(설문 지표)와 S1~S4(수면 지표)의 상관관계를 반영하기 위해 7개 타겟의 Log-Loss를 동시에 최적화하는 다중 헤드(Multi-head) 구조를 채택합니다.

### 2.3. Data Augmentation & Normalization (데이터 증강 및 정규화)
- **Expanding Window Normalization:** 타겟 레이블이 개인의 평균을 기준으로 이진화된 점을 고려하여 사용자별 정규화를 수행합니다. 이때 미래 데이터 누수(Data Leakage)를 막기 위해 전체 평균이 아닌 **Expanding Window(과거 누적)** 방식만 사용합니다.
- **Advanced Features:** 심박 변이도(HRV - RMSSD), 스마트폰/워치 조도(mLight, wLight), 활동량과 심박수 간의 상호작용(Cross-sensor Interaction) 등 도메인 지식이 반영된 고급 파생 변수를 추출합니다.

### 2.4. Cross-Validation & Loss Functions (검증 및 손실 함수)
- **Cross-Validation:** 훈련 셋과 검증 셋에 동일 유저가 섞여 들어가는 타겟 누수를 완벽히 차단하기 위해 **GroupKFold (group=subject_id)** 를 엄격하게 적용합니다.
- **Custom Metrics:** 무한대 손실 방지를 위한 확률값 클리핑(`np.clip`) 검증 로직을 거치며, 모델의 조기 종료(Early Stopping) 및 최우수 가중치 저장(Checkpointing) 기준을 `val_logloss_mean`으로 맞추어 공식 지표에 직접적으로 최적화합니다.

### 2.5. SOTA Code Share Integration (0.59 달성을 위한 신규 전략)
경쟁자들의 SOTA(State of the Art) 오픈 소스를 분석하여 도출된, 0.50 타겟을 향한 핵심 확장 전략입니다.
1. **Subject-Hole CV**: 단순한 `GroupKFold`(유저 완전 격리)나 `StratifiedKFold`를 넘어, **한 유저의 시계열 타임라인 중간중간에 구멍(Hole)을 뚫어 검증 셋으로 빼내는 방식**을 도입합니다. 실제 테스트 셋이 훈련 셋 중간에 듬성듬성 이빨 빠진 형태로 존재하는 것을 완벽히 시뮬레이션하기 위함입니다.
2. **Target-Specific Blending**: 7개 타겟에 대해 하나의 동일한 앙상블 가중치를 쓰지 않고, 각 타겟별로 가장 강한 모델(예: Q2는 XGBoost 100%, S3는 CatBoost 100%)을 독립적으로 조합하여 Log-Loss를 극단적으로 낮춥니다.
3. **Anchor Stacking (베이스 OOF 피처화)**: 단일로 가장 성능이 뛰어난 모델(예: LSTM)의 OOF 예측 확률과 로짓(Logit) 값을 버리지 않고, 두 번째 스테이지의 트리 모델(LightGBM, XGBoost)의 **입력 피처(Feature)**로 재사용하여 앙상블의 시너지를 폭발시킵니다.
4. **LGBM + XGBoost 앙상블 추가**: 트리 성장 방식이 다른 Leaf-wise(LightGBM)와 Level-wise(XGBoost) 모델을 모두 학습시켜 과적합되는 방향의 다양성(Diversity)을 챙깁니다.

---

## 3. 🤖 Operational Guidelines (에이전트 및 HITL 작업 지침)

장기 실행 모드(Long-running session)에서 컨텍스트 메모리 초과를 방지하고, 기계(Agent)와 사람(User)이 효율적으로 협업하기 위한 규칙입니다.

### 3.1. Human-in-the-Loop (HITL) 피드백 루프
- DACON의 제출 횟수 제한을 고려하여, **로컬 CV (Cross-Validation) 점수가 과거 최고 기록을 경신할 때만** 리더보드 제출을 진행합니다.
- 실제 리더보드 점수는 👉 **[제출 및 실험 기록부 (SUBMISSION_HISTORY.md)](SUBMISSION_HISTORY.md)** 에 기록하여 실험 간 점수 차이(Gap)를 추적합니다.
- 훈련된 모델의 CSV 파일과 1:1로 매칭되는 `_meta.json` 로그를 자동 생성하여 과거 실험 조건(파라미터, 피처)을 완벽히 복원할 수 있도록 추적성을 확보합니다.
- 앙상블 스크립트(`scripts/ensemble.py`)를 통해 생성된 Top-K 모델들의 평균 산출물(Soft Voting)을 활용하여 리더보드 점수의 안정성을 극대화합니다.

### 3.2. Memory Efficiency (컨텍스트 절약)
- 무거운 데이터프레임의 전체 출력이나 긴 에러 로그의 무분별한 출력을 제한합니다.
- 코드를 수정할 때는 전체 재작성이 아닌, 외과수술적인(Surgical) 부분 수정(`replace` 도구 등 활용)을 지향합니다.

### 3.3. Shell Script Driven Workflow (터미널 기반 자동화 워크플로우)
- 터미널에 수동으로 긴 명령어를 복잡하게 치는 대신, **`.sh` 형태의 쉘 스크립트 파일**을 작성하고 실행 권한(`chmod +x`)을 부여하여 에이전트와 유저가 실행(Run and verify)하는 방식을 기본으로 합니다.
- **예시**: 전체 Kaggle 파이프라인(푸시 -> 포그라운드 대기 -> 로컬 저장 -> 로그 확인)은 `scripts/run_pipeline.sh` 스크립트 하나로 실행됩니다. 백그라운드 봇에 의존하여 유저가 진행 상황을 잃어버리는 것을 막기 위함입니다.
- 실행 중 에러가 발생하더라도 에이전트는 로그를 직접 파악하여 스크립트나 코드를 수정하고, 에러 없이 끝까지 동작하는 것을 확인한 뒤에야 작업을 종료(루프 종료)합니다.

### 3.4. Separation of Concerns & Kaggle Workflow (작업 분리 및 클라우드 연동)
- 실험 설정 파일, 스크립트, 마크다운 문서 등 성격이 다른 파일들을 명확히 분리합니다.
- 무거운 딥러닝 연산이나 K-Fold 교차 검증은 클라우드 GPU(Kaggle 커널 등)로 오프로드(Off-load) 합니다.
- **Pre-flight Checks**: `make kaggle-push-code` 명령 실행 시, 로컬 스크립트(`scripts/pre_flight_check.py`)가 선제적으로 가동되어 GPU 쿼터 초과 여부, 경로 누락, 메트릭 설정 오류 등을 자동 검증합니다.
- **Auto Submission Bot**: 클라우드 훈련 노트북이 가동되면 백그라운드 봇(`scripts/auto_wait_submit.py`)을 띄워 훈련 종료를 감시하고, 결과물(CSV)을 자동으로 로컬에 풀링(Pull)한 뒤 DACON API를 통해 즉각 리더보드에 무인 제출합니다.

### 3.5. Downtime & Local Optimization Workflow (제출 제한 시 대응 전략)
데이콘의 일일 제출 횟수(Daily Submission Limit)가 모두 소진되었을 때, 리더보드 피드백 없이도 점수를 끌어올리기 위한 **오프라인/로컬 최적화 전략**입니다. 제출을 기다리는 동안(Downtime) 에이전트와 유저는 아래 작업에 집중합니다.
1. **Local CV Trust Building (로컬 검증 신뢰도 확보)**: 리더보드 점수(Public LB)와 로컬 OOF(Out-of-Fold) 점수 간의 상관관계를 분석합니다. 로컬 CV가 떨어지면 LB도 무조건 떨어지도록 강력한 교차 검증(GroupKFold 등)의 오차율을 점검합니다.
2. **Deep Exploratory Data Analysis (심층 EDA)**: 단순히 모델을 돌리는 것을 넘어, 오답 노트(Error Analysis)를 작성합니다. 모델이 가장 헷갈려하는 타겟(예: S1 vs S2)이나 극단적인 노이즈가 있는 유저(`subject_id`)의 패턴을 시각화 스크립트로 파악합니다.
3. **Feature Importance & SHAP Analysis**: 트리 모델(CatBoost) 기반으로 어떤 피처(예: 심박 변이도, 조도 센서)가 0.5점 달성에 가장 큰 기여를 하고 있는지 추출하여 불필요한 피처를 쳐내는 Feature Selection을 수행합니다.
4. **Architectural Experiments (구조적 실험)**: 제출이 막혀있는 동안 시간이 오래 걸리는 구조적 변화(예: 1D-CNN 도입, Transformer 블록 융합, 임베딩 레이어 추가)를 로컬에서 구현하고 에러 없이 훈련이 돌아가도록 코딩 및 테스트를 미리 완료해 둡니다.

---

## 4. 🏁 Final Goal (최종 목표)

- **DACON 리더보드 순위권 진입 및 목표 점수 달성:** 철저한 로컬 검증과 창의적인 피처 엔지니어링을 바탕으로 최종 리더보드 상위권(Team: **softkleenex**)에 안착하는 것이 궁극적인 목표입니다.
- **초기 베이스라인 (2026-05-11):** Public LB Score 0.6362 (순위 223위)
- **현재 타겟 점수 (Target Score):** Log-Loss **0.5000** 달성. 이를 위해 Kaggle의 압도적인 컴퓨팅 파워를 활용하여 더 복잡한 시계열 모델(RNN, Transformer 등) 및 대규모 하이퍼파라미터 튜닝을 진행할 예정입니다.
