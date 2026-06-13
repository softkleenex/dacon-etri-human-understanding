# 초거대 라이프로그 인프라(ETRI Platform & Hugging Face) 비판적 평가 및 로드맵

## 1. 현재 인프라 활용의 비판적 평가 (Critical Evaluation)

지금까지 우리 파이프라인의 가장 큰 패착은 **"코드 깎기(Feature Engineering & Hyperparameter Tuning)"에만 매몰되어, 수백만 배의 가치를 지닌 "외부 지식(External Knowledge)"을 완전히 무시했다는 점**입니다.

### ❌ ETRI Platform & 공공 데이터 (Contextual Blindness)
우리는 사용자의 수면, 감정, 외출 상태를 예측하면서도 가장 기초적인 **"비가 오는지, 휴일인지"**조차 모델에게 알려주지 않았습니다. 
- 센서(스마트폰/스마트워치) 데이터는 사용자의 '현재 상태'만을 보여줄 뿐, 사용자가 왜 그렇게 행동하는지(환경적 맥락)를 설명하지 못합니다.
- (해결) 방금 전 앙상블 수정 과정에서 `holidays` 라이브러리를 추가하여 주말/휴일 피처를 넣었지만, 기상청(강수량/기온)과 ETRI Platform의 한국인 생활패턴 말뭉치 등 거시적인 맥락 데이터를 추가해야만 완벽한 Concept Drift 방어가 가능합니다.

### ❌ Hugging Face (Foundation Model Neglect)
우리는 450일이라는 극도로 적은 훈련 데이터를 가지고 딥러닝(CNN+Transformer) 모델을 바닥부터(From Scratch) 학습시키려 했습니다. 이는 **"수학 공식 하나만 외우고 수능을 치려는 꼴"**과 같습니다.
- 이미 `Chronos`, `TimeGPT`, `PatchTST` 등 전 세계의 시계열 패턴을 학습한 초거대 파운데이션 모델들이 Hugging Face에 존재합니다.
- (해결) 이들의 가중치(Weights)를 다운로드하여 Feature Extractor로 사용하거나, ETRI 2018-2020 데이터를 통해 우리만의 Domain-Specific Foundation Model을 사전 학습(Pre-train)시켜야 합니다.

---

## 2. ETRI 2018-2020 데이터셋 정밀 분석 및 한계점

`user11-12.7z` 내부의 `README_2020.txt`를 까본 결과, 놀라운 사실을 발견했습니다.

### 데이터 구조
- **Dacon 2024**: `mLight`, `mScreenStatus`, `mUsageStats` (주로 스마트폰의 소프트웨어적/환경적 로깅)
- **ETRI 2020**: `e4Acc`, `e4Bvp`, `e4Eda`, `e4Hr`, `e4Temp` (Empatica E4라는 전문 의료용 손목 밴드와 스마트폰의 물리적 IMU/GPS 센서 중심)

### 치명적 불일치 (Schema Mismatch)
ETRI 2018-2020 데이터는 전문 장비(E4)에 의존하고 있어, 현재 Dacon 대회에서 가장 중요한 스마트폰 사용 기록(`mScreenStatus`, `mUsageStats`)이 존재하지 않습니다. 
즉, **단순히 과거 데이터를 합쳐서(Concat) Dacon 모델을 사전 학습시키는 것은 불가능합니다.** (Feature Space가 다름)

---

## 3. 궁극의 인프라 확장 마스터플랜 (Master Plan)

그렇다면 이 50GB짜리 황금 광맥(ETRI)과 Hugging Face를 어떻게 엮어내야 할까요?

### Phase 1: Hugging Face Zero-Shot Feature 연동 (즉각 적용)
ETRI 데이터를 가공하는 데는 며칠이 걸릴 수 있습니다. 그 전에, Dacon의 450일 데이터만으로도 압도적 성능을 내기 위해 Hugging Face의 **Pre-trained Time Series Model**을 즉시 도입합니다.
- `Chronos` 또는 `PatchTST`를 `dl_trainer.py`에 임포트.
- 시계열 데이터를 이 모델에 통과시켜 나온 임베딩 벡터(Embedding Vector)를 CatBoost와 LightGBM의 피처(Feature)로 꽂아 넣습니다.

### Phase 2: ETRI 데이터 기반 "Self-Supervised" 사전 학습
스키마가 달라도 상관없는 **자기 지도 학습(Self-Supervised Learning)**을 구축합니다.
- ETRI의 E4 심박수(`e4Hr`), 온도(`e4Temp`), 가속도(`e4Acc`)만 뽑아서, "일부 시간을 마스킹(Masking)하고 이를 복원하는" 오토인코더(Masked Autoencoder) 모델을 설계합니다.
- 이 모델을 50GB ETRI 데이터로 캐글 GPU에서 며칠간 사전 학습시킵니다.
- 이렇게 학습된 뼈대(Backbone)는 인간의 생체 리듬 자체를 완벽하게 이해하게 됩니다.
- 마지막으로 이 뼈대 위에 Dacon의 `mScreenStatus` 같은 추가 피처 레이어를 덧대어 파인튜닝(Fine-Tuning)합니다.

### Phase 3: 날씨/환경 맥락 병합
- `src/features/build_features.py`에 기상청 API 연동 스크립트를 추가하여, 사용자의 수면/활동 데이터와 해당 날짜의 비/눈/기온 데이터를 병합(Merge)합니다.

이러한 다각적 인프라 전략만이 우리 모델을 0.61의 늪에서 구출하여 우승권(0.50 이하)으로 이끌 유일한 해답입니다.