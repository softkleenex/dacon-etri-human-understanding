# 딥러닝 아키텍처 (Deep Learning) 매뉴얼

이 대회는 단순히 1차원적인 tabular 데이터를 분류하는 것이 아니라, 사용자의 연속된 로그(Time-series Lifelog)를 바탕으로 주관적 설문(Q1~Q3)과 객관적 수면(S1~S4) 7개 타겟을 다중 예측하는 복합적인 문제를 다룹니다. 이를 위해 커스텀 딥러닝 인프라를 구축했습니다.

## 🎯 사용 목적 (Why use this?)
- **Time-Series Sequence 학습**: 단순히 오늘 하루의 통계가 아니라 "최근 일주일간 수면이 부족했다"는 시계열적 뉘앙스를 학습하기 위해 LSTM 기반의 시퀀스 모델이 필요합니다.
- **과적합(Overfitting) 원천 차단**: 450행이라는 극소량의 훈련 데이터에서 수백 개의 피처를 사용할 때 발생하는 차원의 저주를 피하기 위해, 타겟별 분할 없이 7개의 목표를 동시에 학습(Multi-task Learning)하여 내부 레이어가 범용적인 특성(Shared Representation)을 강제로 학습하도록 유도합니다.

## 🚀 아키텍처 상세 (How it works)

### 1. Temporal Attention 기반 MultiTaskLSTM (`src/models/dl_models.py`)
- **Bidirectional LSTM**: 시간의 흐름을 과거->미래, 미래->과거 양방향으로 훑어 문맥을 더 깊게 이해합니다.
- **Temporal Attention**: 시퀀스(예: 7일, 14일) 내에서 "어느 날의 데이터가 오늘의 타겟 예측에 가장 결정적인지"를 모델 스스로 판단하여 가중치를 부여합니다. 패딩된 더미 데이터(Padding)는 마스킹(`padding_mask`)을 통해 어텐션 점수가 부여되지 않도록 완벽히 무시합니다.
- **User Embeddings (`nn.Embedding`)**: 설문 타겟(Q1~Q3)은 유저마다 베이스라인이 천차만별입니다. 유저 ID 자체를 딥러닝 임베딩 공간에 매핑하여, 모델이 유저 고유의 편향(Bias)을 학습하게 합니다.

### 2. DaconDataModule (`src/data/dl_data_module.py`)
PyTorch Lightning 기반의 데이터 로더입니다.
- **MultiTaskSequenceDataset**: 유저별로 데이터를 그룹핑한 뒤 Sliding Window 방식으로 `(Batch, Seq_len, Features)` 형태의 3차원 텐서를 자동 생성합니다.
- 부족한 날짜는 Zero-padding으로 채우며, 어텐션 레이어를 위해 `padding_mask`를 함께 반환합니다.

### 3. Loss Functions & Optimization
- **BCEWithLogitsLoss**: 기본 손실 함수로 사용하여 숫자 연산의 안정성을 높입니다. 시그모이드(Sigmoid)와 Cross-Entropy를 분리하지 않고 결합하여 무한대 페널티 에러(Log-Loss 폭발)를 수학적으로 방지합니다.
- **FocalLoss (선택 사항)**: 매우 불균형한 타겟(예: S3 수면 지연 시간)을 학습할 때, 쉬운 정답은 무시하고 틀리고 있는 어려운 정답에 가중치를 부여(`gamma=2`)하는 옵션도 코드로 구현되어 있습니다.

### 4. DLTrainer (`src/models/dl_trainer.py`)
머신러닝과 완전히 분리된 딥러닝 전용 훈련 루프입니다.
- **Leak-Free Imputation & Scaling**: KNNImputer와 StandardScaler를 `GroupKFold` 훈련 분할 안에서만 `fit()` 하도록 격리하여 통계적 미래 참조 누수를 100% 차단합니다.
- PyTorch Lightning의 `Trainer`를 호출하여 EarlyStopping, 체크포인트 저장, GPU/CPU 스위칭을 자동화합니다.

## ⚠️ 실전 주의 사항
- **P100 GPU 버그**: Kaggle의 P100 GPU 환경에서는 `torch.backends.cudnn.enabled = False`가 없으면 커널이 즉각 폭파됩니다. 이 데이터셋은 크기가 작아 **CPU 훈련을 강력히 권장**하며, `pre_flight_check.py`에서 GPU 활성화 시 에러를 내도록 세팅해 두었습니다.
- **GroupKFold 강제**: 시계열 시퀀스 특성상 훈련 셋과 검증 셋에 동일한 유저가 포함되면 극심한 정보 누수(Memorization)가 발생합니다. 반드시 `GroupKFold`(또는 shuffle=False 형태의 `StratifiedKFold`)를 사용하여 유저 단위로 격리된 교차 검증을 수행해야 합니다.