# 인프라 명세 및 도구 활용 전략 (Infrastructure & Tools Setup)

본 템플릿은 DACON(및 Kaggle) 경진대회에서 빠르고 안정적으로 최고 수준의 성능을 달성하기 위해, 로컬 머신과 클라우드 환경을 유연하게 결합한 **MLOps 파이프라인 개요서**입니다.

이 파이프라인의 핵심은 **"자원의 분산"**과 **"에이전트 주도 자율화"**입니다. 모델 탐색과 데이터 전처리는 로컬에서, 무거운 딥러닝 훈련과 하이퍼파라미터 튜닝은 클라우드 백그라운드에서 실행하며, 이 모든 과정을 AI 에이전트(Gemini CLI)가 조율합니다.

자세한 도구별 세부 설정 및 실무 활용 코드는 아래의 개별 가이드를 확인하세요.

---

## 🖥 1. 환경 분리 전략 (Environments)
작업의 성격과 필요한 컴퓨팅 리소스의 크기에 따라 환경을 명확히 분리하여 병목을 없앱니다.
- 👉 **[로컬 환경 (Apple Silicon) 가이드](local_env/guide.md)**
  - 대상 작업: 실시간 코드 디버깅, EDA, 데이터 전처리 스크립트 검증, 트리 기반 모델(LightGBM) 훈련.
  - 특징: 즉각적인 피드백 루프, MPS(Metal Performance Shaders) 가속.
- 👉 **[클라우드 환경 (Kaggle Notebooks) 가이드](cloud_env/guide.md)**
  - 대상 작업: 대용량 딥러닝 모델 훈련, K-Fold 병렬 교차 검증.
  - 특징: Kaggle CLI를 통한 무인 백그라운드 워커 배포, GPU/TPU 자원 활용.

---

## 🛠 2. 도구별 실무 활용 가이드 (Tools & Libraries)
본격적인 대회 참여 시 무기가 될 기술 스택 명세입니다.

### 🧠 컨트롤 타워
- 👉 **[Gemini CLI (AI Agent) 가이드](gemini/guide.md)**
  - 역할: 프로젝트 코드베이스를 이해하고 진화시키는 메인 에이전트. 논문 탐색부터 코드 수정, 훈련 실행까지 책임집니다.

### ⚙️ 학습 및 최적화 파이프라인
- 👉 **[Kaggle CLI 가이드](kaggle/guide.md)**
  - 역할: 로컬에서 작성한 코드를 클라우드에 밀어 넣고(push), 훈련이 끝난 결과물(Submission, Weights)을 받아옵니다(pull).
- 👉 **[Weights & Biases (W&B) 가이드](wandb/guide.md)**
  - 역할: 클라우드에서 비동기로 도는 수많은 모델의 손실(Loss)과 지표를 실시간으로 모니터링합니다. (전역 토큰 자동 연동)
- 👉 **[Optuna 튜닝 가이드](optuna/guide.md)**
  - 역할: 베이지안 최적화를 활용하여 제한된 제출 횟수 내에서 가장 효율적인 하이퍼파라미터 탐색 공간을 찾아냅니다.

### 📚 데이터 및 모델 생태계
- 👉 **[Hugging Face 가이드](huggingface/guide.md)**
  - 역할: 최신 시계열, 비전 논문과 사전학습 모델 가중치를 코드에 이식합니다. (전역 토큰 자동 연동)
- 👉 **[AI Hub 가이드](aihub/guide.md)**
  - 역할: 날씨, 지리 공간 등 모델의 강건성을 높여줄 양질의 보조 데이터셋을 확보합니다.
- 👉 **[데이터 전처리 및 MLOps 가이드](data_processing/guide.md)**
  - 역할: Polars를 통한 고속 피처 엔지니어링, PyTorch Lightning을 통한 딥러닝 보일러플레이트 제거, DVC를 통한 데이터/모델 버전 관리.

---

## 🌐 3. 데이터 API 참고
- 👉 **[DACON API 가이드](daconapi/daconapi.txt)**
  - DACON 서버 규약 및 외부 데이터 사용 제약 관련 안내.
