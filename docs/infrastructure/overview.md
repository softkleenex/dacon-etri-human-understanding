# 인프라 명세 및 도구 활용 전략 (Infrastructure & Tools Setup)

경진대회 기간 동안 로컬 환경과 클라우드 환경을 유연하게 결합하여 최고의 효율을 내기 위한 MLOps 파이프라인 개요입니다. 각 도구들은 상호 연동되어 에이전트 기반 자율 진화 워크플로우를 완벽하게 보조합니다.

세부 가이드는 아래 링크를 통해 확인하세요.

## 🖥 환경 분리 가이드 (Environments)
작업의 성격에 따라 컴퓨팅 리소스를 분리합니다.
- 👉 **[로컬 환경 (Apple Silicon) 가이드](local_env/guide.md)**: 실시간 코드 디버깅, EDA 및 가벼운 모델 실험.
- 👉 **[클라우드 환경 (Kaggle Notebooks) 가이드](cloud_env/guide.md)**: 대용량 딥러닝 훈련 및 병렬 교차 검증(K-Fold).

## 🛠 도구별 활용 가이드 (Tools & Libraries)
본격적인 대회 참여 시 무기가 될 기술 스택 명세입니다.
- 🧠 **[Gemini CLI (AI Agent) 가이드](gemini/guide.md)**: 프로젝트 코드베이스를 진화시키는 컨트롤 타워.
- 📊 **[Weights & Biases (W&B) 가이드](wandb/guide.md)**: 백그라운드 모델 학습 로깅 및 튜닝 대시보드 (전역 토큰 연동 완료).
- ⚙️ **[Kaggle CLI 가이드](kaggle/guide.md)**: 클라우드 파이프라인 연결 및 제출 자동화.
- 🎯 **[Optuna 튜닝 가이드](optuna/guide.md)**: 베이지안 최적화 기반 하이퍼파라미터 탐색.
- 🤗 **[Hugging Face 가이드](huggingface/guide.md)**: 최신 AI 논문/사전학습 가중치 로드 (전역 토큰 연동 완료).
- ⚡ **[데이터 전처리 및 MLOps 가이드](data_processing/guide.md)**: Polars, PyTorch Lightning, DVC 등을 활용한 한계 돌파.
- 📦 **[AI Hub 가이드](aihub/guide.md)**: 보조 데이터 파이프라인.

## 🌐 데이터 API
- 👉 **[DACON API 가이드](daconapi/daconapi.txt)**: 대회 데이터 로드 관련 참고 규약.
