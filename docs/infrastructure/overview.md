# 인프라 명세 및 도구 활용 전략 (Infrastructure & Tools Setup)

경진대회 기간 동안 로컬 환경과 클라우드 환경을 분리하여 효율적으로 리소스를 활용하며, 여러 도구를 연동하여 자동화된 MLOps 환경을 구축했습니다.

자세한 환경 및 도구별 가이드는 아래 링크를 참고하세요.

## 🖥 환경 분리 가이드
- 👉 **[로컬 환경 (Apple Silicon) 가이드](local_env/guide.md)**
- 👉 **[클라우드 환경 (Kaggle Notebooks) 가이드](cloud_env/guide.md)**

## 🛠 도구별 활용 가이드
- 🧠 **[Gemini CLI (AI Agent) 가이드](gemini/guide.md)**: 전체 인프라를 관장하고 자율 진화(Self-Evolving) 워크플로우를 주도하는 메인 컨트롤 타워
- 👉 **[Kaggle CLI 가이드](kaggle/guide.md)**: 노트북 업로드 및 자동 제출 파이프라인
- 👉 **[Weights & Biases 가이드](wandb/guide.md)**: 백그라운드 모델 학습 로깅 및 모니터링
- 👉 **[Optuna 튜닝 가이드](optuna/guide.md)**: K-Fold 기반 베이지안 최적화
- 👉 **[Hugging Face 가이드](huggingface/guide.md)**: 논문/사전학습 모델 탐색 및 로드
- 👉 **[AI Hub CLI 가이드](aihub/guide.md)**: 보조 데이터 파이프라인
- 👉 **[데이터 전처리 및 MLOps 가이드](data_processing/guide.md)**: Polars, tsaug, Geopandas, DVC, PyTorch Lightning 

## 🌐 데이터 API
- 👉 **[DACON API 가이드](daconapi/daconapi.txt)**
