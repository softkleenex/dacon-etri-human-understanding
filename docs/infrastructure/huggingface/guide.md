# Hugging Face Hub 가이드

시계열, 궤적 최신 연구 논문과 사전 학습된(Pre-trained) 모델 가중치를 프로젝트에 도입하기 위한 툴 체인입니다.

## 상태: [적용 완료]

## 도구 구성
- `huggingface_hub` Python 라이브러리
- Hugging Face CLI (`hf`)
- Hugging Face MCP Tools (에이전트용)

## 활용 목적
- **Paper Search**: 모기 궤적 예측과 유사한 시계열 물리(Physics-informed) 논문 및 최신 구조 탐색.
- **Pretrained Models**: Time-Series Transformer 등 공개된 기본 시계열 모델을 다운로드하여 Transfer Learning 진행.
- **연동 자동화**: Kaggle 노트북 환경 주입 시 `HF_TOKEN`을 통해 Private 모델 접근 허용.
