# Hugging Face Hub 가이드

최신 AI 연구 논문, 사전 학습된 모델 가중치(Pre-trained Weights), 데이터셋을 다운로드하거나 공유하기 위한 플랫폼입니다. 자연어 처리뿐만 아니라 시계열(Time-series), 비전(Vision) 등 다양한 도메인의 모델을 로드할 수 있습니다.

## 1. 인증 및 환경 변수 설정
**시스템 전역(`~/.zshrc` 및 `~/.cache/huggingface/token`)에 `HF_TOKEN`이 인증되어 있습니다.** 
Private 레포지토리나 Llama, Gemma 등 라이선스 동의가 필요한 Gated 모델에 접근할 때 권한 문제가 발생하지 않습니다.

- **전역 토큰 변수**: `HF_TOKEN`
- **CLI 도구**: `hf` CLI가 설치되어 있습니다. (기존 `huggingface-cli`의 신형 래퍼)

## 2. CLI 주요 명령어 (`hf`)
터미널에서 직접 모델이나 데이터셋을 검색하고 다운로드할 때 유용합니다.

- **인증 확인**: `hf auth whoami`
- **모델 검색**: `hf models ls --search "time-series"`
- **모델/데이터셋 다운로드**: `hf download [저장소_이름]`
  ```bash
  # 예시: 특정 모델 가중치 다운로드
  hf download mistralai/Mistral-7B-v0.1 --local-dir ./models/mistral
  ```

## 3. Python API 활용 (`huggingface_hub` & `transformers`)
코드 내에서 모델을 동적으로 로드하거나 결과를 업로드할 때 사용합니다.

```python
from huggingface_hub import hf_hub_download
import torch

# 모델 가중치 파일 다운로드
model_path = hf_hub_download(repo_id="facebook/time-series-transformer", filename="pytorch_model.bin")
state_dict = torch.load(model_path)

# API 연동을 통한 데이터셋 다운로드
from datasets import load_dataset
dataset = load_dataset("my-organization/dacon-preprocessed", token=True)
```

## 4. Dacon 대회 적용 전략
- **베이스라인 모델 탐색**: 대회 주제와 유사한 도메인의 최근 모델(예: Time-Series Transformer, Autoformer, Informer 등)을 Hugging Face에서 검색하여 `src/models.py`에 이식합니다.
- **MCP 연동**: Gemini CLI (에이전트)는 내장된 Hugging Face MCP 도구를 통해 최신 논문(`paper_search`)을 스스로 읽고 아키텍처에 반영할 수 있습니다.
