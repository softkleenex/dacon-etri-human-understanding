# Hugging Face Hub 가이드

자연어 처리(NLP)를 넘어 시계열(Time-series), 비전(Vision), 멀티모달 최신 연구 논문과 사전 학습된(Pre-trained) 모델 가중치를 프로젝트에 신속하게 주입하기 위한 MLOps 툴 체인입니다.

---

## 1. 인증 및 환경 변수 설정
**시스템 전역(`~/.zshrc` 및 `~/.cache/huggingface/token`)에 `HF_TOKEN`이 안전하게 연동되어 있습니다.** 
따라서 별도로 소스 코드 내에 토큰을 기입할 필요 없이, Llama, Gemma, Time-Series Transformer 등 권한이 필요한 Gated 모델이나 Private 레포지토리에 자유롭게 접근 가능합니다.

- **CLI 기반 인증 확인**: `hf auth whoami` (터미널에서 정상 로그인 여부 확인 가능)

---

## 2. CLI 주요 명령어 (`hf`)
파이썬 스크립트 작성 전, 터미널에서 즉각적으로 모델 구조나 데이터셋을 검색하고 로컬에 다운로드해둘 때 매우 유용합니다.

```bash
# 대회 도메인과 관련된 최신 모델 검색
hf models ls --search "time-series forecasting"

# 특정 모델의 가중치 파일들만 로컬 폴더로 초고속 다운로드
hf download facebook/time-series-transformer-tourism-monthly --local-dir ./models/ts_transformer
```

---

## 3. Python API 활용 (`huggingface_hub` & `transformers`)
코드 내에서 모델을 동적으로 로드하거나 커스텀 전처리된 데이터셋을 캐싱하여 사용할 때 강력합니다.

```python
from huggingface_hub import hf_hub_download
import torch

# 1. 모델 가중치 파일만 동적으로 다운로드 및 로드 (Cache 활용)
model_path = hf_hub_download(
    repo_id="facebook/time-series-transformer-tourism-monthly", 
    filename="pytorch_model.bin"
)
state_dict = torch.load(model_path)
my_custom_model.load_state_dict(state_dict, strict=False)

# 2. 비공개 데이터셋 로드 시 전역 토큰 사용 
from datasets import load_dataset
# 토큰 환경변수가 자동으로 인식되므로 token=True 만 선언하면 됩니다.
dataset = load_dataset("softkleenex/dacon-private-data", token=True) 
```

---

## 4. Dacon 대회 적용 전략 (Agent Integration)
- **최신 아키텍처 수혈**: 대회 주제(예: 시계열 예측, 이상 탐지)와 유사한 문제를 푼 최근 SOTA 모델(Informer, Autoformer 등)을 Hugging Face에서 탐색하여 뼈대(`src/models.py`)로 이식합니다. 처음부터 PyTorch 코드를 짤 필요가 없습니다.
- **MCP 연동**: 에이전트(Gemini CLI)는 내장된 Hugging Face 플러그인을 활용하여 최신 논문을 스스로 검색(`paper_search`)하고 초록(Abstract)을 읽어 대회에 맞는 피처 엔지니어링 및 손실 함수 아이디어를 도출할 수 있습니다.
