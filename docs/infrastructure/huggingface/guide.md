# Hugging Face Hub 활용 매뉴얼

대회 문제(예: 시계열 예측, 이상 탐지 등)를 풀기 위해 처음부터 모델 구조를 짜는 대신, 글로벌 탑급 연구원들이 이미 만들어둔 SOTA(State-of-the-Art) 모델 아키텍처와 가중치를 코드에 수혈하기 위해 사용합니다. (전역 토큰 로그인은 완료되어 있습니다.)

## 🎯 사용 목적 (Why use this?)
- **사전학습 가중치 획득**: 방대한 데이터로 미리 학습된 언어 모델(LLM)이나 시계열 파운데이션 모델(Time-Series Foundation Model)을 다운로드하여, 내 대회 데이터에 맞게 조금만 튜닝(Fine-tuning)함으로써 압도적인 성능 우위를 점합니다.
- **최신 논문 및 아키텍처 수혈**: Gemini CLI(AI 에이전트)를 통해 "최근 1년 내 발표된 우수한 다변량 시계열 논문을 검색해줘"라고 지시한 뒤, 그 논문의 공식 코드를 바로 가져와 프로젝트에 이식합니다.

## 🚀 사용 방법 (How to use?)

### 1. CLI로 빠르게 모델/가중치 다운로드
터미널에서 필요한 모델 가중치 파일(`.bin`, `.safetensors`)을 내 프로젝트의 `models/` 폴더로 고속 다운로드합니다.
```bash
# Hugging Face에 등록된 시계열 모델 검색
hf models ls --search "time-series forecasting"

# 선택한 모델을 로컬 특정 폴더로 다운로드
hf download facebook/time-series-transformer-tourism-monthly --local-dir ./models/pretrained_ts
```

### 2. 코드 레벨에서 동적 로드 (Python)
스크립트 안에서 `huggingface_hub` 라이브러리를 사용하면 캐시 시스템을 활용해 코드가 실행될 때 알아서 로드되게 할 수 있습니다.

```python
from huggingface_hub import hf_hub_download
import torch

# 모델 가중치를 로컬 캐시로 다운로드하고 경로 반환
weight_path = hf_hub_download(
    repo_id="facebook/time-series-transformer-tourism-monthly", 
    filename="pytorch_model.bin"
)

# 내 커스텀 PyTorch 모델 뼈대에 가중치 덧씌우기
my_model = CustomTimeSeriesTransformer()
my_model.load_state_dict(torch.load(weight_path), strict=False)
```

## ⚠️ 실전 주의 사항
모델을 다운로드할 때는 라이선스(License) 제약을 반드시 확인하세요. Llama 계열이나 특정 Gated 모델들은 웹사이트에서 "사용 조건 동의(Accept License)" 버튼을 눌러야만 API 토큰을 통해 다운로드가 가능해집니다. 에러가 난다면 웹에서 동의 버튼을 눌렀는지 체크하세요.
