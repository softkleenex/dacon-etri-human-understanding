# Hugging Face Hub 활용 매뉴얼

처음부터 모델 구조를 짜는 대신, 글로벌 SOTA(State-of-the-Art) 모델 아키텍처와 사전학습 가중치를 내 코드에 신속하게 수혈하여 대회 성능을 극대화하기 위해 도입한 플랫폼입니다.

## 🎯 사용 목적 (Why use this?)
- **사전학습 가중치 획득**: 방대한 데이터로 미리 학습된 LLM이나 시계열 파운데이션 모델을 다운로드하여, 대회 데이터로 파인튜닝(Fine-tuning)함으로써 압도적 우위를 점합니다.
- **최신 논문 이식**: 에이전트(Gemini)를 통해 다변량 시계열 등 최신 논문의 공식 구현 코드를 검색하고 프로젝트 뼈대에 즉각 이식합니다.

## 🚀 사용 방법 (How to use?)

### 1. 인증 및 전역 환경 변수 설정
**이 템플릿 환경에서는 시스템 전역(`~/.zshrc` 및 `~/.cache/huggingface/token`)에 `HF_TOKEN` 인증 토큰이 미리 저장(Set Globally)되어 있습니다.**
Gated 모델(라이선스 동의 필수)이나 Private 저장소 접근 시 권한 오류가 나지 않습니다. 파이썬 스크립트 내에 토큰을 절대 하드코딩하지 마세요.

```bash
# 전역 설정 및 CLI 로그인 상태가 정상인지 확인하는 명령어
hf auth whoami
```

### 2. CLI 고속 다운로드 스니펫 (`hf`)
터미널에서 모델의 구조를 파악하거나 필요한 파라미터 가중치를 내 로컬 폴더로 미리 당겨올 때 사용합니다.
```bash
# 특정 모델 가중치를 로컬 ./models/pretrained_ts 폴더로 고속 다운로드
hf download facebook/time-series-transformer-tourism-monthly --local-dir ./models/pretrained_ts
```

### 3. 실전 파이썬 동적 로드 스니펫
코드 실행 중에 캐시 서버를 통해 가중치를 동적으로 읽어오고 싶을 때 사용합니다. 전역 토큰이 세팅되어 있어 별도의 인증 로직이 필요 없습니다.

```python
from huggingface_hub import hf_hub_download
import torch

# 모델 가중치를 캐시로 다운로드 후 경로 반환
weight_path = hf_hub_download(
    repo_id="facebook/time-series-transformer-tourism-monthly", 
    filename="pytorch_model.bin"
)

# 내 PyTorch 커스텀 모델에 외부 가중치 덧씌우기
my_model = CustomTimeSeriesTransformer()
my_model.load_state_dict(torch.load(weight_path), strict=False)
```

## ⚠️ 실전 주의 사항
- **라이선스 동의 (Gated Models)**: Llama, Gemma 등의 모델을 다운로드하다가 접근 거부(401 Unauthorized) 에러가 발생한다면 토큰 문제가 아닙니다. Hugging Face 웹사이트에서 해당 모델 페이지에 들어가 **"사용 조건 동의(Accept License)"** 버튼을 클릭해야만 전역 토큰으로도 다운로드가 가능해집니다.
