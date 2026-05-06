# 로컬 환경 (Apple Silicon) 활용 매뉴얼

이 프로젝트의 주 개발(IDE) 환경이자, 클라우드에 코드를 배포하기 전 모든 실험을 검증하는 "전초 기지"로 활용하기 위해 도입한 환경입니다.

## 🎯 사용 목적 (Why use this?)
- **즉각적인 피드백 루프**: 코드 수정 후 즉시 실행하여 버그를 잡고, 데이터의 분포를 시각화(EDA)하여 인사이트를 도출합니다.
- **트리 모델의 성지**: LightGBM, CatBoost 같은 트리 기반 앙상블 모델은 클라우드 GPU보다 로컬 CPU 및 통합 메모리(Unified Memory) 환경에서 훨씬 빠르고 유연하게 학습됩니다.
- **데이터 전처리 병목 해결**: 24GB+ 의 거대한 통합 메모리를 활용하여 클라우드에서는 메모리가 터져서(OOM) 불가능한 대규모 데이터프레임 병합(Merge)을 완벽히 처리합니다.

## 🚀 사용 방법 (How to use?)

### 1. 환경 및 인증 설정
로컬 환경 자체는 별도의 API 키가 필요하지 않지만, 가상환경 셋업이 필수적입니다.
```bash
# 가상환경 자동 생성 및 패키지 설치
make install

# 필요 시 활성화
source .venv/bin/activate
```

### 2. PyTorch MPS 디바이스 할당 스니펫
로컬에서 딥러닝 텐서를 연산할 때는 Apple Silicon의 GPU(MPS)를 활성화해야 합니다.

```python
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps") # Apple Silicon 가속
    return torch.device("cpu")

device = get_device()
model = model.to(device)
```

## ⚠️ 실전 주의 사항
- **MPS Fallback 주의**: 최신 논문의 특수 딥러닝 연산자 중 일부는 아직 Apple MPS 백엔드에서 지원하지 않는 경우가 있습니다. 이때 에러가 발생하면 터미널에 `export PYTORCH_ENABLE_MPS_FALLBACK=1`을 선언하여 미지원 연산만 CPU로 넘어가도록 조치하세요.
- **Dry Run 필수**: 클라우드 커널에 코드를 던지기 전, 반드시 로컬에서 **1배치 단위의 적은 샘플 데이터**로 1 에폭(Epoch) 훈련을 돌려 문법 에러나 텐서 사이즈 미스매치가 없는지 먼저 검증하세요.
