# 로컬 환경 (Apple Silicon) 활용 매뉴얼

이 프로젝트의 주 개발(IDE) 환경이자, 클라우드에 코드를 배포하기 전 모든 실험을 검증하는 "전초 기지"입니다.

## 🎯 사용 목적 (Why use this?)
- **즉각적인 피드백 루프**: 코드 수정 후 즉시 실행하여 버그를 잡고, 데이터의 분포를 시각화(EDA)하여 인사이트를 도출합니다.
- **트리 모델의 성지**: LightGBM, CatBoost 같은 트리 기반 앙상블 모델은 클라우드 GPU보다 로컬 CPU 및 통합 메모리 환경에서 훨씬 빠르고 유연하게 학습됩니다. 트리 모델 튜닝은 100% 로컬에서 수행합니다.
- **데이터 전처리 병목 해결**: Apple Silicon의 거대한 Unified Memory(24GB+)를 활용하여, 클라우드 커널에서는 메모리가 터져서(OOM) 불가능한 대규모 데이터프레임 병합(Merge) 및 통계 피처 생성을 완벽히 처리합니다.

## 🚀 사용 방법 (How to use?)

### 1. PyTorch MPS(Metal Performance Shaders) 디바이스 할당
로컬에서 딥러닝 텐서를 연산할 때는 GPU를 활성화해야 합니다. 코드 최상단에 디바이스 할당 로직을 아래와 같이 배치하여 Apple GPU를 호출합니다.

```python
import torch

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps") # Apple Silicon 가속
    return torch.device("cpu")

device = get_device()
model = model.to(device)
```

### 2. "Dry Run" 검증 (클라우드 배포 전 필수 단계)
클라우드 커널에 코드를 던지기 전, 반드시 로컬에서 **적은 양의 샘플 데이터(예: 1배치)**만 통과시켜 문법 에러나 텐서 사이즈 미스매치가 없는지 1 에폭(Epoch) 훈련을 돌려봅니다.
- 에러가 나면 즉각 로컬에서 디버깅하여 수정합니다.
- 성공적으로 1 에폭이 완료되면 그때 `cloud_env` 가이드에 따라 코드를 클라우드로 Push합니다.

## ⚠️ 실전 주의 사항
최신 논문의 특수 딥러닝 연산자(Custom Operators) 중 일부는 아직 Apple MPS 백엔드에서 지원하지 않는 경우가 있습니다. 
이때 에러가 발생하면 터미널 환경 변수에 `export PYTORCH_ENABLE_MPS_FALLBACK=1`을 선언하여, 미지원 연산만 자동으로 CPU로 넘어가도록(Fallback) 조치하세요.
