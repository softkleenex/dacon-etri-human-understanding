# 로컬 환경 (Local Environment - Apple Silicon) 가이드

본 프로젝트의 주 개발 환경 및 로컬 테스트 환경입니다. 강력한 성능을 내는 Apple Silicon 칩(M 시리즈)의 잠재력을 최대로 끌어올려 모델 검증 시간을 단축하기 위한 실무 가이드입니다.

---

## 1. 하드웨어 환경 및 강점
- **기기**: MacBook Pro (Apple Silicon)
- **강점 (Unified Memory)**: 일반적인 PC는 RAM과 VRAM(비디오 메모리)이 분리되어 있어 데이터 전송에 병목이 발생합니다. 하지만 Apple Silicon은 통합 메모리를 사용하므로, 24GB+ 의 거대한 데이터프레임을 GPU 메모리 고갈(Out of Memory) 걱정 없이 빠르게 연산할 수 있습니다. EDA 및 대규모 전처리에 매우 유리합니다.

---

## 2. 가상 환경 및 패키지 관리
시스템 파이썬을 오염시키지 않기 위해 프로젝트 디렉토리 내에 독립적인 가상환경(`.venv`)을 구성하고 관리합니다.

```bash
# 1. 의존성 설치 및 가상환경 자동 생성 (Makefile 활용)
make install

# 2. 가상환경 활성화 (필요 시 수동 진입)
source .venv/bin/activate
```

---

## 3. PyTorch MPS (Metal Performance Shaders) 활용
Apple Silicon 환경에서는 NVIDIA의 CUDA 대신 Apple의 MPS 백엔드를 사용하여 GPU 가속을 받습니다. 모델 학습이나 추론 스크립트 작성 시 디바이스 할당 로직을 반드시 아래와 같이 구성하세요.

```python
import torch

def get_device() -> torch.device:
    """사용 가능한 최적의 하드웨어 가속기를 반환합니다."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 MPS (Apple Silicon GPU) 가속 활성화됨!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 CUDA GPU 가속 활성화됨!")
    else:
        device = torch.device("cpu")
        print("⚠️ CPU로 동작합니다.")
    return device

# 모델 및 텐서를 디바이스로 이동
device = get_device()
model = MyNeuralNet().to(device)
inputs = inputs.to(device)
```

> **⚠️ 주의 사항 (MPS Fallback)**
> 최신 논문에 등장하는 일부 복잡한 커스텀 연산자(Custom Operators)는 아직 MPS에서 미지원할 수 있습니다. 이 경우 환경 변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 선언해두면, 미지원 연산만 자동으로 CPU 연산으로 떨어져(Fallback) 에러를 방지할 수 있습니다.

---

## 4. 로컬 환경의 주 활용 전략

1. **EDA 및 데이터 검증**: `.ipynb` (Jupyter) 파일에서 Matplotlib/Seaborn을 활용해 데이터 분포를 시각화하고 인사이트를 발굴합니다.
2. **파이프라인 구축 및 디버깅**: 클라우드(Kaggle)에 무거운 코드를 배포하기 전, **1~2개의 샘플 배치**만 떼어내어 로컬에서 1 에폭(Epoch)을 돌려보는 "Dry Run" 과정을 반드시 거칩니다. 에러를 사전에 잡아냅니다.
3. **트리 계열 모델 완전 학습**: 딥러닝과 달리 LightGBM, CatBoost, XGBoost 같은 트리 기반 앙상블 모델은 로컬 CPU와 통합 메모리 환경만으로도 빠르고 완벽한 K-Fold 학습이 가능합니다. 트리 모델 튜닝은 전적으로 로컬에서 수행합니다.
