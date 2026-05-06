# 로컬 환경 (Local Environment - Apple Silicon) 가이드

본 프로젝트의 주 개발 환경 및 로컬 테스트 환경입니다. 강력한 성능을 내는 Apple Silicon 칩의 잠재력을 최대로 끌어올려 사용하기 위한 가이드입니다.

## 1. 하드웨어 환경
- **기기**: MacBook Pro (Apple Silicon: M 시리즈 칩)
- **장점**: Unified Memory 덕분에 GPU 메모리 병목 없이 대용량 전처리와 시각화를 빠르게 수행 가능.

## 2. 환경 세팅 (Virtual Environment)
시스템 파이썬을 보호하기 위해 프로젝트 디렉토리 내에 독립적인 가상환경(`.venv`)을 구성합니다. 
```bash
# 초기 셋업 (Makefile 활용)
make install
```

## 3. PyTorch MPS (Metal Performance Shaders) 활용
Apple Silicon 환경에서는 CUDA 대신 MPS를 사용하여 GPU 가속을 받습니다.

- `src/utils.py`나 `scripts/train.py`의 최상단에서 디바이스를 설정합니다.
```python
import torch

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS 가속 활성화됨!")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device
```
*주의: 일부 복잡한 연산자(Op)는 아직 MPS에서 구현되지 않아 에러가 날 수 있습니다. 이 경우 환경 변수 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 설정하거나 부분적으로 CPU로 넘겨 계산해야 합니다.*

## 4. 로컬에서의 주 목적
1. **EDA 및 데이터 검증**: `.ipynb` 파일을 통한 시각화 및 아이디어 검증.
2. **파이프라인 구축 및 디버깅**: 클라우드(Kaggle 등)에 코드를 배포하기 전에 에러 없이 완벽히 동작하는지 **적은 양의 샘플 데이터**로 1 에폭만 돌려보는 과정.
3. **가벼운 모델(Tree 계열) 완전 학습**: LightGBM, CatBoost는 로컬 CPU/MPS 만으로도 빠르게 훈련 가능합니다.
