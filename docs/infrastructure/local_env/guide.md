# 로컬 환경 (Local Environment - Apple Silicon) 가이드

본 프로젝트의 주 개발 환경 및 로컬 테스트 환경입니다.

## 하드웨어 스펙
- **기기**: MacBook Pro
- **프로세서**: M4 Pro 칩 (12 Core CPU)
- **메모리**: 24GB Unified Memory

## 주요 용도 및 역할
1. **탐색적 데이터 분석 (EDA)**: 빠른 메모리 로드와 시각화를 통한 데이터 이해.
2. **피처 엔지니어링 (Feature Engineering)**: 복잡한 수학적 계산 및 전처리 스크립트 작성, 테스트.
3. **가벼운 모델 프로토타이핑**: LightGBM, XGBoost, CatBoost 등을 로컬 CPU/MPS에서 즉각적으로 검증.
4. **신경망 아키텍처 디버깅**: PyTorch MPS(Metal Performance Shaders) 백엔드를 활용한 빠른 모델 구조 테스트.

## 장점
- 클라우드 대기 시간 없이 즉각적인 코드 실행 및 수정 가능.
- 백그라운드 Kaggle 학습과 분리되어 로컬과 클라우드의 리소스를 병렬로 활용 가능.
