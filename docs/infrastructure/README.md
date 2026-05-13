# 🏛 인프라 폴더 매뉴얼 (Infrastructure Folder Manual)

이 폴더(`docs/infrastructure/`)는 DACON 및 Kaggle 경진대회에서 사용되는 **모든 MLOps 인프라, 환경 설정, 라이브러리 활용법 및 사용 목적**이 실무 관점에서 작성된 핵심 문서 저장소입니다.

## 🎯 폴더의 목적
1. **AI 에이전트 지식베이스**: Gemini CLI(AI 에이전트)가 대회를 자율적으로 진행하다가 도구의 사용 목적이나 파이프라인 연동 방식에 막힐 때, 스스로 이 폴더의 매뉴얼들을 참조(Context)하여 문제를 해결할 수 있도록 돕는 뇌(Brain) 역할을 합니다.
2. **빠른 전략 온보딩**: 초기 세팅(인증 토큰 등)은 전역으로 1회만 수행하면 끝납니다. 이 폴더의 문서들은 세팅 가이드가 아니라 **"이 도구를 왜 써야 하며(Purpose), 실제 대회 파이프라인에서 어떻게 써먹는지(Usage)"**에 집중하여 작성되었습니다.

## 📂 인프라 매뉴얼 내비게이션
크게 '실행 환경'과 '활용 도구' 두 가지로 나뉘어 있습니다. (각 폴더명을 클릭하면 상세 매뉴얼로 진입합니다.)

### 🖥 실행 환경 (Environments)
- **[`local_env/guide.md`](local_env/guide.md)**: Apple Silicon 머신의 빵빵한 Unified Memory를 활용한 고속 전처리와 트리 모델 튜닝 기지 활용법.
- **[`cloud_env/guide.md`](cloud_env/guide.md)**: 딥러닝 훈련을 위해 무료 T4 GPU 클라우드(Kaggle 커널)로 무거운 연산을 오프로딩(Off-loading)하는 전략. (기존 Kaggle CLI 매뉴얼 통합됨)

### 🛠 활용 도구 (Tools & Libraries)
- **[`deep_learning/guide.md`](deep_learning/guide.md)**: PyTorch Lightning 기반의 시계열 LSTM(Temporal Attention) 멀티태스크 아키텍처 및 데이터로더 매뉴얼.
- **[`automation/guide.md`](automation/guide.md)**: 훈련 대기 봇, 이륙 전 점검(Pre-flight), 지능형 앙상블 등 에이전트의 완전 자율 워크플로우를 보조하는 스크립트 매뉴얼.
- **[`automation/cv_lb_gap_workflow.md`](automation/cv_lb_gap_workflow.md)**: 로컬 검증 점수(CV)와 퍼블릭 리더보드(LB) 간의 점수 격차를 분석하고 오버피팅을 방지하는 모니터링 가이드.
- **[`gemini/guide.md`](gemini/guide.md)**: 에이전트를 조종하여 자율 진화(Self-Evolving) 워크플로우를 가동하는 프롬프트 작성 및 루프 활용법.
- **[`wandb/guide.md`](wandb/guide.md)**: 훈련 스크립트에 파이썬 코드를 심어 실시간으로 학습 곡선을 모니터링하는 법.
- **[`optuna/guide.md`](optuna/guide.md)**: 조기 종료(Early Pruning)와 SQLite 이어서 튜닝하기를 결합한 베이지안 파라미터 탐색법.
- **[`huggingface/guide.md`](huggingface/guide.md)**: 최신 SOTA 시계열/비전 논문 가중치를 즉각적으로 내 코드에 수혈하는 방법.
- **[`data_processing/guide.md`](data_processing/guide.md)**: 데이터 누수(Data Leakage) 원천 차단 및 MLOps 확장 라이브러리 활용법.
- **[`aihub/guide.md`](aihub/guide.md)**: 대회 성능 한계를 돌파하기 위한 날씨/지리 외부 데이터 결합 전략.
- **[`daconapi/guide.md`](daconapi/guide.md)**: 추론이 끝난 최종 CSV 파일을 DACON 리더보드에 스크립트로 자동 제출하는 방법.

---
## 📝 새로운 매뉴얼 추가 규칙 (확장성)
새로운 라이브러리나 기술 스택을 도입할 때는 다음 규칙에 따라 템플릿을 복사하여 확장합니다.
1. `docs/infrastructure/_template/` 폴더를 복사하여 `docs/infrastructure/[새로운_도구명]/` 으로 이름을 변경합니다.
2. 폴더 내부의 `guide.md` 템플릿 양식에 맞춰, 단순 세팅 가이드가 아닌 **`🎯 사용 목적 (Why use this?)`**과 **`🚀 사용 방법 (How to use?)`** 중심의 실전 연동 코드를 작성하세요.
3. 발생하기 쉬운 대회 실무적 에러를 **`⚠️ 실전 주의 사항`**에 기재합니다.
4. 작성 완료 후 현재 문서(`README.md`)의 내비게이션 트리에 링크를 등록합니다.
