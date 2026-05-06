# 🏆 제출 및 실험 기록부 (Submission & Experiment History)

실제 제출 파일과 K-Fold 로컬 검증 점수(CV Score), DACON 리더보드 점수(Public LB Score) 간의 갭(Gap)을 추적하는 공간입니다.

> **💡 핵심 변경 사항 (CSV 트래킹 도입)**
> 로컬 검증(CV) 점수와 실제 제출(LB) 점수를 명확히 구분하고, 파이썬 스크립트가 결과를 자동으로 기록(Append)할 수 있도록 **마크다운 표 대신 `submissions/submission_log.csv` 파일을 사용**하여 데이터를 추적합니다.

## 📈 리더보드 및 로컬 점수 추적 (CSV Log)
모든 실험 결과는 👉 **[`submissions/submission_log.csv`](../submissions/submission_log.csv)** 에 기록됩니다.

### 📝 CSV 컬럼 명세
- `id`: 실험 고유 번호
- `date`: 추론/제출 날짜
- `submission_file`: 생성된 CSV 파일명 (예: `xgb_fold5.csv`)
- `model_description`: 적용한 모델 및 주요 피처 변경점
- **`local_cv_score` (나 스스로 판단한 점수)**: K-Fold 교차 검증 등을 통해 로컬 머신에서 계산된 점수. 모델 훈련/추론 스크립트(`inference.py`)가 자동으로 이 값을 채워 넣습니다.
- **`public_lb_score` (진짜 제출해서 얻은 점수)**: DACON 웹사이트에 실제로 제출(API 또는 수동)한 뒤 얻은 점수. 제출 전에는 비워두고(Null), 제출 후에 사용자가 점수를 기입합니다.
- `is_submitted`: 실제 DACON에 제출했는지 여부 (True/False). 하루 제출 횟수 제한(Quota) 관리를 위해 사용합니다.
- `notes`: 에이전트나 사용자가 남기는 특이사항.

---

## 🛑 튜닝 및 실패 기록 (What Did NOT Work)
(CSV에 담기 힘든 길고 상세한 실패 원인 분석이나 통찰(Insight)은 이 마크다운 문서에 텍스트로 기록합니다.)

- **시도 1**: `feature_x`를 Log 변환 후 투입 (CSV id: 3)
  - **결과**: 로컬 CV 점수가 오히려 하락함.
  - **이유 추정**: Tree 모델 특성상 단조 증가 변환(Monotonic)은 분기(Split)에 큰 영향을 주지 못하며, 노이즈만 증폭시킨 것으로 보임.

