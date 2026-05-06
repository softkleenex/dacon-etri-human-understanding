# Optuna 튜닝 활용 매뉴얼

점수 한계에 부딪혔을 때, 인간의 직관이 아닌 베이지안 최적화(Bayesian Optimization) 기반의 확률적 탐색으로 최적의 하이퍼파라미터 조합을 찾기 위해 도입한 도구입니다.

## 🎯 사용 목적 (Why use this?)
- **최적의 조합 탐색**: 수십 개의 파라미터 간 상관관계를 고려하여 가장 퍼포먼스가 높은 "명당"을 찾아냅니다.
- **백그라운드 무인 튜닝**: 잠들기 전 튜닝 코드를 켜두면, 무인으로 수백 번의 실험을 진행하고 최고 점수의 파라미터만 반환합니다.

## 🚀 사용 방법 (How to use?)

### 1. 인증 및 환경 변수 설정
Optuna 자체는 외부 API와 통신하지 않는 순수 로컬 파이썬 라이브러리이므로 별도의 인증 키나 전역 토큰이 필요하지 않습니다.

### 2. K-Fold 기반 조기 종료(Pruning) 튜닝 스니펫
단일 검증 셋 튜닝은 과적합을 유발합니다. 반드시 K-Fold 교차 검증의 평균 점수를 목적 함수(Objective)로 세팅하세요.

```python
import optuna
import numpy as np

def objective(trial):
    # 1. 탐색할 핵심 파라미터 공간 정의 (Log 스케일 활용)
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    
    cv_scores = []
    # 2. K-Fold 평균 점수 산출
    for fold in range(5):
        score = train_fold_and_evaluate(fold, params)
        cv_scores.append(score)
        
        # 3. 조기 종료 (Early Pruning) 적용
        # 1번째 폴드 점수가 나쁘면 나머지 폴드는 무시하고 버림 (시간 절약)
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(cv_scores)

# DB 파일(optuna_study.db)에 기록을 누적하여 도중 재시작(Resume) 지원
study = optuna.create_study(
    direction='maximize', 
    study_name='lgbm_tuning', 
    storage='sqlite:///optuna_study.db',
    load_if_exists=True 
)

study.optimize(objective, n_trials=50)
print("Best Params:", study.best_params)
```

## ⚠️ 실전 주의 사항
- **탐색 공간 제약**: 파라미터를 10개 이상 무작정 열어두지 마세요. 트리 모델 기준 `learning_rate`, `max_depth`, `num_leaves` 등 점수에 결정적인 3~4개의 Core 파라미터에 집중하는 것이 수렴 속도가 훨씬 빠릅니다.
- **SQLite 동시성**: 여러 프로세스에서 동시에 동일한 SQLite DB에 기록하면 DB Lock이 걸릴 수 있습니다. 병렬 튜닝이 필요하다면 Redis나 RDB 기반 Storage 사용을 고려하세요.
