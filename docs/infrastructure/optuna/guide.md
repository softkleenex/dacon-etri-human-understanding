# Optuna 튜닝 가이드

단순한 Random Search나 Grid Search를 넘어, 베이지안 최적화(Bayesian Optimization) 기반으로 가장 효율적인 하이퍼파라미터 조합을 탐색하는 라이브러리입니다.

## 1. 튜닝 전략 (Tuning Strategy)
대회에서 가장 점수를 많이 올릴 수 있는 중요한 파라미터부터 탐색합니다.
- **Tree 기반 모델 (LightGBM, XGBoost)**: `learning_rate`, `max_depth`, `num_leaves`, `min_child_samples`, `colsample_bytree`
- **Neural Network**: `learning_rate`, `weight_decay`, `dropout`, 히든 레이어 크기, 배치 사이즈

## 2. K-Fold 교차 검증과 연동
대회 리더보드 점수(LB)와 로컬 검증 점수(CV)의 신뢰성을 맞추기 위해, 파라미터 한 세트(Trial)에 대해 항상 K-Fold 교차 검증을 수행하고 그 **평균 점수(Mean Score)**를 반환해야 합니다.

```python
import optuna
import numpy as np

def objective(trial):
    # 1. 파라미터 탐색 공간 정의
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
    }
    
    cv_scores = []
    # 2. K-Fold 검증
    for fold in range(5):
        model = get_model("lightgbm", params)
        score = train_and_evaluate(model, fold)
        cv_scores.append(score)
        
        # Early Pruning: 첫 번째 폴드 결과가 너무 안 좋으면 일찍 종료
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(cv_scores)

# 3. SQLite DB를 활용하여 히스토리 저장 (재시작 가능)
study = optuna.create_study(
    direction='maximize', 
    study_name='dacon_baseline_tuning',
    storage='sqlite:///optuna_study.db',
    load_if_exists=True
)
study.optimize(objective, n_trials=50)

print(f"Best Value: {study.best_value}")
print(f"Best Params: {study.best_params}")
```

## 3. 지속적 최적화 (Continuous Tuning)
Optuna는 `storage` 파라미터에 SQLite DB를 지정하면 실험이 도중에 중단되어도 언제든 다시 이어서 탐색할 수 있습니다. `make run-local` 파이프라인의 일부로 편입시켜 코드를 작성하거나 밥을 먹는 동안 백그라운드 튜닝을 돌려놓는 것을 권장합니다.
