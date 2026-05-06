# Optuna 튜닝 활용 매뉴얼

점수 한계에 부딪혔을 때, 하이퍼파라미터를 직관이나 노가다가 아닌 수학적 확률(베이지안 최적화)을 이용해 가장 점수가 높게 나올 "명당"을 찾아주는 도구입니다.

## 🎯 사용 목적 (Why use this?)
- **최적의 조합 탐색**: `learning_rate`를 낮추면서 `max_depth`를 높이는 게 좋을지, 반대가 좋을지 수백 번 돌려보며 인간이 찾기 힘든 최적의 밸런스 조합을 찾아냅니다.
- **백그라운드 무인 튜닝**: 잠들기 전에 스크립트를 켜두면, 아침에 일어나서 가장 점수가 높았던 파라미터 셋만 복사해서 최종 모델을 훈련시킬 수 있습니다.

## 🚀 사용 방법 (How to use?)

### 1. K-Fold 기반 목적 함수 작성
단일 검증 세트만 튜닝하면 해당 세트에만 과적합(Overfitting)됩니다. 항상 "K-Fold 교차 검증의 평균 점수"를 높이도록 목적 함수를 설계하세요.

```python
import optuna
import numpy as np

def objective(trial):
    # 1. 탐색할 파라미터 공간 정의 (중요한 파라미터만)
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
    }
    
    cv_scores = []
    # 2. K-Fold 평균 점수 산출
    for fold in range(5):
        score = train_fold_and_evaluate(fold, params)
        cv_scores.append(score)
        
        # [중요] 조기 종료(Early Pruning) 적용
        # 1번째 폴드 점수가 이미 쓰레기라면, 2~5번째 폴드는 돌리지 않고 바로 다음 탐색으로 넘어감 (시간 절약)
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
            
    return np.mean(cv_scores)
```

### 2. 끊김 방지(Resume) 튜닝 실행
기본 메모리 탐색은 컴퓨터가 꺼지면 기록이 날아갑니다. SQLite 파일 데이터베이스에 기록을 누적시켜, 도중에 멈춰도 언제든 이어서 튜닝(`load_if_exists=True`)되도록 하세요.

```python
# DB 파일(optuna_study.db)에 탐색 지도를 누적 저장
study = optuna.create_study(
    direction='maximize', 
    study_name='lgbm_tuning', 
    storage='sqlite:///optuna_study.db',
    load_if_exists=True 
)

study.optimize(objective, n_trials=50) # 50번 시도
print("Best Params:", study.best_params)
```

## ⚠️ 실전 주의 사항
탐색 공간(Search Space)을 너무 넓게, 파라미터를 10개 이상 너무 많이 주지 마세요.
트리 모델의 경우 `learning_rate`, `max_depth`, `num_leaves` 등 점수에 결정적인 영향을 미치는 3~4개의 Core 파라미터만 우선적으로 집중 튜닝하는 것이 훨씬 빠르게 최고 점수에 도달하는 비결입니다.
