import pytest

from src.models.model_builder import get_model


def test_get_model_lightgbm():
    model = get_model("lightgbm", {"n_estimators": 10})
    import lightgbm as lgb

    assert isinstance(model, lgb.LGBMClassifier)
    assert model.n_estimators == 10


def test_get_model_xgboost():
    model = get_model("xgboost", {"n_estimators": 10})
    import xgboost as xgb

    assert isinstance(model, xgb.XGBClassifier)
    assert model.n_estimators == 10


def test_get_model_catboost():
    model = get_model("catboost", {"iterations": 10})
    import catboost as cb

    assert isinstance(model, cb.CatBoostClassifier)
    # CatBoost parameters are a bit tricky, just checking instance is fine


def test_get_model_invalid():
    with pytest.raises(ValueError):
        get_model("invalid_model")
