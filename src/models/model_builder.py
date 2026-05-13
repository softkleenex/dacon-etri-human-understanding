def get_model(model_name: str, params: dict = None):
    """
    Factory function to get a model instance.

    Args:
        model_name: Name of the model to initialize.
        params: Dictionary of hyperparameters.

    Returns:
        An instantiated model object.
    """
    if params is None:
        params = {}

    if model_name == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(**params)

    elif model_name == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(**params)

    elif model_name == "catboost":
        import catboost as cb

        return cb.CatBoostClassifier(**params)

    else:
        raise ValueError(f"Unknown model: {model_name}")
