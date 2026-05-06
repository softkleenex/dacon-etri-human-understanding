# Generic model wrappers or factory functions
def get_model(model_name: str, params: dict = None):
    """Factory function to get a model instance."""
    if params is None:
        params = {}
        
    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(**params) # Or LGBMClassifier
    elif model_name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(**params)
    elif model_name == "catboost":
        import catboost as cb
        return cb.CatBoostRegressor(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
