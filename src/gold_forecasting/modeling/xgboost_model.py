"""
XGBoost model training, prediction, and evaluation functions.
"""

import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Tuple


def train_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: Dict[str, Any] | None = None,
    use_gpu: bool = True,
) -> xgb.XGBRegressor:
    """
    Trains an XGBoost regressor model.

    Args:
        X_train: Training features.
        y_train: Training target.
        model_params: Dictionary of parameters for XGBoost. 
                      Defaults will be used if None or if specific params are missing.
        use_gpu: Whether to try using GPU for training (tree_method='gpu_hist').

    Returns:
        Trained XGBoost model.
    """
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }
    if use_gpu:
        params["tree_method"] = "gpu_hist"
    elif "tree_method" not in params: # if not using GPU and tree_method not already set by model_params
        params["tree_method"] = "hist"

    if model_params:
        params.update(model_params) # Override defaults with user-provided params
        # Ensure tree_method reflects use_gpu preference if model_params tries to override it incorrectly
        if use_gpu and params.get("tree_method") != "gpu_hist":
            params["tree_method"] = "gpu_hist"
        elif not use_gpu and params.get("tree_method") == "gpu_hist":
            # If user explicitly passes tree_method='gpu_hist' but use_gpu=False, prioritize use_gpu=False
            params["tree_method"] = "hist"

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def predict_with_xgboost(model: xgb.XGBRegressor, X_test: pd.DataFrame) -> np.ndarray:
    """
    Makes predictions using a trained XGBoost model.

    Args:
        model: Trained XGBoost model.
        X_test: Test features.

    Returns:
        Array of predictions.
    """
    predictions = model.predict(X_test)
    return predictions


def evaluate_xgboost_model(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluates the XGBoost model performance.

    Args:
        y_true: True target values.
        y_pred: Predicted values.

    Returns:
        Dictionary of evaluation metrics (RMSE, MAE).
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmse": rmse, "mae": mae}
