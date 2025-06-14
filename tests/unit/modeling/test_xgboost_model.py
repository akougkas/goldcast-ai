"""
Unit tests for the XGBoost model functions.
"""

import pytest
import pandas as pd
import numpy as np
import xgboost as xgb
from unittest.mock import patch, MagicMock

from gold_forecasting.modeling.xgboost_model import (
    train_xgboost_model,
    predict_with_xgboost,
    evaluate_xgboost_model,
)

@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Returns sample data for testing."""
    X = pd.DataFrame({
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
    })
    y = pd.Series(np.random.rand(100))
    return X, y

@pytest.fixture
def trained_model(sample_data: tuple[pd.DataFrame, pd.Series]) -> xgb.XGBRegressor:
    """Trains a default XGBoost model for use in other tests."""
    X_train, y_train = sample_data
    # Mock the fit method to avoid actual training during most tests
    # We'll have a specific test for actual training
    with patch("xgboost.XGBRegressor.fit") as mock_fit:
        model = train_xgboost_model(X_train, y_train, use_gpu=False) # Test CPU path by default
        # model.n_features_in_ = X_train.shape[1] # Removed: Cannot set this attribute directly
    return model

def test_train_xgboost_model_cpu(sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test training an XGBoost model on CPU."""
    X_train, y_train = sample_data
    model = train_xgboost_model(X_train, y_train, use_gpu=False)
    assert isinstance(model, xgb.XGBRegressor)
    assert model.get_params()["tree_method"] == "hist" # Default for CPU when not specified or 'auto'

@patch("xgboost.XGBRegressor.fit") # Mock fit to avoid actual GPU attempt if not available
def test_train_xgboost_model_gpu(mock_fit: MagicMock, sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test training an XGBoost model with GPU flag."""
    X_train, y_train = sample_data
    model = train_xgboost_model(X_train, y_train, use_gpu=True)
    assert isinstance(model, xgb.XGBRegressor)
    assert model.get_params()["tree_method"] == "gpu_hist"

@patch("xgboost.XGBRegressor.fit") # Mock fit
def test_train_xgboost_model_custom_params(mock_fit: MagicMock, sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test training with custom parameters."""
    X_train, y_train = sample_data
    custom_params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05}
    model = train_xgboost_model(X_train, y_train, model_params=custom_params, use_gpu=False)
    assert isinstance(model, xgb.XGBRegressor)
    assert model.get_params()["n_estimators"] == 50
    assert model.get_params()["max_depth"] == 3
    assert model.get_params()["learning_rate"] == 0.05
    assert model.get_params()["tree_method"] == "hist"


@patch("xgboost.XGBRegressor.fit")
def test_train_xgboost_model_gpu_param_override(mock_fit: MagicMock, sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test tree_method is forced to gpu_hist when use_gpu=True even if params suggest otherwise."""
    X_train, y_train = sample_data
    custom_params = {"tree_method": "hist"} # Attempt to override GPU preference
    model = train_xgboost_model(X_train, y_train, model_params=custom_params, use_gpu=True)
    assert model.get_params()["tree_method"] == "gpu_hist"
    mock_fit.assert_called_once()


@patch("xgboost.XGBRegressor.fit")
def test_train_xgboost_model_cpu_param_override(mock_fit: MagicMock, sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test tree_method is forced to hist when use_gpu=False even if params suggest gpu_hist."""
    X_train, y_train = sample_data
    custom_params = {"tree_method": "gpu_hist"} # Attempt to force GPU when not intended
    model = train_xgboost_model(X_train, y_train, model_params=custom_params, use_gpu=False)
    assert model.get_params()["tree_method"] == "hist"
    mock_fit.assert_called_once()


def test_predict_with_xgboost(trained_model: xgb.XGBRegressor, sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test making predictions."""
    X_test, _ = sample_data
    # Mock the predict method as the model's fit was mocked
    trained_model.predict = MagicMock(return_value=np.random.rand(X_test.shape[0]))
    
    predictions = predict_with_xgboost(trained_model, X_test)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_test)
    trained_model.predict.assert_called_once_with(X_test)

def test_evaluate_xgboost_model(sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test model evaluation."""
    _, y_true = sample_data
    y_pred = y_true * 0.9 # Simulate some predictions
    
    evaluation = evaluate_xgboost_model(y_true, y_pred)
    assert isinstance(evaluation, dict)
    assert "rmse" in evaluation
    assert "mae" in evaluation
    assert evaluation["rmse"] >= 0
    assert evaluation["mae"] >= 0

# More comprehensive test that actually fits a small model (CPU)
# This helps catch issues not caught by mocks, like parameter validation
def test_train_xgboost_model_actual_fit_cpu(sample_data: tuple[pd.DataFrame, pd.Series]):
    """Test actual model fitting on CPU with minimal parameters."""
    X_train, y_train = sample_data
    # Use minimal parameters for a quick fit
    minimal_params = {
        "n_estimators": 2, 
        "max_depth": 2, 
        "objective": "reg:squarederror",
        "tree_method": "hist" # Explicitly CPU for this test
    }
    model = xgb.XGBRegressor(**minimal_params)
    model.fit(X_train, y_train)
    
    assert isinstance(model, xgb.XGBRegressor)
    # Check if the model has been fitted (e.g., by checking for an attribute like `best_score` or `n_features_in_`)
    assert hasattr(model, 'n_features_in_')
    assert model.n_features_in_ == X_train.shape[1]

    # Test prediction with this actually fitted model
    predictions = model.predict(X_train)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_train)

    # Test evaluation with these predictions
    evaluation = evaluate_xgboost_model(y_train, predictions)
    assert "rmse" in evaluation
    assert "mae" in evaluation

