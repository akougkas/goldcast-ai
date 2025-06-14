"""
Tests for the unified training module.
"""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from gold_forecasting.modeling.unified_train import UnifiedModelTrainer

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample time series data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    data = {
        'time': dates,
        'gold': 1500 + np.cumsum(np.random.randn(len(dates)) * 5),
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def cpu_trainer() -> UnifiedModelTrainer:
    """Create a CPU trainer instance."""
    return UnifiedModelTrainer(use_gpu=False)

@pytest.fixture 
def gpu_trainer() -> UnifiedModelTrainer:
    """Create a GPU trainer instance."""
    return UnifiedModelTrainer(use_gpu=True)

def test_trainer_initialization():
    """Test trainer initialization with CPU and GPU modes."""
    cpu_trainer = UnifiedModelTrainer(use_gpu=False)
    assert not cpu_trainer.use_gpu
    
    gpu_trainer = UnifiedModelTrainer(use_gpu=True)
    # GPU availability depends on environment
    assert gpu_trainer.use_gpu in [True, False]

def test_baseline_scores_exist(cpu_trainer):
    """Test that baseline scores are properly defined."""
    assert 'OLS' in cpu_trainer.baseline_scores
    assert 'Ridge' in cpu_trainer.baseline_scores
    assert '21' in cpu_trainer.baseline_scores['OLS']
    assert '63' in cpu_trainer.baseline_scores['OLS']
    assert '126' in cpu_trainer.baseline_scores['OLS']

def test_cpu_model_configs(cpu_trainer):
    """Test CPU model configurations."""
    configs = cpu_trainer.get_model_configs()
    
    # Should have all expected models
    expected_models = [
        'OLS', 'Ridge', 'Lasso', 'ElasticNet', 'GaussianProcess',
        'RandomForest', 'ExtraTrees', 'HuberRegressor', 'QuantileRegressor',
        'RobustElasticNet', 'RobustLasso', 'BayesianRidge'
    ]
    
    for model_name in expected_models:
        assert model_name in configs
        assert 'model' in configs[model_name]
        assert 'params' in configs[model_name]
        assert 'supports_gpu' in configs[model_name]

def test_gpu_model_configs(gpu_trainer):
    """Test GPU model configurations."""
    configs = gpu_trainer.get_model_configs()
    
    # Should have all models
    assert len(configs) >= 12
    
    # Check for GPU-specific models if GPU is available
    if gpu_trainer.use_gpu:
        gpu_models = [name for name, config in configs.items() 
                     if config.get('gpu_accelerated', False)]
        assert len(gpu_models) > 0

def test_enhanced_model_evaluation(cpu_trainer):
    """Test enhanced model evaluation metrics."""
    np.random.seed(42)
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.randn(100) * 0.1  # Add some noise
    
    metrics = cpu_trainer.enhanced_model_evaluation(y_true, y_pred)
    
    # Check all expected metrics are present
    expected_metrics = [
        'mae', 'mse', 'rmse', 'median_ae', 'q90_ae', 'huber_loss',
        'quantile_loss_10', 'quantile_loss_90', 'directional_accuracy',
        'information_coefficient'
    ]
    
    for metric in expected_metrics:
        assert metric in metrics
        assert isinstance(metrics[metric], (int, float))
        assert not np.isnan(metrics[metric])

def test_rolling_window_validation_insufficient_data(cpu_trainer):
    """Test rolling window validation with insufficient data."""
    # Create small dataset
    X = pd.DataFrame(np.random.randn(50, 10))
    y = pd.Series(np.random.randn(50))
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    # Should return empty dict due to insufficient data
    result = cpu_trainer.rolling_window_validation(X, y, model, 'OLS', window_size=126)
    assert result == {}

def test_rolling_window_validation_sufficient_data(cpu_trainer):
    """Test rolling window validation with sufficient data."""
    # Create sufficient dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(300, 10))
    y = pd.Series(np.random.randn(300))
    
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    
    result = cpu_trainer.rolling_window_validation(X, y, model, 'OLS', window_size=63)
    
    # Should have metrics
    assert 'mae' in result
    assert len(result['mae']) > 0
    assert all(isinstance(mae, (int, float)) for mae in result['mae'])

@patch('gold_forecasting.modeling.unified_train.create_features')
def test_train_and_evaluate_models(mock_create_features, cpu_trainer):
    """Test training and evaluation of models."""
    # Mock feature engineering output - need enough data for 21-day window
    np.random.seed(42)
    mock_data = pd.DataFrame({
        'feature_1': np.random.randn(300),
        'feature_2': np.random.randn(300),
        'feature_3': np.random.randn(300),
        'target': np.random.randn(300)
    })
    
    data_dict = {'21': mock_data}
    
    # Train only a subset of models for speed
    original_configs = cpu_trainer.get_model_configs()
    test_configs = {name: config for name, config in original_configs.items() 
                   if name in ['OLS', 'Ridge']}
    
    with patch.object(cpu_trainer, 'get_model_configs', return_value=test_configs):
        results = cpu_trainer.train_and_evaluate_models(data_dict)
    
    # Should have results for the window
    assert '21' in results
    assert 'OLS' in results['21'] or 'Ridge' in results['21']
    
    # Check result structure
    for model_results in results['21'].values():
        assert 'avg_mae' in model_results
        assert 'beats_baseline' in model_results
        assert 'gpu_accelerated' in model_results

def test_run_training_pipeline_with_temp_file(cpu_trainer, sample_data):
    """Test the complete training pipeline with a temporary file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_path = f.name
    
    try:
        # Patch to use only a few models for speed
        original_configs = cpu_trainer.get_model_configs()
        test_configs = {name: config for name, config in original_configs.items() 
                       if name in ['OLS', 'Ridge']}
        
        with patch.object(cpu_trainer, 'get_model_configs', return_value=test_configs):
            results = cpu_trainer.run_training_pipeline(temp_path)
        
        # Check pipeline results structure
        assert 'results' in results
        assert 'summary' in results
        assert 'data_dict' in results
        
        # Check summary
        summary = results['summary']
        assert 'total_models' in summary
        assert 'beating_baseline' in summary
        assert 'success_rate' in summary
        assert 'use_gpu' in summary
        
    finally:
        # Clean up
        Path(temp_path).unlink()

def test_gpu_fallback_when_unavailable():
    """Test that GPU mode falls back to CPU when RAPIDS not available."""
    with patch('gold_forecasting.modeling.unified_train.HAS_GPU', False):
        trainer = UnifiedModelTrainer(use_gpu=True)
        assert not trainer.use_gpu

def test_gpu_model_evaluation_fallback(gpu_trainer):
    """Test GPU model evaluation falls back to CPU when needed."""
    # Should handle regular numpy arrays gracefully
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    
    metrics = gpu_trainer.enhanced_model_evaluation(y_true, y_pred)
    
    assert 'mae' in metrics
    assert isinstance(metrics['mae'], (int, float))

def test_model_name_mapping():
    """Test that model names are consistent between CPU and GPU configs."""
    cpu_trainer = UnifiedModelTrainer(use_gpu=False)
    gpu_trainer = UnifiedModelTrainer(use_gpu=True)
    
    cpu_models = set(cpu_trainer.get_model_configs().keys())
    gpu_models = set(gpu_trainer.get_model_configs().keys())
    
    # Should have the same model names (though implementations may differ)
    assert cpu_models == gpu_models