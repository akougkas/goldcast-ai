"""
Tests for the unified feature engineering module.
"""

import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch

from gold_forecasting.features.unified_features import (
    UnifiedFeatureEngineer,
    create_features,
    create_rolling_features,
    create_enhanced_features_gpu,
)

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample time series data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    np.random.seed(42)
    
    data = {
        'time': dates,
        'Gold_Price_USD': 1500 + np.cumsum(np.random.randn(len(dates)) * 10),
        'SPX_Index': 3000 + np.cumsum(np.random.randn(len(dates)) * 20),
        'DXY_Index': 95 + np.cumsum(np.random.randn(len(dates)) * 0.5),
        'US_10Y_Yield': 2.0 + np.cumsum(np.random.randn(len(dates)) * 0.1),
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def feature_engineer() -> UnifiedFeatureEngineer:
    """Create a feature engineer instance."""
    return UnifiedFeatureEngineer(use_gpu=False)

def test_unified_feature_engineer_init():
    """Test UnifiedFeatureEngineer initialization."""
    # Test CPU initialization
    engineer_cpu = UnifiedFeatureEngineer(use_gpu=False)
    assert not engineer_cpu.use_gpu
    
    # Test auto-detection
    engineer_auto = UnifiedFeatureEngineer(use_gpu=None)
    assert engineer_auto.use_gpu in [True, False]  # Depends on environment

def test_create_lagged_features(feature_engineer, sample_data):
    """Test lagged feature creation."""
    columns = ['Gold_Price_USD', 'SPX_Index']
    lags = [1, 5, 10]
    
    result = feature_engineer.create_lagged_features(sample_data, columns, lags)
    
    # Check that lagged columns are created
    for col in columns:
        for lag in lags:
            lag_col = f'{col}_lag_{lag}'
            assert lag_col in result.columns
            
            # Check that lag values are correct (ignoring NaN rows)
            original_values = sample_data[col].iloc[:-lag].values
            lagged_values = result[lag_col].iloc[lag:].values
            np.testing.assert_array_equal(original_values, lagged_values)

def test_create_rolling_statistics(feature_engineer, sample_data):
    """Test rolling statistics creation."""
    columns = ['Gold_Price_USD']
    windows = [5, 10]
    
    result = feature_engineer.create_rolling_statistics(sample_data, columns, windows)
    
    # Check that rolling statistics columns are created
    for col in columns:
        for window in windows:
            assert f'{col}_rolling_mean_{window}' in result.columns
            assert f'{col}_rolling_std_{window}' in result.columns
            assert f'{col}_rolling_min_{window}' in result.columns
            assert f'{col}_rolling_max_{window}' in result.columns
            assert f'{col}_rolling_median_{window}' in result.columns
            assert f'{col}_rolling_iqr_{window}' in result.columns
            assert f'{col}_rolling_skew_{window}' in result.columns
            assert f'{col}_rolling_kurt_{window}' in result.columns
            assert f'{col}_rolling_volatility_{window}' in result.columns

def test_create_technical_indicators(feature_engineer, sample_data):
    """Test technical indicators creation."""
    price_col = 'Gold_Price_USD'
    windows = [10, 20]
    
    result = feature_engineer.create_technical_indicators(sample_data, price_col, windows)
    
    # Check that technical indicators are created
    for window in windows:
        assert f'{price_col}_sma_{window}' in result.columns
        assert f'{price_col}_ema_{window}' in result.columns
        assert f'{price_col}_bb_upper_{window}' in result.columns
        assert f'{price_col}_bb_lower_{window}' in result.columns
        assert f'{price_col}_bb_width_{window}' in result.columns
        assert f'{price_col}_momentum_{window}' in result.columns
        assert f'{price_col}_position_{window}' in result.columns
        
        if window >= 14:
            assert f'{price_col}_rsi_{window}' in result.columns

def test_create_returns_and_volatility(feature_engineer, sample_data):
    """Test returns and volatility measures creation."""
    price_col = 'Gold_Price_USD'
    windows = [10, 20]
    
    result = feature_engineer.create_returns_and_volatility(sample_data, price_col, windows)
    
    # Check basic return columns
    assert f'{price_col}_return' in result.columns
    assert f'{price_col}_log_return' in result.columns
    
    # Check volatility measures for each window
    for window in windows:
        assert f'{price_col}_vol_cluster_{window}' in result.columns
        assert f'{price_col}_downside_risk_{window}' in result.columns
        assert f'{price_col}_upside_potential_{window}' in result.columns
        assert f'{price_col}_var_5_{window}' in result.columns
        assert f'{price_col}_cvar_5_{window}' in result.columns
        assert f'{price_col}_max_drawdown_{window}' in result.columns

def test_create_interaction_features(feature_engineer, sample_data):
    """Test interaction features creation."""
    price_col = 'Gold_Price_USD'
    other_cols = ['SPX_Index', 'DXY_Index']
    
    result = feature_engineer.create_interaction_features(sample_data, price_col, other_cols)
    
    # Check interaction features
    for col in other_cols:
        assert f'{price_col}_x_{col}' in result.columns
        assert f'{price_col}_ratio_{col}' in result.columns
        assert f'{price_col}_diff_{col}' in result.columns
        assert f'{price_col}_corr_{col}_21' in result.columns
        assert f'{price_col}_corr_{col}_63' in result.columns

def test_create_calendar_features(feature_engineer, sample_data):
    """Test calendar features creation."""
    result = feature_engineer.create_calendar_features(sample_data, 'time')
    
    # Check basic calendar features
    calendar_features = [
        'year', 'month', 'day', 'day_of_week', 'day_of_year', 
        'week_of_year', 'quarter'
    ]
    
    for feature in calendar_features:
        assert feature in result.columns
    
    # Check cyclical encoding
    cyclical_features = [
        'month_sin', 'month_cos', 'day_of_week_sin', 'day_of_week_cos',
        'day_of_year_sin', 'day_of_year_cos'
    ]
    
    for feature in cyclical_features:
        assert feature in result.columns
    
    # Check financial calendar features
    financial_features = [
        'is_month_start', 'is_month_end', 'is_quarter_start',
        'is_quarter_end', 'is_year_start', 'is_year_end'
    ]
    
    for feature in financial_features:
        assert feature in result.columns

def test_create_target_variable(feature_engineer, sample_data):
    """Test target variable creation."""
    price_col = 'Gold_Price_USD'
    
    # Test return target
    result_return = feature_engineer.create_target_variable(sample_data, price_col, 'return', 1)
    assert 'target' in result_return.columns
    
    # Test log return target
    result_log = feature_engineer.create_target_variable(sample_data, price_col, 'log_return', 1)
    assert 'target' in result_log.columns
    
    # Test price target
    result_price = feature_engineer.create_target_variable(sample_data, price_col, 'price', 1)
    assert 'target' in result_price.columns

def test_generate_features_for_window(feature_engineer, sample_data):
    """Test feature generation for a specific window."""
    target_col = 'Gold_Price_USD'
    window_size = 21
    
    result = feature_engineer.generate_features_for_window(sample_data, target_col, window_size)
    
    # Check that we have more features than the original data
    assert len(result.columns) > len(sample_data.columns)
    
    # Check that target column exists
    assert 'target' in result.columns
    
    # Check that we have fewer rows due to NaN removal
    assert len(result) <= len(sample_data)

def test_generate_features_for_all_windows(feature_engineer, sample_data):
    """Test feature generation for multiple windows."""
    target_col = 'Gold_Price_USD'
    windows = [21, 63]
    
    results = feature_engineer.generate_features_for_all_windows(sample_data, target_col, windows)
    
    # Check that we get results for each window
    for window in windows:
        assert str(window) in results
        assert len(results[str(window)].columns) > len(sample_data.columns)
        assert 'target' in results[str(window)].columns

def test_create_features_interface(sample_data):
    """Test the main create_features interface."""
    target_col = 'Gold_Price_USD'
    windows = [21, 63]
    
    # Test CPU version
    results_cpu = create_features(sample_data, target_col, windows, use_gpu=False)
    assert isinstance(results_cpu, dict)
    assert len(results_cpu) == len(windows)
    
    # Test with default GPU setting
    results_default = create_features(sample_data, target_col, windows)
    assert isinstance(results_default, dict)

def test_backward_compatibility_functions(sample_data):
    """Test backward compatibility functions."""
    target_col = 'Gold_Price_USD'
    windows = [21, 63]
    
    # Test CPU backward compatibility
    results_cpu = create_rolling_features(sample_data, target_col, windows)
    assert isinstance(results_cpu, dict)
    
    # Test GPU backward compatibility
    results_gpu = create_enhanced_features_gpu(sample_data, target_col, windows)
    assert isinstance(results_gpu, dict)

def test_missing_columns_handling(feature_engineer, sample_data):
    """Test handling of missing columns."""
    # Test with non-existent column
    result = feature_engineer.create_lagged_features(sample_data, ['NonExistent'], [1])
    # Should return original data without errors
    assert len(result.columns) == len(sample_data.columns)

def test_insufficient_data_handling(feature_engineer):
    """Test handling of insufficient data."""
    # Create very small dataset
    small_data = pd.DataFrame({
        'time': pd.date_range('2020-01-01', periods=5),
        'Gold_Price_USD': [100, 101, 102, 103, 104]
    })
    
    # Try to generate features for large window
    results = feature_engineer.generate_features_for_all_windows(small_data, 'Gold_Price_USD', [100])
    
    # Should return empty dict due to insufficient data
    assert len(results) == 0

@patch('gold_forecasting.features.unified_features.HAS_GPU', False)
def test_gpu_fallback(sample_data):
    """Test GPU fallback when RAPIDS not available."""
    target_col = 'Gold_Price_USD'
    
    # Should fall back to CPU even when GPU requested
    results = create_features(sample_data, target_col, [21], use_gpu=True)
    assert isinstance(results, dict)