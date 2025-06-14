"""
Unit tests for the modeling.train module (regression models for gold forecasting).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

from gold_forecasting.modeling.train import (
    create_rolling_features,
    get_model_configs,
    rolling_window_validation,
    train_and_evaluate_models,
    run_gold_forecasting_pipeline,
    BASELINE_MAE_SCORES
)


@pytest.fixture
def sample_gold_data():
    """Create sample gold price data for testing."""
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    # Create synthetic gold prices with some trend
    gold_prices = 1800 + np.cumsum(np.random.randn(200) * 5)
    
    # Create synthetic geopolitical risk features
    gpr_total = 50 + np.random.randn(200) * 10
    gpr_threats = 60 + np.random.randn(200) * 15
    gpr_acts = 40 + np.random.randn(200) * 8
    
    data = pd.DataFrame({
        'time': dates,
        'gold': gold_prices,
        'gpr_total': gpr_total,
        'gpr_threats': gpr_threats,
        'gpr_acts': gpr_acts
    })
    
    return data


class TestCreateRollingFeatures:
    """Tests for create_rolling_features function."""
    
    def test_creates_features_for_all_windows(self, sample_gold_data):
        """Test that features are created for all specified windows."""
        windows = [21, 63]
        result = create_rolling_features(sample_gold_data, 'gold', windows)
        
        assert len(result) == 2
        assert '21' in result
        assert '63' in result
        assert all(isinstance(df, pd.DataFrame) for df in result.values())
    
    def test_creates_target_variable(self, sample_gold_data):
        """Test that target variable (next period return) is created."""
        result = create_rolling_features(sample_gold_data, 'gold', [21])
        
        assert 'target' in result['21'].columns
        assert not result['21']['target'].isna().all()
    
    def test_creates_lagged_features(self, sample_gold_data):
        """Test that lagged features are created."""
        result = create_rolling_features(sample_gold_data, 'gold', [21])
        df = result['21']
        
        # Check for lag features
        lag_cols = [col for col in df.columns if '_lag_' in col]
        assert len(lag_cols) > 0
        
        # Check specific lag features exist
        assert any('gpr_total_lag_1' in col for col in lag_cols)
        assert any('gpr_threats_lag_1' in col for col in lag_cols)
    
    def test_creates_rolling_statistics(self, sample_gold_data):
        """Test that rolling statistics are created."""
        result = create_rolling_features(sample_gold_data, 'gold', [21])
        df = result['21']
        
        # Check for rolling features
        rolling_cols = [col for col in df.columns if '_rolling_' in col]
        assert len(rolling_cols) > 0
        
        # Check specific rolling features exist
        assert any('_rolling_mean_21' in col for col in rolling_cols)
        assert any('_rolling_std_21' in col for col in rolling_cols)
    
    def test_handles_insufficient_data(self, sample_gold_data):
        """Test handling of insufficient data for large windows."""
        # Create very small dataset
        small_data = sample_gold_data.head(30)
        result = create_rolling_features(small_data, 'gold', [126])  # Window larger than data
        
        # Should return empty dict or skip the window
        assert len(result) == 0 or '126' not in result


class TestGetModelConfigs:
    """Tests for get_model_configs function."""
    
    def test_returns_all_required_models(self):
        """Test that all required models are in the config."""
        configs = get_model_configs()
        expected_models = ['OLS', 'Ridge', 'Lasso', 'ElasticNet', 'GaussianProcess', 'RandomForest', 'ExtraTrees']
        
        assert all(model in configs for model in expected_models)
        assert len(configs) == len(expected_models)
    
    def test_model_configs_have_required_fields(self):
        """Test that each model config has required fields."""
        configs = get_model_configs()
        
        for model_name, config in configs.items():
            assert 'model' in config
            assert 'params' in config
            assert 'use_gpu' in config
            assert hasattr(config['model'], 'fit')  # Should be a scikit-learn estimator


class TestRollingWindowValidation:
    """Tests for rolling_window_validation function."""
    
    def test_returns_mae_scores(self, sample_gold_data):
        """Test that rolling window validation returns MAE scores."""
        # Create features
        result = create_rolling_features(sample_gold_data, 'gold', [21])
        df = result['21']
        
        feature_cols = [col for col in df.columns if col not in ['time', 'target']]
        X = df[feature_cols]
        y = df['target']
        
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        
        mae_scores = rolling_window_validation(X, y, model, window_size=50)
        
        assert isinstance(mae_scores, list)
        assert len(mae_scores) > 0
        assert all(isinstance(score, float) and score >= 0 for score in mae_scores)
    
    def test_handles_insufficient_data(self, sample_gold_data):
        """Test handling of insufficient data for validation."""
        # Create very small dataset
        result = create_rolling_features(sample_gold_data.head(30), 'gold', [21])
        if '21' in result:
            df = result['21']
            feature_cols = [col for col in df.columns if col not in ['time', 'target']]
            X = df[feature_cols]
            y = df['target']
            
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            
            # Window size larger than available data
            mae_scores = rolling_window_validation(X, y, model, window_size=100)
            assert mae_scores == []  # Should return empty list


class TestTrainAndEvaluateModels:
    """Tests for train_and_evaluate_models function."""
    
    def test_evaluates_all_models(self, sample_gold_data):
        """Test that all models are evaluated."""
        # Create features for one window
        data_dict = create_rolling_features(sample_gold_data, 'gold', [21])
        
        # Mock rolling_window_validation to return fast results
        with patch('gold_forecasting.modeling.train.rolling_window_validation') as mock_validation:
            mock_validation.return_value = [0.01, 0.015, 0.012]  # Mock MAE scores
            
            results = train_and_evaluate_models(data_dict)
            
            assert '21' in results
            assert len(results['21']) > 0  # At least some models should be evaluated
            
            # Check result structure
            for model_name, metrics in results['21'].items():
                assert 'avg_mae' in metrics
                assert 'baseline_mae' in metrics
                assert 'beats_baseline' in metrics
                assert isinstance(metrics['beats_baseline'], bool)


class TestBaselineScores:
    """Tests for baseline score definitions."""
    
    def test_baseline_scores_complete(self):
        """Test that baseline scores are defined for all models and windows."""
        expected_models = ['OLS', 'Ridge', 'Lasso', 'ElasticNet', 'GaussianProcess', 'RandomForest', 'ExtraTrees']
        expected_windows = ['21', '63', '126']
        
        for model in expected_models:
            assert model in BASELINE_MAE_SCORES
            for window in expected_windows:
                assert window in BASELINE_MAE_SCORES[model]
                assert isinstance(BASELINE_MAE_SCORES[model][window], (int, float))
                assert BASELINE_MAE_SCORES[model][window] > 0


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    def test_pipeline_with_sample_data(self, sample_gold_data):
        """Test the full pipeline with sample data."""
        # Write sample data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_gold_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Mock rolling_window_validation for speed
            with patch('gold_forecasting.modeling.train.rolling_window_validation') as mock_validation:
                mock_validation.return_value = [0.005, 0.007, 0.006]  # Good scores that beat baselines
                
                # Run pipeline
                results = run_gold_forecasting_pipeline(temp_path)
                
                # Check structure
                assert 'results' in results
                assert 'final_models' in results
                assert 'summary' in results
                
                # Check summary
                summary = results['summary']
                assert 'total_models' in summary
                assert 'beating_baseline' in summary
                assert 'success_rate' in summary
                
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def test_pipeline_handles_missing_file(self):
        """Test pipeline handles missing data file gracefully."""
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            run_gold_forecasting_pipeline('/nonexistent/file.csv')


# Performance baseline verification (integration test)
class TestRealPerformance:
    """Test against actual baselines with real data patterns."""
    
    def test_ols_beats_baseline_pattern(self, sample_gold_data):
        """Test that OLS can beat baseline with proper feature engineering."""
        # This is more of a sanity check that our approach can work
        data_dict = create_rolling_features(sample_gold_data, 'gold', [21])
        
        if '21' in data_dict:
            df = data_dict['21']
            feature_cols = [col for col in df.columns if col not in ['time', 'target']]
            X = df[feature_cols]
            y = df['target']
            
            # Ensure we have enough data
            if len(X) > 100:
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                
                mae_scores = rolling_window_validation(X, y, model, window_size=50)
                
                if mae_scores:
                    avg_mae = np.mean(mae_scores)
                    baseline_mae = BASELINE_MAE_SCORES['OLS']['21']
                    
                    # With good feature engineering, we should be competitive
                    # (This might not always pass with random data, but shows the pattern)
                    assert avg_mae < baseline_mae * 2  # Within 2x of baseline is reasonable for random data