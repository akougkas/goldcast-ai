#!/usr/bin/env python3
"""
Unified training module for gold price forecasting models.
Supports both CPU and GPU acceleration with dynamic model selection.
"""

import argparse
import pandas as pd
import numpy as np
import structlog
from typing import Any, Dict, List, Optional
import warnings
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

warnings.filterwarnings('ignore')

# Standard imports
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import clone

# CPU model imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import HuberRegressor, QuantileRegressor, SGDRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# GPU imports (optional)
try:
    import cupy as cp
    import cuml
    from cuml.linear_model import LinearRegression as cuLinearRegression
    from cuml.linear_model import Ridge as cuRidge
    from cuml.linear_model import Lasso as cuLasso
    from cuml.linear_model import ElasticNet as cuElasticNet
    from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cuml = None

# Local imports
from gold_forecasting.features.unified_features import create_features
from gold_forecasting.config import settings

logger = structlog.get_logger(__name__)

class UnifiedModelTrainer:
    """
    Unified model trainer supporting both CPU and GPU acceleration.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the trainer.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and HAS_GPU
        
        if use_gpu and not HAS_GPU:
            logger.warning("GPU requested but RAPIDS not available, falling back to CPU")
            self.use_gpu = False
            
        logger.info(f"UnifiedModelTrainer initialized with {'GPU' if self.use_gpu else 'CPU'} acceleration")
        
        # Baseline MAE scores to beat
        self.baseline_scores = {
            'OLS': {'21': 0.025050, '63': 0.020837, '126': 0.018284},
            'Ridge': {'21': 0.024659, '63': 0.020573, '126': 0.018098},
            'Lasso': {'21': 0.024659, '63': 0.020573, '126': 0.018098},
            'ElasticNet': {'21': 0.024659, '63': 0.020573, '126': 0.018098},
            'GaussianProcess': {'21': 0.015234, '63': 0.012678, '126': 0.011134},
            'RandomForest': {'21': 0.007892, '63': 0.007134, '126': 0.006824},
            'ExtraTrees': {'21': 0.008845, '63': 0.007967, '126': 0.007612},
            'HuberRegressor': {'21': 0.024000, '63': 0.020000, '126': 0.017500},
            'QuantileRegressor': {'21': 0.023000, '63': 0.019500, '126': 0.017000},
            'RobustElasticNet': {'21': 0.022000, '63': 0.018500, '126': 0.016000},
            'RobustLasso': {'21': 0.021500, '63': 0.018000, '126': 0.015500},
            'BayesianRidge': {'21': 0.024000, '63': 0.020000, '126': 0.017500}
        }
    
    def get_model_configs(self) -> Dict[str, Dict]:
        """
        Get model configurations with dynamic CPU/GPU selection.
        """
        if self.use_gpu:
            return self._get_gpu_model_configs()
        else:
            return self._get_cpu_model_configs()
    
    def _get_cpu_model_configs(self) -> Dict[str, Dict]:
        """CPU model configurations."""
        return {
            'OLS': {
                'model': LinearRegression(),
                'params': {},
                'supports_gpu': True
            },
            'Ridge': {
                'model': Ridge(alpha=1.0, random_state=42),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'supports_gpu': True
            },
            'Lasso': {
                'model': Lasso(alpha=1.0, random_state=42, max_iter=10000),
                'params': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
                'supports_gpu': True
            },
            'ElasticNet': {
                'model': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=10000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                'supports_gpu': True
            },
            'GaussianProcess': {
                'model': GaussianProcessRegressor(
                    kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5),
                    alpha=1e-10,
                    normalize_y=True,
                    random_state=42,
                    n_restarts_optimizer=2
                ),
                'params': {},
                'supports_gpu': False
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                },
                'supports_gpu': True
            },
            'ExtraTrees': {
                'model': ExtraTreesRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                },
                'supports_gpu': False
            },
            'HuberRegressor': {
                'model': HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=100),
                'params': {
                    'epsilon': [1.35, 1.5, 2.0],
                    'alpha': [0.0001, 0.001, 0.01]
                },
                'supports_gpu': False
            },
            'QuantileRegressor': {
                'model': QuantileRegressor(quantile=0.5, alpha=1.0),
                'params': {
                    'quantile': [0.1, 0.5, 0.9],
                    'alpha': [0.1, 1.0, 10.0]
                },
                'supports_gpu': False
            },
            'RobustElasticNet': {
                'model': SGDRegressor(
                    loss='huber',
                    penalty='elasticnet',
                    l1_ratio=0.5,
                    epsilon=0.1,
                    alpha=0.01,
                    random_state=42,
                    max_iter=1000
                ),
                'params': {
                    'alpha': [0.001, 0.01, 0.1],
                    'l1_ratio': [0.15, 0.5, 0.85],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'supports_gpu': False
            },
            'RobustLasso': {
                'model': SGDRegressor(
                    loss='huber',
                    penalty='l1',
                    epsilon=0.1,
                    alpha=0.01,
                    random_state=42,
                    max_iter=1000
                ),
                'params': {
                    'alpha': [0.001, 0.01, 0.1],
                    'epsilon': [0.01, 0.1, 0.2]
                },
                'supports_gpu': False
            },
            'BayesianRidge': {
                'model': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
                'params': {
                    'alpha_1': [1e-6, 1e-5, 1e-4],
                    'alpha_2': [1e-6, 1e-5, 1e-4],
                    'lambda_1': [1e-6, 1e-5, 1e-4],
                    'lambda_2': [1e-6, 1e-5, 1e-4]
                },
                'supports_gpu': False,
                'provides_uncertainty': True
            }
        }
    
    def _get_gpu_model_configs(self) -> Dict[str, Dict]:
        """GPU model configurations."""
        gpu_configs = {
            'OLS': {
                'model': cuLinearRegression(),
                'params': {},
                'supports_gpu': True,
                'gpu_accelerated': True
            },
            'Ridge': {
                'model': cuRidge(alpha=1.0),
                'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
                'supports_gpu': True,
                'gpu_accelerated': True
            },
            'Lasso': {
                'model': cuLasso(alpha=1.0, max_iter=1000),
                'params': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
                'supports_gpu': True,
                'gpu_accelerated': True
            },
            'ElasticNet': {
                'model': cuElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000),
                'params': {
                    'alpha': [0.001, 0.01, 0.1, 1.0],
                    'l1_ratio': [0.1, 0.5, 0.7, 0.9]
                },
                'supports_gpu': True,
                'gpu_accelerated': True
            },
            'RandomForest': {
                'model': cuRandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    n_bins=32,
                    split_criterion='mse',
                    bootstrap=True,
                    random_state=42
                ),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 16],
                    'n_bins': [16, 32, 64]
                },
                'supports_gpu': True,
                'gpu_accelerated': True
            }
        }
        
        # Add CPU fallbacks for models not available in cuML
        cpu_fallbacks = self._get_cpu_model_configs()
        for model_name in ['GaussianProcess', 'ExtraTrees', 'HuberRegressor', 
                          'QuantileRegressor', 'RobustElasticNet', 'RobustLasso', 'BayesianRidge']:
            gpu_configs[model_name] = cpu_fallbacks[model_name].copy()
            gpu_configs[model_name]['gpu_accelerated'] = False
        
        return gpu_configs
    
    def enhanced_model_evaluation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive evaluation metrics with GPU/CPU compatibility.
        """
        # Convert GPU arrays to CPU if needed
        if self.use_gpu and HAS_GPU and hasattr(y_true, '__array_interface__'):
            y_true = cp.asnumpy(y_true) if (cp is not None and isinstance(y_true, cp.ndarray)) else y_true
            y_pred = cp.asnumpy(y_pred) if (cp is not None and isinstance(y_pred, cp.ndarray)) else y_pred
        
        metrics = {}
        
        # Standard regression metrics
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Robust metrics
        abs_errors = np.abs(y_true - y_pred)
        metrics['median_ae'] = np.median(abs_errors)
        metrics['q90_ae'] = np.percentile(abs_errors, 90)
        
        # Huber loss
        delta = 1.35
        residual = np.abs(y_true - y_pred)
        huber_mask = residual <= delta
        huber_losses = np.where(
            huber_mask,
            0.5 * residual**2,
            delta * residual - 0.5 * delta**2
        )
        metrics['huber_loss'] = np.mean(huber_losses)
        
        # Quantile losses
        def quantile_loss(y_true, y_pred, quantile):
            residual = y_true - y_pred
            return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
        
        metrics['quantile_loss_10'] = quantile_loss(y_true, y_pred, 0.1)
        metrics['quantile_loss_90'] = quantile_loss(y_true, y_pred, 0.9)
        
        # Directional accuracy
        if len(y_true) > 1:
            direction_actual = np.sign(y_true)
            direction_pred = np.sign(y_pred)
            metrics['directional_accuracy'] = np.mean(direction_actual == direction_pred)
        else:
            metrics['directional_accuracy'] = 0.0
        
        # Information Coefficient
        if np.std(y_true) > 0 and np.std(y_pred) > 0:
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            metrics['information_coefficient'] = correlation if not np.isnan(correlation) else 0.0
        else:
            metrics['information_coefficient'] = 0.0
        
        return metrics
    
    def rolling_window_validation(self, X: pd.DataFrame, y: pd.Series, model: Any, 
                                 model_name: str, window_size: int = 126) -> Dict[str, List[float]]:
        """
        Unified rolling window validation with GPU/CPU support.
        """
        n_samples = len(X)
        if n_samples < window_size * 2:
            logger.warning(f"Insufficient data for rolling validation: {n_samples} < {window_size * 2}")
            return {}
        
        all_metrics = {
            'mae': [], 'mse': [], 'rmse': [], 'median_ae': [], 'q90_ae': [],
            'huber_loss': [], 'quantile_loss_10': [], 'quantile_loss_90': [],
            'directional_accuracy': [], 'information_coefficient': [],
            'prediction_std': [], 'prediction_uncertainty': []
        }
        
        n_folds = (n_samples - window_size) // (window_size // 4)
        model_configs = self.get_model_configs()
        is_gpu_model = (self.use_gpu and model_configs.get(model_name, {}).get('gpu_accelerated', False))
        
        for i in range(n_folds):
            start_idx = i * (window_size // 4)
            train_end = start_idx + window_size
            test_end = min(train_end + window_size // 4, n_samples)
            
            if test_end >= n_samples:
                break
                
            X_train = X.iloc[start_idx:train_end]
            y_train = y.iloc[start_idx:train_end]
            X_test = X.iloc[train_end:test_end]
            y_test = y.iloc[train_end:test_end]
            
            if is_gpu_model and HAS_GPU:
                # GPU-accelerated training
                scaler = cuStandardScaler()
                X_train_gpu = cp.asarray(X_train.values, dtype=cp.float32)
                X_test_gpu = cp.asarray(X_test.values, dtype=cp.float32)
                y_train_gpu = cp.asarray(y_train.values, dtype=cp.float32)
                y_test_gpu = cp.asarray(y_test.values, dtype=cp.float32)
                
                # Critical: Fit scaler only on training data
                X_train_scaled = scaler.fit_transform(X_train_gpu)
                X_test_scaled = scaler.transform(X_test_gpu)
                
                # Train model on GPU
                model_copy = clone(model)
                model_copy.fit(X_train_scaled, y_train_gpu)
                
                # Predict on GPU
                y_pred_gpu = model_copy.predict(X_test_scaled)
                fold_metrics = self.enhanced_model_evaluation(
                    cp.asnumpy(y_test_gpu), cp.asnumpy(y_pred_gpu)
                )
            else:
                # CPU training
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model_copy = clone(model)
                model_copy.fit(X_train_scaled, y_train)
                
                y_pred = model_copy.predict(X_test_scaled)
                fold_metrics = self.enhanced_model_evaluation(y_test.values, y_pred)
                
                # Add uncertainty for Bayesian models
                if (hasattr(model_copy, 'predict') and 
                    hasattr(model_copy, 'alpha_') and hasattr(model_copy, 'lambda_')):
                    try:
                        y_pred_mean, y_pred_std = model_copy.predict(X_test_scaled, return_std=True)
                        fold_metrics['prediction_std'] = np.mean(y_pred_std)
                        fold_metrics['prediction_uncertainty'] = np.mean(
                            y_pred_std / (np.abs(y_pred_mean) + 1e-8)
                        )
                    except:
                        fold_metrics['prediction_std'] = 0.0
                        fold_metrics['prediction_uncertainty'] = 0.0
            
            # Store all metrics
            for metric_name, value in fold_metrics.items():
                if metric_name in all_metrics:
                    all_metrics[metric_name].append(value)
                elif metric_name in ['prediction_std', 'prediction_uncertainty']:
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
            
            logger.debug(f"Fold {i+1}: Train {start_idx}-{train_end}, Test {train_end}-{test_end}, "
                        f"MAE: {fold_metrics['mae']:.6f}, Dir.Acc: {fold_metrics['directional_accuracy']:.3f}")
        
        return all_metrics
    
    def train_and_evaluate_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Train and evaluate all models for all lookback windows.
        """
        model_configs = self.get_model_configs()
        results = {}
        
        for window, data in data_dict.items():
            logger.info(f"{'GPU' if self.use_gpu else 'CPU'} training models for {window}-day window ({len(data)} samples)")
            
            # Prepare features and target
            feature_cols = [col for col in data.columns if col not in ['time', 'target']]
            X = data[feature_cols]
            y = data['target']
            
            window_results = {}
            
            for model_name, config in model_configs.items():
                gpu_status = "🚀 GPU" if config.get('gpu_accelerated', False) else "💻 CPU"
                logger.info(f"{gpu_status} training {model_name} for {window}-day window")
                
                try:
                    # Perform rolling window validation
                    all_metrics = self.rolling_window_validation(X, y, config['model'], model_name)
                    
                    if not all_metrics or not all_metrics.get('mae'):
                        logger.warning(f"No validation scores for {model_name} on {window}-day window")
                        continue
                    
                    # Calculate averages for all metrics
                    avg_metrics = {}
                    std_metrics = {}
                    for metric_name, values in all_metrics.items():
                        if values:
                            avg_metrics[metric_name] = np.mean(values)
                            std_metrics[metric_name] = np.std(values)
                    
                    avg_mae = avg_metrics.get('mae', float('inf'))
                    baseline_mae = self.baseline_scores[model_name][window]
                    beats_baseline = bool(avg_mae < baseline_mae)
                    
                    window_results[model_name] = {
                        'avg_mae': avg_mae,
                        'std_mae': std_metrics.get('mae', 0),
                        'baseline_mae': baseline_mae,
                        'beats_baseline': beats_baseline,
                        'improvement': (baseline_mae - avg_mae) / baseline_mae * 100 if baseline_mae > 0 else 0,
                        'n_folds': len(all_metrics['mae']),
                        'avg_directional_accuracy': avg_metrics.get('directional_accuracy', 0),
                        'avg_information_coefficient': avg_metrics.get('information_coefficient', 0),
                        'avg_huber_loss': avg_metrics.get('huber_loss', float('inf')),
                        'avg_median_ae': avg_metrics.get('median_ae', float('inf')),
                        'avg_q90_ae': avg_metrics.get('q90_ae', float('inf')),
                        'avg_prediction_std': avg_metrics.get('prediction_std', 0),
                        'avg_prediction_uncertainty': avg_metrics.get('prediction_uncertainty', 0),
                        'all_metrics': avg_metrics,
                        'gpu_accelerated': config.get('gpu_accelerated', False)
                    }
                    
                    status = "✅ BEATS BASELINE" if beats_baseline else "❌ Below baseline"
                    dir_acc = avg_metrics.get('directional_accuracy', 0)
                    logger.info(f"{gpu_status} {model_name} ({window}d): MAE={avg_mae:.6f} vs baseline={baseline_mae:.6f} {status}, "
                               f"Dir.Acc={dir_acc:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {window}-day window: {e}")
                    continue
            
            results[window] = window_results
        
        return results
    
    def run_training_pipeline(self, data_path: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        """
        acceleration = "GPU-accelerated" if self.use_gpu else "CPU"
        logger.info(f"🚀 Starting {acceleration} gold forecasting pipeline")
        
        # Load data
        data = pd.read_csv(data_path)
        data['time'] = pd.to_datetime(data['time'])
        data = data.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Loaded {len(data)} rows of data from {data['time'].min()} to {data['time'].max()}")
        
        # Create features using unified feature engineering
        data_dict = create_features(data, target_col='gold', lookback_windows=[21, 63, 126], use_gpu=self.use_gpu)
        
        # Train and evaluate models
        results = self.train_and_evaluate_models(data_dict)
        
        # Summary
        logger.info("=== PIPELINE RESULTS ===")
        total_models = 0
        beating_baseline = 0
        gpu_models = 0
        
        for window in ['21', '63', '126']:
            if window in results:
                window_beating = sum(1 for r in results[window].values() if r['beats_baseline'])
                window_total = len(results[window])
                window_gpu = sum(1 for r in results[window].values() if r.get('gpu_accelerated', False))
                
                total_models += window_total
                beating_baseline += window_beating
                gpu_models += window_gpu
                
                logger.info(f"{window}-day window: {window_beating}/{window_total} models beat baseline ({window_gpu} GPU-accelerated)")
        
        logger.info(f"OVERALL: {beating_baseline}/{total_models} models beat their baselines")
        if self.use_gpu:
            logger.info(f"GPU ACCELERATION: {gpu_models}/{total_models} models ran on GPU")
        
        return {
            'results': results,
            'data_dict': data_dict,
            'summary': {
                'total_models': total_models,
                'beating_baseline': beating_baseline,
                'success_rate': beating_baseline / total_models if total_models > 0 else 0,
                'gpu_accelerated_models': gpu_models,
                'gpu_acceleration_rate': gpu_models / total_models if total_models > 0 else 0,
                'use_gpu': self.use_gpu
            }
        }

def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(description='Unified Gold Price Forecasting Training')
    parser.add_argument('--use-gpu', action='store_true', 
                       help='Use GPU acceleration with RAPIDS cuML')
    parser.add_argument('--data-path', type=str, 
                       default='/home/akougkas/development/apm-ai/asset_data/db_com_capm_gold.csv',
                       help='Path to the gold price CSV file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Configure logging
    import logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    try:
        # Initialize trainer
        trainer = UnifiedModelTrainer(use_gpu=args.use_gpu)
        
        # Run pipeline
        pipeline_results = trainer.run_training_pipeline(args.data_path)
        
        # Display results
        print(f"\n=== {'🚀 GPU-ACCELERATED' if args.use_gpu else '💻 CPU'} FINAL RESULTS ===")
        for window, window_results in pipeline_results['results'].items():
            print(f"\n{window}-day lookback window:")
            for model_name, metrics in window_results.items():
                status = "✅ BEATS" if metrics['beats_baseline'] else "❌ Below"
                gpu_status = "🚀 GPU" if metrics.get('gpu_accelerated', False) else "💻 CPU"
                dir_acc = metrics.get('avg_directional_accuracy', 0)
                uncertainty = metrics.get('avg_prediction_uncertainty', 0)
                
                print(f"  {gpu_status} {model_name:15s}: MAE={metrics['avg_mae']:.6f} vs {metrics['baseline_mae']:.6f} {status}")
                print(f"                         Dir.Acc={dir_acc:.3f} | Uncertainty={uncertainty:.3f}")
        
        success_rate = pipeline_results['summary']['success_rate']
        gpu_rate = pipeline_results['summary']['gpu_acceleration_rate']
        print(f"\nSUCCESS RATE: {pipeline_results['summary']['beating_baseline']}/{pipeline_results['summary']['total_models']} ({success_rate:.1%})")
        
        if args.use_gpu:
            print(f"GPU ACCELERATION: {pipeline_results['summary']['gpu_accelerated_models']}/{pipeline_results['summary']['total_models']} ({gpu_rate:.1%})")
            print(f"\n🚀 GPU Optimizations:")
            print(f"   • cuML GPU-accelerated: Linear, Ridge, Lasso, ElasticNet, RandomForest")
            print(f"   • CPU fallbacks for unsupported models")
            print(f"   • Unified feature engineering with GPU acceleration")
        else:
            print(f"\n💻 CPU Optimizations:")
            print(f"   • Multi-threaded scikit-learn models")
            print(f"   • Efficient pandas operations")
            print(f"   • Optimized feature engineering")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())