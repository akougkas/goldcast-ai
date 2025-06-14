"""
GPU-accelerated training module using RAPIDS cuML for enhanced performance.
Leverages RTX 5090's 32GB VRAM for massive speed improvements.
"""

import pandas as pd
import numpy as np
import structlog
from typing import Any, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# GPU-accelerated imports
import cuml
import cupy as cp
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.linear_model import Ridge as cuRidge
from cuml.linear_model import Lasso as cuLasso
from cuml.linear_model import ElasticNet as cuElasticNet
from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.model_selection import train_test_split as cu_train_test_split

# CPU fallbacks for models not available in cuML
from sklearn.linear_model import HuberRegressor, QuantileRegressor, SGDRegressor, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

logger = structlog.get_logger(__name__)

def enhanced_model_evaluation_gpu(y_true: cp.ndarray, y_pred: cp.ndarray) -> Dict[str, float]:
    """
    GPU-accelerated comprehensive evaluation metrics for financial models.
    """
    metrics = {}
    
    # Convert to numpy for sklearn metrics if needed
    y_true_np = cp.asnumpy(y_true) if isinstance(y_true, cp.ndarray) else y_true
    y_pred_np = cp.asnumpy(y_pred) if isinstance(y_pred, cp.ndarray) else y_pred
    
    # Standard regression metrics (GPU accelerated where possible)
    metrics['mae'] = float(cp.mean(cp.abs(y_true - y_pred)))
    metrics['mse'] = float(cp.mean((y_true - y_pred) ** 2))
    metrics['rmse'] = float(cp.sqrt(metrics['mse']))
    
    # Robust metrics (GPU accelerated)
    abs_errors = cp.abs(y_true - y_pred)
    metrics['median_ae'] = float(cp.median(abs_errors))
    metrics['q90_ae'] = float(cp.percentile(abs_errors, 90))
    
    # Huber loss (GPU accelerated)
    delta = 1.35
    residual = cp.abs(y_true - y_pred)
    huber_mask = residual <= delta
    huber_losses = cp.where(
        huber_mask,
        0.5 * residual**2,
        delta * residual - 0.5 * delta**2
    )
    metrics['huber_loss'] = float(cp.mean(huber_losses))
    
    # Quantile losses (GPU accelerated)
    def quantile_loss_gpu(y_true, y_pred, quantile):
        residual = y_true - y_pred
        return float(cp.mean(cp.maximum(quantile * residual, (quantile - 1) * residual)))
    
    metrics['quantile_loss_10'] = quantile_loss_gpu(y_true, y_pred, 0.1)
    metrics['quantile_loss_90'] = quantile_loss_gpu(y_true, y_pred, 0.9)
    
    # Directional accuracy (GPU accelerated)
    if len(y_true) > 1:
        direction_actual = cp.sign(y_true)
        direction_pred = cp.sign(y_pred)
        metrics['directional_accuracy'] = float(cp.mean(direction_actual == direction_pred))
    else:
        metrics['directional_accuracy'] = 0.0
    
    # Information Coefficient (GPU accelerated correlation)
    if cp.std(y_true) > 0 and cp.std(y_pred) > 0:
        # GPU correlation calculation
        y_true_centered = y_true - cp.mean(y_true)
        y_pred_centered = y_pred - cp.mean(y_pred)
        correlation = cp.sum(y_true_centered * y_pred_centered) / cp.sqrt(
            cp.sum(y_true_centered**2) * cp.sum(y_pred_centered**2)
        )
        metrics['information_coefficient'] = float(correlation)
    else:
        metrics['information_coefficient'] = 0.0
    
    return metrics

# Enhanced baseline scores for GPU models
BASELINE_MAE_SCORES_GPU = {
    'cuOLS': {'21': 0.025050, '63': 0.020837, '126': 0.018284},
    'cuRidge': {'21': 0.024659, '63': 0.020573, '126': 0.018098},
    'cuLasso': {'21': 0.024659, '63': 0.020573, '126': 0.018098},
    'cuElasticNet': {'21': 0.024659, '63': 0.020573, '126': 0.018098},
    'cuRandomForest': {'21': 0.007892, '63': 0.007134, '126': 0.006824},
    # CPU fallbacks for models not in cuML
    'HuberRegressor': {'21': 0.024000, '63': 0.020000, '126': 0.017500},
    'QuantileRegressor': {'21': 0.023000, '63': 0.019500, '126': 0.017000},
    'RobustElasticNet': {'21': 0.022000, '63': 0.018500, '126': 0.016000},
    'RobustLasso': {'21': 0.021500, '63': 0.018000, '126': 0.015500},
    'BayesianRidge': {'21': 0.024000, '63': 0.020000, '126': 0.017500},
    'GaussianProcess': {'21': 0.015234, '63': 0.012678, '126': 0.011134},
    'ExtraTrees': {'21': 0.008845, '63': 0.007967, '126': 0.007612}
}

def create_enhanced_features_gpu(data: pd.DataFrame, target_col: str, lookback_windows: List[int] = [21, 63, 126]) -> Dict[str, pd.DataFrame]:
    """
    GPU-accelerated enhanced feature engineering using unified feature engineering.
    """
    from ..features.unified_features import create_features
    
    logger.info("Using unified feature engineering with GPU acceleration")
    return create_features(data, target_col, lookback_windows, use_gpu=True)

def get_gpu_model_configs() -> Dict[str, Dict]:
    """Get GPU-accelerated model configurations for RTX 5090."""
    return {
        # GPU-accelerated models using cuML
        'cuOLS': {
            'model': cuLinearRegression(),
            'params': {},
            'use_gpu': True
        },
        'cuRidge': {
            'model': cuRidge(alpha=1.0),
            'params': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'use_gpu': True
        },
        'cuLasso': {
            'model': cuLasso(alpha=1.0, max_iter=1000),
            'params': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
            'use_gpu': True
        },
        'cuElasticNet': {
            'model': cuElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            },
            'use_gpu': True
        },
        'cuRandomForest': {
            'model': cuRandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                n_bins=32,  # GPU optimization
                split_criterion='mse',
                bootstrap=True,
                random_state=42
            ),
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 16],
                'n_bins': [16, 32, 64]  # GPU-specific parameter
            },
            'use_gpu': True
        },
        
        # CPU fallbacks for models not available in cuML
        'HuberRegressor': {
            'model': HuberRegressor(epsilon=1.35, alpha=0.0001, max_iter=100),
            'params': {
                'epsilon': [1.35, 1.5, 2.0],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'use_gpu': False
        },
        'QuantileRegressor': {
            'model': QuantileRegressor(quantile=0.5, alpha=1.0),
            'params': {
                'quantile': [0.1, 0.5, 0.9],
                'alpha': [0.1, 1.0, 10.0]
            },
            'use_gpu': False
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
            'use_gpu': False
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
            'use_gpu': False
        },
        'BayesianRidge': {
            'model': BayesianRidge(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6),
            'params': {
                'alpha_1': [1e-6, 1e-5, 1e-4],
                'alpha_2': [1e-6, 1e-5, 1e-4],
                'lambda_1': [1e-6, 1e-5, 1e-4],
                'lambda_2': [1e-6, 1e-5, 1e-4]
            },
            'use_gpu': False,
            'provides_uncertainty': True
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
            'use_gpu': False
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
            'use_gpu': False
        }
    }

def rolling_window_validation_gpu(X: pd.DataFrame, y: pd.Series, model: Any, model_name: str, window_size: int = 126) -> Dict[str, List[float]]:
    """
    GPU-accelerated rolling window validation with enhanced metrics.
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
    is_gpu_model = model_name.startswith('cu')
    
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
        
        if is_gpu_model:
            # GPU-accelerated scaling and training
            scaler = cuStandardScaler()
            X_train_gpu = cp.asarray(X_train.values, dtype=cp.float32)
            X_test_gpu = cp.asarray(X_test.values, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train.values, dtype=cp.float32)
            y_test_gpu = cp.asarray(y_test.values, dtype=cp.float32)
            
            # Fit scaler on GPU
            X_train_scaled = scaler.fit_transform(X_train_gpu)
            X_test_scaled = scaler.transform(X_test_gpu)
            
            # Train model on GPU
            from sklearn.base import clone
            model_copy = clone(model)
            model_copy.fit(X_train_scaled, y_train_gpu)
            
            # Predict on GPU
            y_pred_gpu = model_copy.predict(X_test_scaled)
            fold_metrics = enhanced_model_evaluation_gpu(y_test_gpu, y_pred_gpu)
            
        else:
            # CPU fallback for models not in cuML
            from sklearn.preprocessing import StandardScaler
            from sklearn.base import clone
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model_copy = clone(model)
            model_copy.fit(X_train_scaled, y_train)
            
            y_pred = model_copy.predict(X_test_scaled)
            
            # Convert to cupy for consistent evaluation
            y_test_gpu = cp.asarray(y_test.values)
            y_pred_gpu = cp.asarray(y_pred)
            fold_metrics = enhanced_model_evaluation_gpu(y_test_gpu, y_pred_gpu)
            
            # Add uncertainty for Bayesian models
            if hasattr(model_copy, 'predict') and hasattr(model_copy, 'alpha_') and hasattr(model_copy, 'lambda_'):
                try:
                    y_pred_mean, y_pred_std = model_copy.predict(X_test_scaled, return_std=True)
                    fold_metrics['prediction_std'] = np.mean(y_pred_std)
                    fold_metrics['prediction_uncertainty'] = np.mean(y_pred_std / (np.abs(y_pred_mean) + 1e-8))
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

def train_and_evaluate_gpu_models(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
    """
    GPU-accelerated training and evaluation of all models.
    """
    model_configs = get_gpu_model_configs()
    results = {}
    
    for window, data in data_dict.items():
        logger.info(f"GPU training models for {window}-day window ({len(data)} samples)")
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col not in ['time', 'target']]
        X = data[feature_cols]
        y = data['target']
        
        window_results = {}
        
        for model_name, config in model_configs.items():
            logger.info(f"{'GPU' if config['use_gpu'] else 'CPU'} training {model_name} for {window}-day window")
            
            try:
                # Perform rolling window validation with GPU acceleration
                all_metrics = rolling_window_validation_gpu(X, y, config['model'], model_name)
                
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
                baseline_mae = BASELINE_MAE_SCORES_GPU[model_name][window]
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
                    'gpu_accelerated': config['use_gpu']
                }
                
                status = "✅ BEATS BASELINE" if beats_baseline else "❌ Below baseline"
                gpu_status = "🚀 GPU" if config['use_gpu'] else "💻 CPU"
                dir_acc = avg_metrics.get('directional_accuracy', 0)
                logger.info(f"{gpu_status} {model_name} ({window}d): MAE={avg_mae:.6f} vs baseline={baseline_mae:.6f} {status}, "
                           f"Dir.Acc={dir_acc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} for {window}-day window: {e}")
                continue
        
        results[window] = window_results
    
    return results

def run_gpu_gold_forecasting_pipeline(data_path: str) -> Dict[str, Any]:
    """
    Run the complete GPU-accelerated gold forecasting pipeline.
    """
    logger.info("🚀 Starting GPU-accelerated gold forecasting pipeline")
    
    # Load data using our data loader
    from ..data_processing.loader import load_data
    data = load_data(data_path)
    data = data.reset_index()  # Reset index to get time as column
    
    logger.info(f"Loaded {len(data)} rows of data from {data['time'].min()} to {data['time'].max()}")
    
    # Create GPU-enhanced features
    data_dict = create_enhanced_features_gpu(data, target_col='Gold_Price_USD', lookback_windows=[21, 63, 126])
    
    # Train and evaluate models with GPU acceleration
    results = train_and_evaluate_gpu_models(data_dict)
    
    # Summary with GPU performance insights
    logger.info("=== GPU PIPELINE RESULTS ===")
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
    logger.info(f"GPU ACCELERATION: {gpu_models}/{total_models} models ran on RTX 5090")
    
    return {
        'results': results,
        'data_dict': data_dict,
        'summary': {
            'total_models': total_models,
            'beating_baseline': beating_baseline,
            'success_rate': beating_baseline / total_models if total_models > 0 else 0,
            'gpu_accelerated_models': gpu_models,
            'gpu_acceleration_rate': gpu_models / total_models if total_models > 0 else 0
        }
    }

if __name__ == "__main__":
    import sys
    data_path = "/home/akougkas/development/apm-ai/asset_data/db_com_capm_gold.csv"
    
    try:
        pipeline_results = run_gpu_gold_forecasting_pipeline(data_path)
        
        print("\n=== 🚀 GPU-ACCELERATED FINAL RESULTS ===")
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
        print(f"GPU ACCELERATION: {pipeline_results['summary']['gpu_accelerated_models']}/{pipeline_results['summary']['total_models']} ({gpu_rate:.1%})")
        print(f"\n🚀 GPU Optimizations:")
        print(f"   • cuML GPU-accelerated: Linear, Ridge, Lasso, ElasticNet, RandomForest")
        print(f"   • GPU-accelerated feature engineering with CuPy")
        print(f"   • GPU-accelerated metrics evaluation")
        print(f"   • Leveraging RTX 5090's 32GB VRAM for massive datasets")
        print(f"   • Hybrid GPU/CPU approach for maximum model coverage")
        
    except Exception as e:
        logger.error(f"GPU Pipeline failed: {e}", exc_info=True)
        sys.exit(1)