# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a gold price forecasting system that must beat specific benchmark MAE scores using machine learning models with rolling window predictions. The project implements 9 different models (OLS, Ridge, Lasso, ElasticNet, Gaussian Process, Random Forest, Extra Trees, Theil-Sen, RANSAC) across 3 window sizes (21, 63, 126 days).

## Common Development Commands

**Package Management (uv only):**
```bash
uv sync                    # Install all dependencies
uv add package_name        # Add new dependency
uv run python script.py    # Run Python scripts
uv run pytest             # Run tests
```

**Code Quality:**
```bash
uv run ruff check         # Lint code
uv run ruff format        # Format code
uv run mypy src/          # Type checking
uv run pytest --cov=src/gold_forecasting  # Run tests with coverage
```

**Development Server:**
```bash
uv run uvicorn gold_forecasting.api.main:app --reload  # Start API server
```

## Critical Implementation Rules

### Data Leakage Prevention (NON-NEGOTIABLE)
Always fit StandardScaler inside the rolling window loop - NEVER on the entire dataset:

```python
# CORRECT pattern - inside each CV fold
for i in range(start_idx, len(X)):
    # Split data for this window
    X_train = X[train_start:train_end]
    X_test = X[i:i+1]
    
    # Scale ONLY on current training window
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and predict
    model.fit(X_train_scaled, y_train)
```

### Benchmark MAE Scores to Beat
Each model/window combination must achieve lower MAE than these baselines:
- OLS: 21-day (0.025050), 63-day (0.010710), 126-day (0.008134)
- Ridge: 21-day (0.007527), 63-day (0.006930), 126-day (0.006843)
- Lasso: 21-day (0.007684), 63-day (0.006947), 126-day (0.006826)
- ElasticNet: 21-day (0.007651), 63-day (0.006944), 126-day (0.006824)
- Gaussian Process: 21-day (0.007165), 63-day (0.007213), 126-day (0.007224)
- Random Forest: 21-day (0.007584), 63-day (0.007314), 126-day (0.007172)
- Extra Trees: 21-day (0.007780), 63-day (0.007592), 126-day (0.007089)

## Architecture Overview

### Data Pipeline
- `data_processing/loader.py`: Loads and validates raw CSV data (1987-2025)
- `features/unified_features.py`: **UNIFIED** feature engineering supporting both CPU and GPU pipelines
  - Comprehensive features: lagged, rolling stats, technical indicators, volatility measures
  - Automatic GPU acceleration with graceful CPU fallback
  - Single codebase eliminates duplication between training pipelines
- Configuration in `config.py` using Pydantic BaseSettings

### Model Implementation
- **UNIFIED** training script supporting both CPU and GPU execution (`modeling/unified_train.py`)
- Dynamic model selection: scikit-learn (CPU) vs cuML (GPU) with automatic fallback  
- Rolling forecast function with proper cross-validation and data leakage prevention
- Models: Linear (OLS, Ridge, Lasso, ElasticNet), Ensemble (RF, ExtraTrees), Robust (Huber, Quantile), Bayesian
- GPU acceleration with RAPIDS cuML for 5+ models when available

### API Layer
- FastAPI backend with async endpoints
- WebSocket support for real-time progress updates
- Endpoints: `/forecast`, `/models`, `/compare`, `/metrics/{model}/{window}`

### Key Dependencies
- **Data**: pandas, numpy, cudf-cu12 (GPU DataFrames)
- **ML**: scikit-learn, xgboost[gpu], tensorflow, cuml-cu12, statsmodels, prophet
- **API**: fastapi, uvicorn, pydantic
- **Testing**: pytest, pytest-cov, pytest-asyncio
- **Quality**: ruff, mypy, black, isort

## Testing Strategy

Tests must verify:
1. No data leakage in preprocessing
2. All models beat benchmark MAE scores  
3. Feature engineering produces expected shapes/ranges
4. API endpoints return correct formats
5. Cross-platform compatibility

Use `uv run pytest tests/` with coverage reporting enabled.

## Performance Requirements

Every model/window combination must beat its baseline MAE score. Use TimeSeriesSplit for hyperparameter tuning on first 20% of data. Implement parallel processing with joblib for efficiency.

## Recent Milestones

### ✅ Unified Feature Engineering (Completed)
- **Created** `features/unified_features.py` consolidating CPU and GPU feature generation
- **Eliminated** code duplication between `train.py` and `gpu_train.py` 
- **Added** comprehensive test coverage (15 tests, 82% coverage)
- **Supports** automatic GPU detection with CPU fallback
- **Interface**: `create_features(data, target_col, windows, use_gpu=None)`

### ✅ Unified Training Script (Completed)
- **Created** `modeling/unified_train.py` consolidating CPU and GPU training logic
- **Added** `--use-gpu` flag for dynamic GPU/CPU selection
- **Unified** model configurations with automatic GPU/CPU model mapping
- **Eliminated** redundant code between separate training scripts
- **Added** comprehensive test coverage (12 tests, 70% coverage)
- **Usage**: `uv run python src/gold_forecasting/modeling/unified_train.py [--use-gpu]`