# Gold Price Forecasting Project Plan

## Project Overview
This project implements a comprehensive gold price forecasting system using multiple machine learning models with rolling window predictions. The goal is to outperform established benchmark MAE scores by leveraging geopolitical risk indicators and advanced feature engineering.

## Phase 1: Foundation & Data Pipeline (Days 1-3)

### 1.1 Project Setup
- **Task**: Initialize project with astral uv
- **Objective**: Create a modern Python project structure using uv
- **Deliverables**:
  - `uv init` project with proper pyproject.toml
  - Directory structure for modular development
  - Initial dependencies specification

### 1.2 Data Loading & Exploration
- **Task**: Implement robust data loading pipeline
- **Objective**: Load and validate the gold price dataset
- **Key Requirements**:
  - Parse dates correctly (time column)
  - Handle missing values appropriately
  - Create initial data quality report
  - Verify data range (1987-01-01 to 2025-05-28)

### 1.3 Feature Engineering Pipeline
- **Task**: Create comprehensive feature engineering module
- **Objective**: Generate time-series features for modeling
- **Critical Features**:
  - Lagged gold prices (1, 2, 3, 5, 10, 20 days)
  - Rolling statistics (mean, std, min, max) for various windows
  - Technical indicators (moving averages, volatility)
  - Interaction features between gold and GPR indices
  - Date-based features (day of week, month, quarter effects)

### 1.4 Data Validation Framework
- **Task**: Implement data quality checks
- **Objective**: Ensure data integrity throughout pipeline
- **Checks**:
  - Temporal consistency
  - Feature range validation
  - Missing value detection
  - Outlier identification

## Phase 2: Model Implementation (Days 4-7)

### 2.1 Base Model Framework
- **Task**: Create abstract base class for all models
- **Objective**: Ensure consistent interface across models
- **Requirements**:
  - Standardized fit/predict methods
  - Built-in performance tracking
  - Support for different lookback windows

### 2.2 Linear Models Implementation
- **Models**: OLS, Ridge, Lasso, ElasticNet
- **Key Considerations**:
  - Proper regularization parameter ranges
  - Standardization inside CV loop
  - Coefficient tracking for interpretability

### 2.3 Robust Models Implementation
- **Models**: Theil-Sen, RANSAC
- **Key Considerations**:
  - Outlier handling strategies
  - Computational efficiency for large datasets
  - Stability across different market conditions

### 2.4 Non-linear Models Implementation
- **Models**: Gaussian Process, Random Forest, Extra Trees
- **Key Considerations**:
  - Kernel selection for GP (start with RBF)
  - Tree depth and ensemble size optimization
  - Feature importance extraction

### 2.5 Rolling Forecast Engine
- **Task**: Implement the core rolling forecast logic
- **Critical Requirements**:
  - **MUST** fit StandardScaler only on training data within each window
  - Proper temporal train/test splitting
  - Efficient window sliding mechanism
  - Parallel processing support

## Phase 3: Hyperparameter Optimization (Days 8-10)

### 3.1 Tuning Framework
- **Task**: Implement systematic hyperparameter search
- **Approach**:
  - Use initial 20% of data for tuning
  - Implement both grid search and Bayesian optimization
  - Track all experiments with MLflow or similar

### 3.2 Model-Specific Tuning
- **Ridge/Lasso**: Alpha parameter optimization
- **ElasticNet**: Alpha and l1_ratio optimization
- **Random Forest**: n_estimators, max_depth, min_samples_split
- **GP**: Kernel parameters, noise level
- **RANSAC**: Residual threshold, max trials

### 3.3 Ensemble Strategy
- **Task**: Explore ensemble methods to beat benchmarks
- **Options**:
  - Simple averaging
  - Weighted averaging based on recent performance
  - Stacking with meta-learner

## Phase 4: Visualization & Frontend (Days 11-17)

### 4.1 Backend API Development
- **Framework**: FastAPI (modern, async, automatic API docs)
- **Endpoints**:
  - `/forecast`: Run predictions with specified model/window
  - `/compare`: Compare multiple models
  - `/metrics`: Get performance metrics
  - `/visualize`: Generate plot data

### 4.2 Frontend Dashboard
- **Framework**: React/Next.js with Recharts/D3.js
- **Key Views**:
  - Model comparison dashboard
  - Time series visualization with predictions
  - Performance metrics table
  - Feature importance charts
  - Error distribution analysis

### 4.3 Real-time Updates
- **WebSocket integration for live updates**
- **Progress tracking for long-running forecasts**
- **Result caching for performance**

## Phase 5: Testing & Documentation (Days 18-20)

### 5.1 Testing Strategy
- **Unit Tests**: Core functions and data transformations
- **Integration Tests**: Full pipeline runs
- **Performance Tests**: Benchmark beating validation
- **Cross-platform Tests**: Windows, macOS, Linux

### 5.2 Documentation
- **API Documentation**: Auto-generated from FastAPI
- **User Guide**: Step-by-step usage instructions
- **Developer Guide**: Architecture and extension points
- **Performance Report**: Detailed comparison with baselines

## Success Validation Checklist

### Data Integrity
- [ ] No data leakage in any model
- [ ] Proper temporal validation
- [ ] Scaling done correctly within CV loops

### Performance Targets
- [ ] All models beat baseline MAE scores
- [ ] Results are reproducible
- [ ] Performance is stable across different data periods

### Code Quality
- [ ] Modular, extensible architecture
- [ ] Comprehensive error handling
- [ ] Type hints throughout
- [ ] >90% test coverage

### Deployment
- [ ] Cross-platform compatibility verified
- [ ] uv-based installation works smoothly
- [ ] Frontend accessible and responsive
- [ ] Documentation complete and clear

## Risk Mitigation Strategies

### If Models Don't Beat Baseline:
1. **Enhanced Feature Engineering**:
   - Add more sophisticated technical indicators
   - Explore feature interactions
   - Use automated feature selection

2. **Advanced Techniques**:
   - Implement online learning for adaptation
   - Try neural network approaches (LSTM, Transformer)
   - Explore regime-switching models

3. **Data Augmentation**:
   - Incorporate additional economic indicators
   - Use synthetic data generation for rare events
   - Apply advanced preprocessing (wavelets, EMD)

### Performance Optimization:
- Implement caching for repeated calculations
- Use joblib for parallel processing
- Profile and optimize bottlenecks
- Consider Numba/Cython for critical loops

## Key Implementation Notes

### Critical: Preventing Data Leakage
```python
# CORRECT approach - fit scaler inside loop
for train_idx, test_idx in time_series_split:
    X_train, X_test = X[train_idx], X[test_idx]
    
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Only transform
    
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
```

### Window Specifications
- **21 days**: ~1 trading month (short-term trends)
- **63 days**: ~3 trading months (quarterly effects)
- **126 days**: ~6 trading months (semi-annual patterns)

This plan ensures systematic development while maintaining focus on beating the baseline benchmarks through careful implementation and rigorous validation.