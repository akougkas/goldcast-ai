# Implementation Plan for Gold Price Forecasting System

## Phase 1: Project Foundation (Milestone: Working Development Environment)

### 1. **Task: Initialize Project with uv**
- **Objective:** Set up a modern Python project using astral uv
- **Critical Instructions:** 
  - Run `uv init gold-forecasting --package`
  - Create the directory structure as specified in spec.md
  - Configure pyproject.toml with all required dependencies
  - Ensure cross-platform compatibility settings

### 2. **Task: Implement Configuration Management**
- **Objective:** Create a robust configuration system for the project
- **Critical Instructions:**
  - Create `src/gold_forecasting/config.py` with pydantic BaseSettings
  - Include all model hyperparameters, data paths, and API settings
  - Support environment variable overrides
  - Implement separate configs for development/testing/production

### 3. **Task: Set Up Logging Infrastructure**
- **Objective:** Implement structured logging throughout the application
- **Critical Instructions:**
  - Use structlog for structured logging
  - Create logging utilities in `src/gold_forecasting/utils/logging.py`
  - Include correlation IDs for request tracking
  - Set up different log levels for different environments

## Phase 2: Data Pipeline (Milestone: Clean, Feature-Rich Dataset)

### 4. **Task: Implement Data Loader**
- **Objective:** Create robust data loading functionality
- **Critical Instructions:**
  - Load `db_com_capm_gold.csv` with proper date parsing
  - Validate data integrity (no missing values in critical columns)
  - Create data quality report showing basic statistics
  - Handle timezone considerations if any
  - Verify date range: 1987-01-01 to 2025-05-28

### 5. **Task: Implement Feature Engineering Pipeline**
- **Objective:** Create comprehensive features for time series modeling
- **Critical Instructions:**
  - **Lagged Features**: Create lags for gold prices (1, 2, 3, 5, 10, 20 days)
  - **Rolling Statistics**: Implement rolling windows (5, 10, 20, 60 days) for:
    - Mean, Standard Deviation, Min, Max
    - Skewness and Kurtosis
  - **Technical Indicators**:
    - Simple Moving Averages (SMA)
    - Exponential Moving Averages (EMA)
    - Relative Strength Index (RSI)
    - Bollinger Bands
  - **GPR Interaction Features**:
    - Gold price changes × GPR indices
    - Ratios of gold to GPR indices
  - **Calendar Features**: Day of week, month, quarter effects

### 6. **Task: Implement Data Validation Framework**
- **Objective:** Ensure data integrity throughout the pipeline
- **Critical Instructions:**
  - Check for temporal consistency (no future data leakage)
  - Validate feature ranges are reasonable
  - Implement automated data quality checks
  - Create visualization tools for data inspection

## Phase 3: Model Implementation (Milestone: All Models Functional)

### 7. **Task: Create Abstract Base Model Class**
- **Objective:** Establish consistent interface for all models
- **Critical Instructions:**
  - Define abstract methods: fit(), predict(), get_params(), set_params()
  - Include automatic performance tracking
  - Support for different lookback windows
  - Built-in cross-validation support

### 8. **Task: Implement Linear Models**
- **Objective:** Create OLS, Ridge, Lasso, and ElasticNet implementations
- **Critical Instructions:**
  - Use statsmodels for OLS implementation
  - Use scikit-learn for Ridge, Lasso, ElasticNet
  - Implement proper parameter ranges for regularization
  - Add coefficient extraction for interpretability

### 9. **Task: Implement Robust Models**
- **Objective:** Create Theil-Sen and RANSAC implementations
- **Critical Instructions:**
  - Use scikit-learn's TheilSenRegressor and RANSACRegressor
  - Configure appropriate hyperparameters for financial data
  - Implement outlier detection and reporting

### 10. **Task: Implement Non-linear Models**
- **Objective:** Create Gaussian Process, Random Forest, and Extra Trees
- **Critical Instructions:**
  - For GP: Start with RBF kernel, implement proper length scale bounds
  - For RF/ET: Configure appropriate tree depths and ensemble sizes
  - Extract and store feature importances
  - Implement efficient prediction methods

### 11. **Task: Implement the Rolling Forecast Function with Correct Scaling**
- **Objective:** Create the core rolling forecast engine
- **Critical Instructions for Coder:** 
  - The function must accept a model, the data, and a window size as input
  - **CRITICAL**: Inside the function's loop, you MUST initialize a `StandardScaler` and fit it ONLY on the current training window's data
  - Then, use this fitted scaler to transform both the training and test data for that window before fitting the model
  - This prevents data leakage and is NON-NEGOTIABLE
  - Implement parallel processing support using joblib
  - Add progress tracking and logging

## Phase 4: Hyperparameter Optimization (Milestone: Optimized Models)

### 12. **Task: Implement Hyperparameter Tuning Framework**
- **Objective:** Systematic parameter optimization for all models
- **Critical Instructions:**
  - Use TimeSeriesSplit for temporal cross-validation
  - Implement both GridSearchCV and Optuna for Bayesian optimization
  - Use first 20% of data for tuning to prevent overfitting
  - Log all experiments with parameters and scores

### 13. **Task: Model-Specific Hyperparameter Optimization**
- **Objective:** Find optimal parameters for each model type
- **Critical Instructions:**
  - **Ridge/Lasso**: Alpha from 1e-4 to 1e2 (log scale)
  - **ElasticNet**: Alpha (same as above) + l1_ratio from 0.1 to 0.9
  - **Random Forest**: n_estimators (100-500), max_depth (5-20), min_samples_split (2-20)
  - **Gaussian Process**: Kernel length scales, noise levels
  - **RANSAC**: Residual threshold based on data statistics
  - Store best parameters in configuration

## Phase 5: Evaluation & Validation (Milestone: Benchmark-Beating Performance)

### 14. **Task: Implement Comprehensive Evaluation Metrics**
- **Objective:** Create evaluation framework to track performance
- **Critical Instructions:**
  - Implement MAE as primary metric
  - Add RMSE, MAPE, directional accuracy as secondary metrics
  - Create comparison tables with baseline scores
  - Implement statistical significance tests

### 15. **Task: Run Full Backtesting Suite**
- **Objective:** Validate all models meet success criteria
- **Critical Instructions:**
  - Run all models with all window sizes (21, 63, 126 days)
  - Generate detailed performance reports
  - Create visualizations of predictions vs actuals
  - Verify ALL models beat their respective baselines
  - If any model fails, implement improvements

## Phase 6: API Development (Milestone: Functional REST API)

### 16. **Task: Implement FastAPI Backend**
- **Objective:** Create RESTful API for model serving
- **Critical Instructions:**
  - Set up FastAPI application with proper project structure
  - Implement endpoints:
    - POST `/forecast` - Run prediction with specified model/window
    - GET `/models` - List available models and their performance
    - POST `/compare` - Compare multiple models
    - GET `/metrics/{model}/{window}` - Get detailed metrics
  - Add request validation with Pydantic
  - Implement async request handling

### 17. **Task: Add WebSocket Support for Real-time Updates**
- **Objective:** Enable real-time progress tracking
- **Critical Instructions:**
  - Implement WebSocket endpoint for live updates
  - Send progress updates during long-running forecasts
  - Handle connection management and error recovery

## Phase 7: Frontend Development (Milestone: Interactive Dashboard)

### 18. **Task: Set Up Next.js Frontend Application**
- **Objective:** Create modern web interface
- **Critical Instructions:**
  - Initialize Next.js 14+ project with TypeScript
  - Set up Tailwind CSS and shadcn/ui components
  - Configure API client with proper error handling
  - Implement responsive layout

### 19. **Task: Implement Visualization Components**
- **Objective:** Create interactive charts for results analysis
- **Critical Instructions:**
  - Time series plot with actual vs predicted values
  - Model comparison dashboard with MAE scores
  - Feature importance visualization
  - Error distribution histograms
  - Performance metrics table with sorting/filtering

### 20. **Task: Add Interactive Model Selection and Configuration**
- **Objective:** User-friendly interface for running forecasts
- **Critical Instructions:**
  - Model selection dropdown with descriptions
  - Window size selector (21, 63, 126 days)
  - Date range picker for backtesting period
  - Real-time progress indicator during computation

## Phase 8: Testing & Documentation (Milestone: Production-Ready System)

### 21. **Task: Implement Comprehensive Test Suite**
- **Objective:** Ensure code reliability and correctness
- **Critical Instructions:**
  - Unit tests for all data transformations and features
  - Integration tests for full pipeline runs
  - Performance tests to verify benchmark beating
  - API endpoint tests with various scenarios
  - Frontend component tests

### 22. **Task: Create User and Developer Documentation**
- **Objective:** Make the system easy to use and extend
- **Critical Instructions:**
  - API documentation (auto-generated with FastAPI)
  - User guide with screenshots and examples
  - Developer guide with architecture details
  - Performance report comparing to baselines
  - README with quick start instructions

## Phase 9: Final Validation (Milestone: All Benchmarks Beaten)

### 23. **Task: Final Performance Validation**
- **Objective:** Confirm all success criteria are met
- **Critical Instructions:**
  - Run final backtests for all model/window combinations
  - Generate comprehensive performance report
  - Verify every model beats its baseline MAE
  - Create visualization showing improvement over baselines
  - Document any special techniques used to beat benchmarks

### 24. **Task: Cross-Platform Testing**
- **Objective:** Ensure system works on all platforms
- **Critical Instructions:**
  - Test on Windows, macOS, and Linux
  - Verify uv installation process works smoothly
  - Check file path handling is OS-agnostic
  - Test frontend in multiple browsers

## Success Verification Checklist

Before considering the project complete, verify:
- [ ] Project installs and runs with just `uv` commands
- [ ] All 9 models implemented and functional
- [ ] Every model/window combination beats baseline MAE
- [ ] No data leakage (scaling done inside CV loops)
- [ ] API fully functional with all endpoints
- [ ] Frontend dashboard working and responsive
- [ ] All tests passing with >90% coverage
- [ ] Documentation complete and accurate
- [ ] Cross-platform compatibility confirmed

## Critical Implementation Notes

1. **Data Leakage Prevention is PARAMOUNT** - Always fit preprocessing inside the cross-validation loop
2. **Performance is the PRIMARY GOAL** - Every decision should aim to beat baselines
3. **Use Modern Tools** - Leverage uv, FastAPI, and Next.js capabilities fully
4. **Test Continuously** - Verify performance after each major change
5. **Document Decisions** - Explain why specific approaches were chosen