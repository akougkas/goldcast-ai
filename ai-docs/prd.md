# Product Requirements Document (PRD)
## Gold Price Forecasting System

### Executive Summary
This project aims to develop a state-of-the-art gold price forecasting system that outperforms established benchmarks through implementing a comprehensive suite of machine learning models with rolling window forecasts.

### Product Vision
Create a robust, scalable, and accurate gold price prediction system that leverages multiple regression techniques and geopolitical risk indicators to deliver superior forecasting performance.

### Goals & Success Metrics
1. **Primary Goal**: Beat baseline MAE scores across all model types and lookback windows
2. **Secondary Goals**:
   - Provide intuitive visualization of forecast results
   - Enable easy model comparison and performance analysis
   - Create a reproducible and extensible framework

### Success Criteria
- MAE scores lower than baseline for:
  - OLS: 21-day (0.025050), 63-day (0.010710), 126-day (0.008134)
  - Ridge: 21-day (0.007527), 63-day (0.006930), 126-day (0.006843)
  - Lasso: 21-day (0.007684), 63-day (0.006947), 126-day (0.006826)
  - ElasticNet: 21-day (0.007651), 63-day (0.006944), 126-day (0.006824)
  - GP: 21-day (0.007165), 63-day (0.007213), 126-day (0.007224)
  - RF: 21-day (0.007584), 63-day (0.007314), 126-day (0.007172)
  - Extra Trees: 21-day (0.007780), 63-day (0.007592), 126-day (0.007089)
  - XGBoost: 21-day (0.008730), 63-day (0.008306), 126-day (0.008105)

### Key Features
1. **Data Processing Pipeline**
   - Automated data ingestion from CSV
   - Feature engineering with lagged variables
   - Proper train-test splitting for time series

2. **Model Implementation**
   - 9 regression models (OLS, Ridge, Lasso, ElasticNet, Theil-Sen, RANSAC, GP, RF, Extra Trees)
   - 3 lookback windows (21, 63, 126 days)
   - Rolling forecast methodology

3. **Visualization Dashboard**
   - Interactive plots for actual vs predicted values
   - Model performance comparison charts
   - Error distribution analysis
   - Feature importance visualization

4. **Performance Analytics**
   - Real-time MAE calculation
   - Model comparison matrix
   - Performance tracking over time

### Technical Requirements
- **Platform**: Cross-platform (Windows, macOS, Linux)
- **Language**: Python 3.11+
- **Package Manager**: astral uv
- **Frontend**: Web-based dashboard (Django/FastAPI + modern JS framework)
- **Data Storage**: Local filesystem with optional database integration

### User Stories
1. As a data scientist, I want to easily compare model performances to select the best approach
2. As an analyst, I want to visualize forecast results to understand prediction quality
3. As a researcher, I want to reproduce results and extend the framework with new models
4. As a developer, I want clean, modular code that's easy to maintain and extend

### Non-Functional Requirements
- **Performance**: Process full dataset and generate forecasts within 5 minutes
- **Scalability**: Handle datasets up to 50 years of daily data
- **Maintainability**: Modular architecture with clear separation of concerns
- **Documentation**: Comprehensive API docs and user guides
- **Testing**: 90%+ code coverage with unit and integration tests

### Constraints
- Must use astral uv for all Python package management
- Cannot use traditional pip/poetry/conda
- Must beat ALL baseline MAE scores to be considered successful
- Must prevent data leakage in time series validation

### Timeline
- Phase 1: Core implementation (1 week)
- Phase 2: Optimization and tuning (3 days)
- Phase 3: Frontend development (1 week)
- Phase 4: Testing and documentation (3 days)

### Risks & Mitigation
- **Risk**: Models may not beat baseline
  - **Mitigation**: Extensive hyperparameter tuning, feature engineering, ensemble methods
- **Risk**: Computational performance issues with large datasets
  - **Mitigation**: Efficient data structures, parallel processing, caching
- **Risk**: Data leakage in time series validation
  - **Mitigation**: Strict temporal validation, proper scaling inside CV loops