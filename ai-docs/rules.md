# Gold Price Forecasting Project Rules

## Project Overview
You are implementing a gold price forecasting system that MUST beat specific benchmark MAE scores using various machine learning models and rolling window predictions.

## Critical Success Criteria
The implementation MUST achieve MAE scores lower than these baselines:
- OLS: 21-day (0.025050), 63-day (0.010710), 126-day (0.008134)
- Ridge: 21-day (0.007527), 63-day (0.006930), 126-day (0.006843)
- Lasso: 21-day (0.007684), 63-day (0.006947), 126-day (0.006826)
- ElasticNet: 21-day (0.007651), 63-day (0.006944), 126-day (0.006824)
- Gaussian Process: 21-day (0.007165), 63-day (0.007213), 126-day (0.007224)
- Random Forest: 21-day (0.007584), 63-day (0.007314), 126-day (0.007172)
- Extra Trees: 21-day (0.007780), 63-day (0.007592), 126-day (0.007089)

## Mandatory Technical Requirements

### Package Management
- Use ONLY `uv` commands (no pip, poetry, or conda)
- Initialize with: `uv init`
- Add dependencies with: `uv add package_name`
- Run scripts with: `uv run python script.py`

### Data Handling Rules
1. **CRITICAL: Prevent Data Leakage**
   - ALWAYS fit StandardScaler inside the rolling window loop
   - NEVER fit scaler on the entire dataset
   - NEVER use future information in predictions

2. **Correct Scaling Pattern**:
   ```python
   # CORRECT - Inside each window iteration
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
   X_test_scaled = scaler.transform(X_test)        # Transform only
   ```

3. **Feature Engineering**
   - Create lagged features for gold prices (1, 2, 3, 5, 10, 20 days)
   - Add rolling statistics (mean, std, min, max)
   - Include technical indicators
   - Create interaction features with GPR indices

### Code Quality Standards
1. **Type Hints**: Use type hints for all functions
2. **Docstrings**: Google-style docstrings for all classes and functions
3. **Error Handling**: Comprehensive try-except blocks with logging
4. **Testing**: Write tests alongside implementation
5. **Formatting**: Use Black and Ruff for consistent style

### Performance Optimization
1. Use NumPy vectorization over loops
2. Implement parallel processing with joblib
3. Cache expensive computations
4. Profile code to identify bottlenecks

### Model Implementation Guidelines
1. Create abstract base class for all models
2. Ensure consistent interface (fit, predict, score)
3. Track performance metrics automatically
4. Support all three lookback windows (21, 63, 126)

### Rolling Forecast Implementation
```python
def rolling_forecast(model, X, y, window_size, start_idx):
    """
    CRITICAL: This is the core pattern that must be followed
    """
    predictions = []
    actuals = []
    
    for i in range(start_idx, len(X)):
        # Define training window
        train_start = max(0, i - window_size)
        train_end = i
        
        # Split data
        X_train = X[train_start:train_end]
        y_train = y[train_start:train_end]
        X_test = X[i:i+1]
        y_test = y[i:i+1]
        
        # CRITICAL: Scale inside the loop
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and predict
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        
        predictions.append(pred[0])
        actuals.append(y_test[0])
    
    return np.array(predictions), np.array(actuals)
```

### Hyperparameter Tuning Strategy
1. Use first 20% of data for parameter search
2. Implement TimeSeriesSplit for validation
3. Try both GridSearchCV and Bayesian optimization
4. Log all experiments and parameters

### Frontend Development
1. Use FastAPI for backend (not Django)
2. Implement WebSocket for real-time updates
3. Create interactive visualizations with Recharts/D3
4. Ensure responsive design for all screen sizes

### Testing Requirements
1. Unit tests for all data transformations
2. Integration tests for full pipeline
3. Performance tests to verify benchmark beating
4. Cross-platform tests (Windows, macOS, Linux)

## Common Pitfalls to Avoid
1. **DON'T** fit preprocessing on test data
2. **DON'T** use simple train_test_split for time series
3. **DON'T** ignore computational efficiency
4. **DON'T** forget to handle edge cases
5. **DON'T** implement without tests

## Development Workflow
1. Implement feature → Write tests → Verify performance
2. Use version control with meaningful commits
3. Document all design decisions
4. Profile before optimizing
5. Validate results against benchmarks frequently

## Success Checklist
Before considering any implementation complete:
- [ ] All models beat their respective baseline MAE scores
- [ ] No data leakage in any part of the pipeline
- [ ] Scaling done correctly inside CV loops
- [ ] All tests passing with >90% coverage
- [ ] Cross-platform compatibility verified
- [ ] API documentation complete
- [ ] Frontend fully functional
- [ ] Performance benchmarks documented

## Remember
This is a competition to beat established benchmarks. Every implementation decision should be made with the goal of achieving the lowest possible MAE scores while maintaining scientific validity and preventing data leakage.