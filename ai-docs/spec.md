# Technical Specification
## Gold Price Forecasting System

### Technology Stack

#### Core Development
- **Python**: 3.11+ (managed by uv)
- **Package Manager**: astral uv (latest version)
- **Virtual Environment**: Managed automatically by uv

#### Data Science Stack
```toml
# pyproject.toml dependencies
[project]
dependencies = [
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "scikit-learn>=1.4.0",
    "statsmodels>=0.14.0",  # For OLS
    "joblib>=1.3.0",  # For parallel processing
    "numba>=0.59.0",  # For performance optimization
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-asyncio>=0.23.0",
    "black>=24.0.0",
    "ruff>=0.3.0",
    "mypy>=1.9.0",
    "pre-commit>=3.6.0",
]
```

#### Backend Framework
- **FastAPI**: 0.110.0+ (async, modern, automatic OpenAPI)
- **Pydantic**: 2.6.0+ (data validation)
- **Uvicorn**: 0.27.0+ (ASGI server)

#### Frontend Stack
- **Framework**: Next.js 14+ with TypeScript
- **UI Library**: Tailwind CSS + shadcn/ui
- **Charts**: Recharts + D3.js for custom visualizations
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)

#### Development Tools
- **Testing**: pytest, pytest-asyncio, pytest-cov
- **Linting**: Ruff (fast, comprehensive)
- **Formatting**: Black + Ruff formatter
- **Type Checking**: mypy with strict mode
- **Pre-commit**: Automated code quality checks

### Project Structure
```
gold-forecasting/
├── pyproject.toml          # uv project configuration
├── uv.lock                 # Lock file for reproducibility
├── .python-version         # Python version pin
├── README.md
├── .cursorrules           # IDE agent rules
├── src/
│   └── gold_forecasting/
│       ├── __init__.py
│       ├── config.py      # Configuration management
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loader.py  # Data loading utilities
│       │   ├── features.py # Feature engineering
│       │   └── validation.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py    # Abstract base model
│       │   ├── linear.py  # OLS, Ridge, Lasso, ElasticNet
│       │   ├── robust.py  # Theil-Sen, RANSAC
│       │   ├── ensemble.py # RF, Extra Trees
│       │   └── gaussian.py # Gaussian Process
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── metrics.py
│       │   └── backtesting.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py    # FastAPI app
│       │   ├── routes/
│       │   └── schemas/
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── caching.py
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   ├── lib/
│   │   └── types/
│   └── public/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── scripts/
│   ├── run_backtest.py
│   └── generate_report.py
└── data/
    └── raw/
        └── db_com_capm_gold.csv
```

### Architecture Decisions

#### 1. Data Pipeline Architecture
- **Immutable Data Flow**: Raw data → Features → Scaled Features → Predictions
- **Lazy Loading**: Use generators for large dataset processing
- **Caching Strategy**: 
  - Feature matrices cached with joblib
  - Model predictions cached in memory with TTL
  - Use functools.lru_cache for expensive computations

#### 2. Model Training Pipeline
```python
# Pseudocode for correct implementation
class RollingForecaster:
    def __init__(self, model, window_size: int):
        self.model = model
        self.window_size = window_size
        
    def forecast(self, X, y, start_date):
        predictions = []
        
        for i in range(start_date, len(X)):
            # Extract training window
            train_start = max(0, i - self.window_size)
            X_train = X[train_start:i]
            y_train = y[train_start:i]
            X_test = X[i:i+1]
            
            # CRITICAL: Fit scaler on training data only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and predict
            self.model.fit(X_train_scaled, y_train)
            pred = self.model.predict(X_test_scaled)
            predictions.append(pred[0])
            
        return np.array(predictions)
```

#### 3. API Design Principles
- **RESTful with async/await**: All endpoints are async
- **Pydantic Models**: Strong typing for request/response
- **Background Tasks**: Long-running forecasts use background tasks
- **WebSocket Support**: Real-time progress updates

#### 4. Frontend Architecture
- **Server-Side Rendering**: Next.js App Router for SEO and performance
- **Component Library**: Reusable chart components
- **Real-time Updates**: WebSocket connection for live data
- **Responsive Design**: Mobile-first approach

### Performance Optimizations

#### 1. Computational Efficiency
- **Vectorization**: NumPy operations wherever possible
- **Parallel Processing**: joblib.Parallel for independent computations
- **Numba JIT**: For critical inner loops
- **Sparse Matrices**: For high-dimensional feature spaces

#### 2. Memory Management
- **Chunked Processing**: Process large datasets in chunks
- **Garbage Collection**: Explicit cleanup after large operations
- **Memory Profiling**: Use memory_profiler in development

#### 3. Caching Strategy
```python
from functools import lru_cache
from joblib import Memory

# Disk-based caching for expensive operations
memory = Memory('cache_directory', verbose=0)

@memory.cache
def compute_features(df, window_sizes):
    # Expensive feature computation
    pass

# In-memory caching for frequently accessed data
@lru_cache(maxsize=128)
def get_model_predictions(model_name, window_size):
    pass
```

### Error Handling & Logging

#### Structured Logging
```python
import structlog

logger = structlog.get_logger()

logger.info("forecast_started", 
    model=model_name, 
    window_size=window_size,
    data_points=len(data))
```

#### Error Recovery
- Graceful degradation for missing data
- Automatic retries for transient failures
- Comprehensive error messages for debugging

### Testing Strategy

#### 1. Unit Tests
- Test each component in isolation
- Mock external dependencies
- Parametrized tests for multiple scenarios

#### 2. Integration Tests
- Full pipeline tests with sample data
- API endpoint tests
- Cross-model consistency checks

#### 3. Performance Tests
- Benchmark against baseline MAE scores
- Load testing for API endpoints
- Memory usage profiling

### Security Considerations
- Input validation on all user inputs
- Rate limiting on API endpoints
- CORS configuration for frontend
- Environment variables for sensitive config

### Deployment Configuration

#### Development
```bash
# Initialize project
uv init gold-forecasting
cd gold-forecasting

# Install dependencies
uv sync

# Run development servers
uv run uvicorn src.gold_forecasting.api.main:app --reload
cd frontend && npm run dev
```

#### Production
```bash
# Build frontend
cd frontend && npm run build

# Run production server
uv run uvicorn src.gold_forecasting.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
```

### Cross-Platform Compatibility
- Use pathlib.Path for all file operations
- Avoid platform-specific dependencies
- Test on Windows, macOS, and Linux CI/CD
- Use os-agnostic path separators

### Monitoring & Observability
- Prometheus metrics for API performance
- Custom metrics for model performance
- Health check endpoints
- Structured logging with correlation IDs

This specification ensures a robust, scalable, and maintainable system that can beat the baseline benchmarks while providing a great developer and user experience.