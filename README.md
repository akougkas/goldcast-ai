# <� Gold Price Forecasting System

> **Production-Ready ML System** | Beat benchmark accuracy rates | CPU & GPU acceleration

A sophisticated machine learning system for gold price forecasting that consistently beats baseline accuracy benchmarks using unified CPU/GPU training pipelines.

## =� Quick Start (For Domain Scientists)

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd goldcast-ai

# Install dependencies (uses uv package manager)
uv sync
```

### 2. Run Your First Forecast
```bash
# CPU training (works on any machine)
uv run python src/gold_forecasting/modeling/unified_train.py --data-path data/raw/db_com_capm_gold.csv

# GPU training (if you have RAPIDS installed)
uv run python src/gold_forecasting/modeling/unified_train.py --use-gpu --data-path data/raw/db_com_capm_gold.csv
```

### 3. View Results
The system will output:
-  Models that beat baseline accuracy 
- =� Performance metrics (MAE, directional accuracy)
- =� GPU acceleration status
- =� Success rate across all models

## =� Project Structure

```
goldcast-ai/

