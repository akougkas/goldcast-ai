# ✨ Gold Price Forecasting System

[](https://www.python.org/downloads/)
[](https://github.com/astral-sh/uv)
[](https://www.google.com/search?q=https://github.com/akougkas/goldcast-ai)
[](https://www.google.com/search?q=https://github.com/akougkas/goldcast-ai)
[](https://www.google.com/search?q=LICENSE)

> **Production-Ready ML System** | Beat benchmark accuracy rates | CPU & GPU acceleration

A sophisticated, scalable, and robust machine learning system for gold price forecasting that consistently beats baseline accuracy benchmarks using a unified CPU/GPU training pipeline. This project aims to deliver state-of-the-art forecasting by leveraging multiple regression techniques and geopolitical risk indicators.

-----

## 📜 Table of Contents

  - [🚀 Project Goals](https://www.google.com/search?q=%23-project-goals)
  - [🎯 Performance Benchmarks](https://www.google.com/search?q=%23-performance-benchmarks)
  - [✨ Key Features](https://www.google.com/search?q=%23-key-features)
  - [🛠️ Tech Stack](https://www.google.com/search?q=%23%EF%B8%8F-tech-stack)
  - [🏁 Quick Start](https://www.google.com/search?q=%23-quick-start)
  - [🗂️ Project Structure](https://www.google.com/search?q=%23%EF%B8%8F-project-structure)
  - [👥 Who is this for?](https://www.google.com/search?q=%23-who-is-this-for)
  - [🤝 Contributing](https://www.google.com/search?q=%23-contributing)
  - [📄 License](https://www.google.com/search?q=%23-license)

-----

## 🚀 Project Goals

The primary goal of this system is to develop a state-of-the-art gold price forecasting tool that outperforms established benchmarks.

1.  **Primary Goal**: Beat baseline Mean Absolute Error (MAE) scores across all implemented model types and lookback windows.
2.  **Secondary Goals**:
      - Provide intuitive visualization for forecast results.
      - Enable easy model comparison and performance analysis.
      - Create a reproducible, extensible, and maintainable framework for time-series forecasting.

-----

## 🎯 Performance Benchmarks

The system is engineered to surpass the following baseline MAE scores, ensuring superior forecasting accuracy. The models are considered successful only if they beat **all** relevant baseline scores.

| Model         | 21-Day Window | 63-Day Window | 126-Day Window |
| :------------ | :------------ | :------------ | :------------- |
| **OLS** | 0.025050      | 0.010710      | 0.008134       |
| **Ridge** | 0.007527      | 0.006930      | 0.006843       |
| **Lasso** | 0.007684      | 0.006947      | 0.006826       |
| **ElasticNet**| 0.007651      | 0.006944      | 0.006824       |
| **XGBoost** | 0.008730      | 0.008306      | 0.008105       |

-----

## ✨ Key Features

  - **Data Processing Pipeline**: Automated data ingestion from CSV, feature engineering with lagged variables, and proper train-test splitting for time series to prevent data leakage.
  - **Advanced Model Implementation**: Implements 9 different regression models, including OLS, Ridge, Lasso, ElasticNet, XGBoost, and more.
  - **Flexible Forecasting Windows**: Utilizes a rolling forecast methodology with multiple lookback windows (21, 63, and 126 days) for robust analysis.
  - **Unified Training Pipeline**: A single, clean script to run training on either CPU or GPU.
  - **Performance Analytics**: Real-time MAE calculation and a model comparison matrix to easily evaluate results.
  - **High Code Quality**: A modular architecture with a 90%+ testing coverage goal to ensure reliability and maintainability.

-----

## 🛠️ Tech Stack

This project leverages a modern, performance-oriented tech stack:

  - **Language**: Python 3.11+
  - **Package Manager**: `uv` (from Astral) for ultra-fast dependency management
  - **Core Libraries**: Pandas, Scikit-learn, XGBoost, Statsmodels
  - **GPU Acceleration**: RAPIDS (cuDF, cuML)
  - **Planned Frontend**: Web-based dashboard using FastAPI or Django

-----

## 🏁 Quick Start

Get your first forecast running in just a few steps.

### 1\. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/akougkas/goldcast-ai.git
cd goldcast-ai

# Install dependencies using the uv package manager
uv sync
```

### 2\. Run a Forecast

The `unified_train.py` script provides a single interface for both CPU and GPU training.

```bash
# To run training on CPU (works on any machine)
uv run python src/gold_forecasting/modeling/unified_train.py --data-path data/raw/db_com_capm_gold.csv

# To run training on GPU (requires a RAPIDS-compatible environment)
uv run python src/gold_forecasting/modeling/unified_train.py --use-gpu --data-path data/raw/db_com_capm_gold.csv
```

### 3\. View Results

The system will output a summary of the training process, including:

  - A list of models that successfully beat their baseline accuracy benchmarks.
  - Key performance metrics (MAE, directional accuracy).
  - Confirmation of whether GPU acceleration was used.
  - The final success rate across all models tested.

-----

## 🗂️ Project Structure

The project follows a modular structure to ensure maintainability and scalability.

```
goldcast-ai/
├── ai-docs/                # Documentation (PRD, Spec, etc.)
│   ├── prd.md
│   └── ...
├── data/
│   └── raw/
│       └── db_com_capm_gold.csv
├── src/
│   └── gold_forecasting/
│       ├── data_processing/
│       ├── features/
│       └── modeling/
│           ├── unified_train.py  # Main training script
│           ├── lstm_model.py
│           └── xgboost_model.py
├── tests/
│   ├── unit/
│   └── integration/
├── coverage.xml            # Test coverage report
├── pyproject.toml          # Project config and dependencies
└── README.md
```

-----

## 👥 Who is this for?

This project is designed for:

  - **Data Scientists** who need to rapidly compare and deploy high-performance forecasting models.
  - **Financial Analysts** who require accurate, visualized forecast results to understand prediction quality and inform decisions.
  - **Researchers** who want to reproduce results and extend a robust framework with new models or features.
  - **ML Engineers & Developers** looking for a clean, modular, and well-documented codebase that is easy to maintain and extend.

-----

## 🤝 Contributing

Contributions are welcome\! Please feel free to submit a pull request or open an issue to discuss your ideas. Before contributing, please review the contribution guidelines (link to be added).

-----

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
