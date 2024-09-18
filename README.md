# Time Series Forecasting System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python library for time series forecasting that compares baseline, statistical, and machine learning models with proper backtesting and evaluation.

## Overview

This project demonstrates best practices in time series forecasting by implementing and comparing multiple forecasting approaches:

- **Baseline Models**: Naive, Mean, and Drift forecasts
- **Statistical Models**: ARIMA and Prophet
- **Machine Learning Models**: XGBoost and LSTM neural networks

## Features

- **Multiple Model Types**: Compare classical statistical methods with modern ML approaches
- **Robust Backtesting**: Time series cross-validation with expanding and rolling windows
- **Comprehensive Evaluation**: 9 different metrics including MAE, RMSE, MAPE, MASE, and R-squared
- **Synthetic Data Generation**: Create realistic time series with trend, seasonality, noise, and changepoints
- **Easy-to-Use API**: Scikit-learn style interface (fit/predict) for all models
- **Visualization**: Built-in plotting functions for forecasts and comparisons

## Project Structure

```
forecasting-system/
├── forecasting/
│   ├── __init__.py           # Package initialization
│   ├── baseline.py           # Naive, Mean, Drift forecasters
│   ├── statistical.py        # ARIMA, Prophet forecasters
│   ├── ml_forecaster.py      # XGBoost, LSTM forecasters
│   ├── evaluation.py         # Evaluation metrics
│   ├── backtesting.py        # Cross-validation framework
│   ├── data_generator.py     # Synthetic data generation
│   └── config.py             # Configuration
├── tests/
│   ├── __init__.py
│   └── test_statistical.py   # Test suite
├── data/
│   ├── create_sample.py      # Sample data generator
│   └── sample_datasets.py    # Embedded sample data
├── notebooks/
│   └── forecasting_demo.ipynb # Jupyter notebook demo
├── main.py                   # Main pipeline script
├── requirements.txt          # Dependencies
├── pytest.ini                # Pytest configuration
├── setup.py                  # Package setup
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Install Dependencies

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For LSTM support, install TensorFlow:

```bash
pip install tensorflow>=2.8.0
```

## Quick Start

### Basic Usage

```python
import pandas as pd
import numpy as np
from forecasting import (
    NaiveForecaster,
    ARIMAForecaster,
    XGBoostForecaster,
    ForecastEvaluator,
)
from forecasting.data_generator import generate_sample_data

# Generate sample data
y = generate_sample_data(n_samples=365, random_seed=42)

# Split into train/test
y_train = y.iloc[:-30]
y_test = y.iloc[-30:]

# Fit and predict with different models
models = {
    "Naive": NaiveForecaster(),
    "ARIMA": ARIMAForecaster(order=(1, 1, 1)),
    "XGBoost": XGBoostForecaster(lags=12, n_estimators=100),
}

predictions = {}
for name, model in models.items():
    model.fit(y_train)
    predictions[name] = model.predict(30)

# Evaluate
evaluator = ForecastEvaluator()
comparison = evaluator.compare_models(y_test.values, predictions)
print(comparison)
```

### Running the Full Pipeline

```bash
python main.py --test-size 30 --n-splits 5 --horizon 14
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data` | Data source ('sample' or CSV path) | 'sample' |
| `--test-size` | Number of test observations | 30 |
| `--n-splits` | Number of CV splits | 5 |
| `--horizon` | Forecast horizon | 14 |
| `--output-dir` | Output directory | 'output' |
| `--no-plot` | Disable plotting | False |

## Models

### Baseline Models

#### Naive Forecaster
Uses the last observed value (or seasonal value) for all future predictions.

```python
from forecasting import NaiveForecaster

# Random walk forecast
model = NaiveForecaster(strategy="last")

# Seasonal naive (e.g., use same day last week)
model = NaiveForecaster(strategy="seasonal", seasonality=7)
```

#### Mean Forecaster
Predicts the historical mean (or rolling window mean).

```python
from forecasting import MeanForecaster

# Full history mean
model = MeanForecaster()

# Rolling window mean
model = MeanForecaster(window=30)
```

#### Drift Forecaster
Extends the last value with the average historical trend.

```python
from forecasting import DriftForecaster

model = DriftForecaster()
```

### Statistical Models

#### ARIMA
AutoRegressive Integrated Moving Average model.

```python
from forecasting import ARIMAForecaster

# Simple ARIMA
model = ARIMAForecaster(order=(1, 1, 1))

# Seasonal ARIMA (SARIMA)
model = ARIMAForecaster(
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)  # Monthly seasonality
)

# Get forecasts with confidence intervals
mean, lower, upper = model.predict_with_ci(30)
```

#### Prophet
Facebook's Prophet for time series with trends and seasonality.

```python
from forecasting import ProphetForecaster

model = ProphetForecaster(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05
)

# Prophet requires DatetimeIndex
model.fit(y)  # y must be pd.Series with DatetimeIndex
```

### Machine Learning Models

#### XGBoost
Gradient boosting with lagged features.

```python
from forecasting import XGBoostForecaster

model = XGBoostForecaster(
    lags=12,              # Number of lagged observations
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    include_trend=True    # Add time index feature
)

# Get feature importance
importance = model.get_feature_importance()
```

#### LSTM
Long Short-Term Memory neural network.

```python
from forecasting import LSTMForecaster

model = LSTMForecaster(
    lags=12,
    hidden_units=[50, 25],  # Stacked LSTM layers
    epochs=100,
    batch_size=32,
    dropout=0.2
)
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average magnitude of errors |
| **MSE** | Mean Squared Error | Penalizes larger errors |
| **RMSE** | Root Mean Squared Error | In same units as data |
| **MAPE** | Mean Absolute Percentage Error | Error as percentage |
| **sMAPE** | Symmetric MAPE | Handles zero values better |
| **MASE** | Mean Absolute Scaled Error | Scaled by naive forecast |
| **R²** | Coefficient of Determination | Variance explained |
| **Theil's U** | Relative to naive | <1 better than naive |
| **Bias** | Mean Bias Error | Over/under-forecasting |

### Using Evaluation

```python
from forecasting.evaluation import ForecastEvaluator, evaluate

# Single evaluation
metrics = evaluate(y_true, y_pred, metrics=["mae", "rmse", "mape"])

# Compare multiple models
evaluator = ForecastEvaluator(metrics=["mae", "rmse", "mase"])
comparison = evaluator.compare_models(
    y_test,
    {"Naive": naive_pred, "ARIMA": arima_pred},
    y_train=y_train  # For MASE calculation
)
```

## Backtesting

Time series cross-validation ensures robust model evaluation.

### Expanding Window

Training set grows with each fold:

```python
from forecasting.backtesting import TimeSeriesCrossValidator

cv = TimeSeriesCrossValidator(
    n_splits=5,
    horizon=10,
    gap=0,
    expanding=True
)

for train_idx, test_idx in cv.split(y):
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    # ... fit and evaluate
```

### Rolling Window

Fixed training set size:

```python
cv = TimeSeriesCrossValidator(
    n_splits=5,
    horizon=10,
    min_train_size=100,
    expanding=False  # Rolling window
)
```

### Quick Backtest

```python
from forecasting.backtesting import run_backtest

results = run_backtest(
    model,
    y,
    n_splits=5,
    horizon=14,
    metrics=["mae", "rmse"]
)

print(f"Mean RMSE: {results['mean_metrics']['rmse']:.4f}")
```

## Data Generation

Create synthetic time series for testing:

```python
from forecasting.data_generator import TimeSeriesGenerator

gen = TimeSeriesGenerator(
    n_samples=365,
    start_date="2023-01-01",
    freq="D",
    random_seed=42
)

y = gen.generate(
    trend="linear",           # 'linear', 'exponential', 'polynomial'
    trend_strength=0.5,
    seasonality=[7, 365],     # Weekly and yearly
    seasonality_strength=0.3,
    noise=True,
    noise_std=0.02,
    outliers=True,            # Add some outliers
    changepoints=True,        # Add trend changepoints
)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=forecasting --cov-report=html

# Run specific test file
pytest tests/test_statistical.py -v

# Skip slow tests
pytest tests/ -v -m "not slow"
```

## Examples

### Compare All Models

```python
from forecasting import (
    NaiveForecaster, MeanForecaster, DriftForecaster,
    ARIMAForecaster, XGBoostForecaster
)
from forecasting.evaluation import ForecastEvaluator
from forecasting.data_generator import generate_sample_data

# Generate data
y = generate_sample_data(n_samples=500, random_seed=42)
y_train, y_test = y.iloc[:-50], y.iloc[-50:]

# Define models
models = {
    "Naive": NaiveForecaster(),
    "Mean": MeanForecaster(),
    "Drift": DriftForecaster(),
    "ARIMA": ARIMAForecaster(order=(1, 1, 1)),
    "XGBoost": XGBoostForecaster(lags=12),
}

# Fit, predict, and compare
predictions = {}
for name, model in models.items():
    model.fit(y_train)
    predictions[name] = model.predict(50)

evaluator = ForecastEvaluator()
comparison = evaluator.compare_models(y_test.values, predictions)
print(comparison.round(4))
```

### Custom Model Comparison

```python
from forecasting.backtesting import BacktestEngine, TimeSeriesCrossValidator

# Setup
cv = TimeSeriesCrossValidator(n_splits=10, horizon=7)
engine = BacktestEngine(cv, metrics=["mae", "rmse", "mase"])

# Compare models
models = {
    "Naive": NaiveForecaster(),
    "SNaive": NaiveForecaster(strategy="seasonal", seasonality=7),
    "Mean": MeanForecaster(window=30),
}

results = engine.compare_models(models, y)
print(results)
```

## Model Selection Guidelines

| Data Pattern | Recommended Models |
|--------------|-------------------|
| Stationary | Mean, ARIMA |
| Trend | Drift, Prophet, XGBoost with trend |
| Seasonal | Prophet, Seasonal Naive, SARIMA |
| Complex patterns | XGBoost, LSTM |
| Limited history | Naive, Mean |
| Many series | Naive (as benchmark), XGBoost |

## Best Practices

1. **Always use baselines**: Compare sophisticated models against naive/mean forecasts
2. **Use cross-validation**: Single train/test splits can be misleading
3. **Check MASE**: Values > 1 indicate the model is worse than naive
4. **Consider bias**: Positive bias = under-forecasting, negative = over-forecasting
5. **Respect temporal order**: Never use future data in training

## Dependencies

- **Required**: numpy, pandas, scikit-learn
- **Statistical Models**: statsmodels, prophet
- **ML Models**: xgboost, tensorflow (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License

## References

- Hyndman, R.J., & Athanasopoulos, G. (2021) *Forecasting: Principles and Practice*, 3rd edition
- Taylor, S.J., & Letham, B. (2018) *Forecasting at Scale*, The American Statistician
- Box, G.E.P., & Jenkins, G.M. (1976) *Time Series Analysis: Forecasting and Control*
