"""
Time Series Forecasting System

A comprehensive forecasting library comparing baseline, statistical, and ML models.
"""

from forecasting.baseline import NaiveForecaster, MeanForecaster, DriftForecaster
from forecasting.statistical import ARIMAForecaster, ProphetForecaster
from forecasting.ml_forecaster import XGBoostForecaster, LSTMForecaster
from forecasting.evaluation import ForecastEvaluator
from forecasting.backtesting import TimeSeriesCrossValidator
from forecasting.data_generator import TimeSeriesGenerator

__version__ = "1.0.0"
__all__ = [
    "NaiveForecaster",
    "MeanForecaster",
    "DriftForecaster",
    "ARIMAForecaster",
    "ProphetForecaster",
    "XGBoostForecaster",
    "LSTMForecaster",
    "ForecastEvaluator",
    "TimeSeriesCrossValidator",
    "TimeSeriesGenerator",
]
