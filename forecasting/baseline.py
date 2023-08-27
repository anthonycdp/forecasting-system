"""
Baseline forecasting models for time series analysis.

These models serve as simple benchmarks that more sophisticated models
should outperform to justify their complexity.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self):
        self._fitted = False
        self._y_train: Optional[np.ndarray] = None
        self._last_date: Optional[pd.Timestamp] = None
        self._freq: Optional[str] = None

    @abstractmethod
    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "BaseForecaster":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts for the specified number of steps."""
        pass

    def _validate_input(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Validate and convert input to numpy array."""
        if isinstance(y, pd.Series):
            if y.empty:
                raise ValueError("Input series cannot be empty")
            return y.values
        elif isinstance(y, np.ndarray):
            if y.size == 0:
                raise ValueError("Input array cannot be empty")
            return y
        else:
            raise TypeError(f"Expected pd.Series or np.ndarray, got {type(y)}")

    def _extract_metadata(self, y: Union[pd.Series, np.ndarray]) -> None:
        """Extract metadata from pandas Series if available."""
        if isinstance(y, pd.Series):
            if isinstance(y.index, pd.DatetimeIndex):
                self._last_date = y.index[-1]
                self._freq = y.index.freqstr if y.index.freqstr else pd.infer_freq(y.index)

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._fitted


class NaiveForecaster(BaseForecaster):
    """
    Naive forecasting model.

    Predicts the last observed value for all future steps.
    This is the simplest possible forecast and serves as a baseline.

    Parameters
    ----------
    strategy : str, default='last'
        The naive strategy to use:
        - 'last': Use the last observed value (random walk forecast)
        - 'seasonal': Use value from same period in previous cycle

    seasonality : int, optional
        The seasonal period for 'seasonal' strategy.
        For example, 12 for monthly data with yearly seasonality.

    Examples
    --------
    >>> import pandas as pd
    >>> from forecasting.baseline import NaiveForecaster
    >>> y = pd.Series([1, 2, 3, 4, 5])
    >>> model = NaiveForecaster()
    >>> model.fit(y)
    >>> model.predict(3)
    array([5., 5., 5.])
    """

    def __init__(self, strategy: str = "last", seasonality: Optional[int] = None):
        super().__init__()
        if strategy not in ["last", "seasonal"]:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'last' or 'seasonal'.")
        if strategy == "seasonal" and seasonality is None:
            raise ValueError("seasonality must be provided for 'seasonal' strategy")
        self.strategy = strategy
        self.seasonality = seasonality

    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "NaiveForecaster":
        """Fit the naive model by storing the training data."""
        y_arr = self._validate_input(y)
        self._extract_metadata(y)
        self._y_train = y_arr
        self._fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Generate naive forecasts."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        if self.strategy == "last":
            return np.full(steps, self._y_train[-1])
        else:  # seasonal
            forecasts = np.zeros(steps)
            for i in range(steps):
                seasonal_idx = -(self.seasonality - (i % self.seasonality))
                if abs(seasonal_idx) > len(self._y_train):
                    # Fall back to last value if not enough history
                    forecasts[i] = self._y_train[-1]
                else:
                    forecasts[i] = self._y_train[seasonal_idx]
            return forecasts


class MeanForecaster(BaseForecaster):
    """
    Mean forecasting model.

    Predicts the historical mean for all future steps.
    Useful for stationary time series.

    Parameters
    ----------
    window : int, optional
        If provided, use a rolling window of the last `window` observations
        to compute the mean instead of the full history.

    Examples
    --------
    >>> import pandas as pd
    >>> from forecasting.baseline import MeanForecaster
    >>> y = pd.Series([1, 2, 3, 4, 5])
    >>> model = MeanForecaster()
    >>> model.fit(y)
    >>> model.predict(3)
    array([3., 3., 3.])
    """

    def __init__(self, window: Optional[int] = None):
        super().__init__()
        if window is not None and window < 1:
            raise ValueError("window must be a positive integer")
        self.window = window
        self._mean_value: Optional[float] = None

    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "MeanForecaster":
        """Fit the mean model by computing the historical mean."""
        y_arr = self._validate_input(y)
        self._extract_metadata(y)
        self._y_train = y_arr

        if self.window is not None:
            self._mean_value = np.mean(y_arr[-self.window :])
        else:
            self._mean_value = np.mean(y_arr)

        self._fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Generate mean forecasts."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        return np.full(steps, self._mean_value)


class DriftForecaster(BaseForecaster):
    """
    Drift forecasting model.

    Extends the last value with the average trend (drift) observed in the data.
    This is equivalent to drawing a line from the first to the last observation
    and extrapolating it forward.

    Formula: y_t = y_T + (t - T) * (y_T - y_1) / (T - 1)

    where T is the last observation in the training set.

    Examples
    --------
    >>> import pandas as pd
    >>> from forecasting.baseline import DriftForecaster
    >>> y = pd.Series([1, 2, 3, 4, 5])
    >>> model = DriftForecaster()
    >>> model.fit(y)
    >>> model.predict(3)
    array([6., 7., 8.])
    """

    def __init__(self):
        super().__init__()
        self._drift: Optional[float] = None
        self._last_value: Optional[float] = None

    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "DriftForecaster":
        """Fit the drift model by computing the average trend."""
        y_arr = self._validate_input(y)
        self._extract_metadata(y)
        self._y_train = y_arr

        if len(y_arr) < 2:
            raise ValueError("Drift forecaster requires at least 2 observations")

        self._drift = (y_arr[-1] - y_arr[0]) / (len(y_arr) - 1)
        self._last_value = y_arr[-1]

        self._fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Generate drift forecasts."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        forecasts = np.zeros(steps)
        for i in range(steps):
            forecasts[i] = self._last_value + (i + 1) * self._drift

        return forecasts
