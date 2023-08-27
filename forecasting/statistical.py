"""
Statistical forecasting models for time series analysis.

These models use well-established statistical methods for forecasting,
including ARIMA and Prophet.
"""

from typing import Dict, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from forecasting.baseline import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA forecasting model.

    AutoRegressive Integrated Moving Average model for time series forecasting.
    This implementation uses statsmodels' SARIMAX which supports both
    non-seasonal and seasonal ARIMA models.

    Parameters
    ----------
    order : tuple, default=(1, 0, 0)
        The (p, d, q) order of the ARIMA model:
        - p: autoregressive order
        - d: differencing order
        - q: moving average order

    seasonal_order : tuple, optional
        The (P, D, Q, s) seasonal order:
        - P: seasonal autoregressive order
        - D: seasonal differencing order
        - Q: seasonal moving average order
        - s: seasonal period (e.g., 12 for monthly data)

    trend : str, default='c'
        Trend component: 'c' for constant, 't' for linear trend,
        'ct' for both, None for no trend.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from forecasting.statistical import ARIMAForecaster
    >>> y = pd.Series(np.random.randn(100).cumsum())
    >>> model = ARIMAForecaster(order=(1, 1, 1))
    >>> model.fit(y)
    >>> model.predict(10)  # doctest: +SKIP
    """

    def __init__(
        self,
        order: Tuple[int, int, int] = (1, 0, 0),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        trend: Optional[str] = "c",
    ):
        super().__init__()
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self._model = None
        self._fitted_model = None

    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "ARIMAForecaster":
        """
        Fit the ARIMA model to the data.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            The time series data to fit.

        **kwargs : dict
            Additional arguments passed to statsmodels.tsa.SARIMAX.fit()

        Returns
        -------
        self : ARIMAForecaster
            The fitted model instance.
        """
        import statsmodels.api as sm

        y_arr = self._validate_input(y)
        self._extract_metadata(y)
        self._y_train = y_arr

        # Convert to pandas Series with proper index for statsmodels
        if isinstance(y, pd.Series) and isinstance(y.index, pd.DatetimeIndex):
            y_series = y
            if y.index.freq is None:
                y_series = y.asfreq(pd.infer_freq(y.index))
        else:
            y_series = pd.Series(y_arr)

        # Build model kwargs
        model_kwargs = {"endog": y_series, "order": self.order, "trend": self.trend}

        if self.seasonal_order is not None:
            model_kwargs["seasonal_order"] = self.seasonal_order

        # Suppress warnings during fitting
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = sm.tsa.SARIMAX(**model_kwargs)
            self._fitted_model = self._model.fit(disp=0, **kwargs)

        self._fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for the specified number of steps.

        Parameters
        ----------
        steps : int
            Number of steps to forecast ahead.

        Returns
        -------
        forecasts : np.ndarray
            Array of forecasted values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        forecast = self._fitted_model.get_forecast(steps=steps)
        return forecast.predicted_mean.values

    def predict_with_ci(
        self, steps: int, alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts with confidence intervals.

        Parameters
        ----------
        steps : int
            Number of steps to forecast ahead.
        alpha : float
            Significance level for confidence intervals (default: 0.05 for 95% CI).

        Returns
        -------
        forecasts : tuple
            (predicted_mean, lower_bound, upper_bound)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        forecast = self._fitted_model.get_forecast(steps=steps)
        predicted_mean = forecast.predicted_mean.values
        conf_int = forecast.conf_int(alpha=alpha)

        return predicted_mean, conf_int.iloc[:, 0].values, conf_int.iloc[:, 1].values

    def summary(self) -> str:
        """Return the model summary."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return str(self._fitted_model.summary())

    def get_aic(self) -> float:
        """Return the Akaike Information Criterion."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return self._fitted_model.aic

    def get_bic(self) -> float:
        """Return the Bayesian Information Criterion."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")
        return self._fitted_model.bic


class ProphetForecaster(BaseForecaster):
    """
    Prophet forecasting model.

    Prophet is a forecasting procedure developed by Facebook for
    forecasting time series data with trends and seasonality.

    Parameters
    ----------
    growth : str, default='linear'
        Type of trend: 'linear' or 'logistic'.

    seasonality_mode : str, default='additive'
        Type of seasonality: 'additive' or 'multiplicative'.

    yearly_seasonality : bool or 'auto', default='auto'
        Fit yearly seasonality.

    weekly_seasonality : bool or 'auto', default='auto'
        Fit weekly seasonality.

    daily_seasonality : bool or 'auto', default='auto'
        Fit daily seasonality.

    changepoint_prior_scale : float, default=0.05
        Flexibility of the trend. Higher values allow more changepoints.

    **prophet_kwargs : dict
        Additional arguments passed to Prophet constructor.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from forecasting.statistical import ProphetForecaster
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> y = pd.Series(np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1, index=dates)
    >>> model = ProphetForecaster()
    >>> model.fit(y)  # doctest: +SKIP
    >>> model.predict(10)  # doctest: +SKIP
    """

    def __init__(
        self,
        growth: str = "linear",
        seasonality_mode: str = "additive",
        yearly_seasonality: Union[bool, str] = "auto",
        weekly_seasonality: Union[bool, str] = "auto",
        daily_seasonality: Union[bool, str] = "auto",
        changepoint_prior_scale: float = 0.05,
        **prophet_kwargs,
    ):
        super().__init__()
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.prophet_kwargs = prophet_kwargs
        self._model = None
        self._last_date = None
        self._freq = None

    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "ProphetForecaster":
        """
        Fit the Prophet model to the data.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            The time series data. If pd.Series, must have DatetimeIndex.

        **kwargs : dict
            Additional arguments passed to Prophet.fit()

        Returns
        -------
        self : ProphetForecaster
            The fitted model instance.
        """
        from prophet import Prophet

        y_arr = self._validate_input(y)
        self._y_train = y_arr

        # Prophet requires DatetimeIndex
        if not isinstance(y, pd.Series) or not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError(
                "ProphetForecaster requires a pandas Series with DatetimeIndex"
            )

        self._extract_metadata(y)

        # Create Prophet-compatible dataframe
        df = pd.DataFrame({"ds": y.index, "y": y.values})

        # Initialize Prophet model
        self._model = Prophet(
            growth=self.growth,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            **self.prophet_kwargs,
        )

        # Suppress cmdstanpy output
        import logging

        logger = logging.getLogger("cmdstanpy")
        original_level = logger.level
        logger.setLevel(logging.WARNING)

        try:
            self._model.fit(df, **kwargs)
        finally:
            logger.setLevel(original_level)

        self._fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for the specified number of steps.

        Parameters
        ----------
        steps : int
            Number of steps to forecast ahead.

        Returns
        -------
        forecasts : np.ndarray
            Array of forecasted values.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Create future dataframe
        future = self._model.make_future_dataframe(periods=steps, freq=self._freq)

        # Generate predictions
        forecast = self._model.predict(future)

        # Return only the future predictions
        return forecast["yhat"].iloc[-steps:].values

    def predict_with_ci(
        self, steps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate forecasts with uncertainty intervals.

        Parameters
        ----------
        steps : int
            Number of steps to forecast ahead.

        Returns
        -------
        forecasts : tuple
            (predicted_mean, lower_bound, upper_bound)
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        future = self._model.make_future_dataframe(periods=steps, freq=self._freq)
        forecast = self._model.predict(future)

        predictions = forecast.iloc[-steps:]
        return (
            predictions["yhat"].values,
            predictions["yhat_lower"].values,
            predictions["yhat_upper"].values,
        )

    def get_components(self, steps: int) -> pd.DataFrame:
        """
        Get the decomposed forecast components.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.

        Returns
        -------
        components : pd.DataFrame
            DataFrame with trend, seasonal, and other components.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction")

        future = self._model.make_future_dataframe(periods=steps, freq=self._freq)
        forecast = self._model.predict(future)
        return forecast.iloc[-steps:]
