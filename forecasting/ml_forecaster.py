"""
Machine Learning forecasting models for time series analysis.

These models use ML techniques like XGBoost and LSTM for forecasting,
converting time series into supervised learning problems.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from forecasting.baseline import BaseForecaster


def create_lagged_features(
    y: np.ndarray, lags: int, include_trend: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create lagged features for time series supervised learning.

    Parameters
    ----------
    y : np.ndarray
        The time series data.
    lags : int
        Number of lagged observations to use as features.
    include_trend : bool
        Whether to include a time index feature.

    Returns
    -------
    X : np.ndarray
        Feature matrix with lagged values.
    y_next : np.ndarray
        Target values (next step ahead).
    """
    X_list = []
    y_list = []

    for i in range(lags, len(y)):
        features = y[i - lags : i]
        if include_trend:
            features = np.append(features, i)
        X_list.append(features)
        y_list.append(y[i])

    return np.array(X_list), np.array(y_list)


class XGBoostForecaster(BaseForecaster):
    """
    XGBoost forecasting model.

    Uses XGBoost's gradient boosting for time series forecasting
    by creating lagged features from the time series.

    Parameters
    ----------
    lags : int, default=12
        Number of lagged observations to use as features.
    n_estimators : int, default=100
        Number of boosting rounds.
    max_depth : int, default=6
        Maximum tree depth.
    learning_rate : float, default=0.1
        Learning rate (eta).
    subsample : float, default=1.0
        Subsample ratio of training instances.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    include_trend : bool, default=False
        Whether to include a time index feature.
    **xgb_kwargs : dict
        Additional arguments passed to XGBRegressor.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from forecasting.ml_forecaster import XGBoostForecaster
    >>> y = pd.Series(np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1)
    >>> model = XGBoostForecaster(lags=5, n_estimators=50)
    >>> model.fit(y)
    >>> model.predict(10)
    array([...])
    """

    def __init__(
        self,
        lags: int = 12,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        include_trend: bool = False,
        **xgb_kwargs,
    ):
        super().__init__()
        if lags < 1:
            raise ValueError("lags must be a positive integer")
        self.lags = lags
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.include_trend = include_trend
        self.xgb_kwargs = xgb_kwargs
        self._model = None
        self._last_values: Optional[np.ndarray] = None
        self._current_idx: int = 0

    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "XGBoostForecaster":
        """
        Fit the XGBoost model to the data.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            The time series data to fit.

        **kwargs : dict
            Additional arguments passed to XGBRegressor.fit()

        Returns
        -------
        self : XGBoostForecaster
            The fitted model instance.
        """
        import xgboost as xgb

        y_arr = self._validate_input(y)
        self._extract_metadata(y)
        self._y_train = y_arr

        if len(y_arr) <= self.lags:
            raise ValueError(
                f"Time series too short for {self.lags} lags. "
                f"Need at least {self.lags + 1} observations."
            )

        # Create features
        X, y_target = create_lagged_features(y_arr, self.lags, self.include_trend)

        # Store last values for prediction
        self._last_values = y_arr[-self.lags :].copy()
        self._current_idx = len(y_arr)

        # Initialize and fit model
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            **self.xgb_kwargs,
        )

        self._model.fit(X, y_target, **kwargs)
        self._fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for the specified number of steps.

        Uses iterative forecasting where each prediction becomes
        a feature for the next prediction.

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

        forecasts = np.zeros(steps)
        current_values = self._last_values.copy()

        for i in range(steps):
            # Create features from current lagged values
            features = current_values[-self.lags :]
            if self.include_trend:
                features = np.append(features, self._current_idx + i)

            # Predict next value
            pred = self._model.predict(features.reshape(1, -1))[0]
            forecasts[i] = pred

            # Update current values for next iteration
            current_values = np.append(current_values[1:], pred)

        return forecasts

    def get_feature_importance(self) -> dict:
        """
        Get feature importance scores.

        Returns
        -------
        importance : dict
            Dictionary mapping feature names to importance scores.
        """
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        feature_names = [f"lag_{i}" for i in range(self.lags, 0, -1)]
        if self.include_trend:
            feature_names.append("trend")

        return dict(zip(feature_names, self._model.feature_importances_))


class LSTMForecaster(BaseForecaster):
    """
    LSTM (Long Short-Term Memory) forecasting model.

    Uses a neural network with LSTM layers for time series forecasting.
    Requires TensorFlow/Keras to be installed.

    Parameters
    ----------
    lags : int, default=12
        Number of lagged observations to use as features.
    hidden_units : int or list, default=50
        Number of LSTM units. If list, creates stacked LSTM layers.
    epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=32
        Training batch size.
    dropout : float, default=0.0
        Dropout rate for regularization.
    learning_rate : float, default=0.001
        Learning rate for the Adam optimizer.
    verbose : int, default=0
        Verbosity mode (0, 1, or 2).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from forecasting.ml_forecaster import LSTMForecaster
    >>> y = pd.Series(np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1)
    >>> model = LSTMForecaster(lags=10, epochs=50)
    >>> model.fit(y)  # doctest: +SKIP
    >>> model.predict(10)  # doctest: +SKIP
    """

    def __init__(
        self,
        lags: int = 12,
        hidden_units: Union[int, List[int]] = 50,
        epochs: int = 100,
        batch_size: int = 32,
        dropout: float = 0.0,
        learning_rate: float = 0.001,
        verbose: int = 0,
    ):
        super().__init__()
        if lags < 1:
            raise ValueError("lags must be a positive integer")
        self.lags = lags
        if isinstance(hidden_units, int):
            self.hidden_units = [hidden_units]
        else:
            self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.verbose = verbose
        self._model = None
        self._last_values: Optional[np.ndarray] = None
        self._scaler = None

    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> "LSTMForecaster":
        """
        Fit the LSTM model to the data.

        Parameters
        ----------
        y : pd.Series or np.ndarray
            The time series data to fit.

        **kwargs : dict
            Additional arguments passed to model.fit()

        Returns
        -------
        self : LSTMForecaster
            The fitted model instance.
        """
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        import tensorflow as tf
        from tensorflow import keras
        from sklearn.preprocessing import MinMaxScaler

        tf.get_logger().setLevel("ERROR")

        y_arr = self._validate_input(y)
        self._extract_metadata(y)
        self._y_train = y_arr

        if len(y_arr) <= self.lags:
            raise ValueError(
                f"Time series too short for {self.lags} lags. "
                f"Need at least {self.lags + 1} observations."
            )

        # Scale the data
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        y_scaled = self._scaler.fit_transform(y_arr.reshape(-1, 1)).flatten()

        # Create features
        X, y_target = create_lagged_features(y_scaled, self.lags)

        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Store last values for prediction
        self._last_values = y_scaled[-self.lags :].copy()

        # Build model
        model = keras.Sequential()

        for i, units in enumerate(self.hidden_units):
            return_sequences = i < len(self.hidden_units) - 1
            if i == 0:
                model.add(
                    keras.layers.LSTM(
                        units,
                        return_sequences=return_sequences,
                        input_shape=(self.lags, 1),
                    )
                )
            else:
                model.add(
                    keras.layers.LSTM(units, return_sequences=return_sequences)
                )

            if self.dropout > 0:
                model.add(keras.layers.Dropout(self.dropout))

        model.add(keras.layers.Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
        )

        # Train model
        model.fit(
            X,
            y_target,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
            **kwargs,
        )

        self._model = model
        self._fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        """
        Generate forecasts for the specified number of steps.

        Uses iterative forecasting where each prediction becomes
        a feature for the next prediction.

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

        forecasts = np.zeros(steps)
        current_values = self._last_values.copy()

        for i in range(steps):
            # Create features from current lagged values
            features = current_values[-self.lags :].reshape(1, self.lags, 1)

            # Predict next value
            pred = self._model.predict(features, verbose=0)[0, 0]
            forecasts[i] = pred

            # Update current values for next iteration
            current_values = np.append(current_values[1:], pred)

        # Inverse transform to original scale
        forecasts = self._scaler.inverse_transform(forecasts.reshape(-1, 1)).flatten()

        return forecasts

    def get_model_summary(self) -> str:
        """Get the model architecture summary."""
        if not self._fitted:
            raise RuntimeError("Model must be fitted first")

        summary_list = []
        self._model.summary(print_fn=lambda x: summary_list.append(x))
        return "\n".join(summary_list)
