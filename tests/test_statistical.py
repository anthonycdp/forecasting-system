"""
Additional tests for statistical and ML models.

These tests are separated because they require additional dependencies.
"""

import numpy as np
import pandas as pd
import pytest


class TestARIMAForecaster:
    """Tests for ARIMAForecaster (requires statsmodels)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        y = pd.Series(
            np.random.randn(100).cumsum() + 100,
            index=dates
        )
        return y

    def test_arima_basic(self, sample_data):
        """Test basic ARIMA fit and predict."""
        pytest.importorskip("statsmodels")

        from forecasting.statistical import ARIMAForecaster

        model = ARIMAForecaster(order=(1, 0, 0))
        model.fit(sample_data)

        assert model.is_fitted
        predictions = model.predict(10)
        assert len(predictions) == 10

    def test_arima_with_differencing(self, sample_data):
        """Test ARIMA with differencing."""
        pytest.importorskip("statsmodels")

        from forecasting.statistical import ARIMAForecaster

        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(sample_data)

        predictions = model.predict(5)
        assert len(predictions) == 5

    def test_arima_predict_with_ci(self, sample_data):
        """Test ARIMA predictions with confidence intervals."""
        pytest.importorskip("statsmodels")

        from forecasting.statistical import ARIMAForecaster

        model = ARIMAForecaster(order=(1, 0, 0))
        model.fit(sample_data)

        mean, lower, upper = model.predict_with_ci(10)

        assert len(mean) == 10
        assert len(lower) == 10
        assert len(upper) == 10
        assert np.all(lower <= mean)
        assert np.all(mean <= upper)

    def test_arima_aic_bic(self, sample_data):
        """Test getting AIC and BIC values."""
        pytest.importorskip("statsmodels")

        from forecasting.statistical import ARIMAForecaster

        model = ARIMAForecaster(order=(1, 0, 0))
        model.fit(sample_data)

        aic = model.get_aic()
        bic = model.get_bic()

        assert isinstance(aic, float)
        assert isinstance(bic, float)

    def test_arima_requires_series(self, sample_data):
        """Test that ARIMA requires proper input."""
        pytest.importorskip("statsmodels")

        from forecasting.statistical import ARIMAForecaster

        model = ARIMAForecaster(order=(1, 0, 0))

        # Should work with pandas Series
        model.fit(sample_data)
        assert model.is_fitted


class TestXGBoostForecaster:
    """Tests for XGBoostForecaster."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        y = pd.Series(np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1)
        return y

    def test_xgboost_basic(self, sample_data):
        """Test basic XGBoost fit and predict."""
        pytest.importorskip("xgboost")

        from forecasting.ml_forecaster import XGBoostForecaster

        model = XGBoostForecaster(lags=5, n_estimators=50)
        model.fit(sample_data)

        assert model.is_fitted
        predictions = model.predict(10)
        assert len(predictions) == 10

    def test_xgboost_feature_importance(self, sample_data):
        """Test getting feature importance."""
        pytest.importorskip("xgboost")

        from forecasting.ml_forecaster import XGBoostForecaster

        model = XGBoostForecaster(lags=5, n_estimators=50)
        model.fit(sample_data)

        importance = model.get_feature_importance()

        assert len(importance) == 5
        assert all(v >= 0 for v in importance.values())

    def test_xgboost_with_trend(self, sample_data):
        """Test XGBoost with trend feature."""
        pytest.importorskip("xgboost")

        from forecasting.ml_forecaster import XGBoostForecaster

        model = XGBoostForecaster(lags=5, n_estimators=50, include_trend=True)
        model.fit(sample_data)

        importance = model.get_feature_importance()
        assert "trend" in importance

    def test_xgboost_insufficient_data(self):
        """Test that insufficient data raises error."""
        pytest.importorskip("xgboost")

        from forecasting.ml_forecaster import XGBoostForecaster

        model = XGBoostForecaster(lags=10)
        y = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError):
            model.fit(y)

    def test_xgboost_invalid_lags(self):
        """Test that invalid lags raises error."""
        pytest.importorskip("xgboost")

        from forecasting.ml_forecaster import XGBoostForecaster

        with pytest.raises(ValueError):
            XGBoostForecaster(lags=0)


class TestLSTMForecaster:
    """Tests for LSTMForecaster (requires TensorFlow)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data."""
        np.random.seed(42)
        y = pd.Series(np.sin(np.arange(100) * 0.1) + np.random.randn(100) * 0.1)
        return y

    @pytest.mark.slow
    def test_lstm_basic(self, sample_data):
        """Test basic LSTM fit and predict."""
        pytest.importorskip("tensorflow")

        from forecasting.ml_forecaster import LSTMForecaster

        model = LSTMForecaster(lags=10, epochs=5, verbose=0)
        model.fit(sample_data)

        assert model.is_fitted
        predictions = model.predict(5)
        assert len(predictions) == 5

    @pytest.mark.slow
    def test_lstm_stacked(self, sample_data):
        """Test stacked LSTM layers."""
        pytest.importorskip("tensorflow")

        from forecasting.ml_forecaster import LSTMForecaster

        model = LSTMForecaster(
            lags=10,
            hidden_units=[32, 16],
            epochs=5,
            verbose=0
        )
        model.fit(sample_data)

        summary = model.get_model_summary()
        assert "lstm" in summary.lower()

    @pytest.mark.slow
    def test_lstm_with_dropout(self, sample_data):
        """Test LSTM with dropout."""
        pytest.importorskip("tensorflow")

        from forecasting.ml_forecaster import LSTMForecaster

        model = LSTMForecaster(lags=10, epochs=5, dropout=0.2, verbose=0)
        model.fit(sample_data)

        predictions = model.predict(5)
        assert len(predictions) == 5


class TestProphetForecaster:
    """Tests for ProphetForecaster (requires Prophet)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample time series data with DatetimeIndex."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        y = pd.Series(
            100 + np.sin(np.arange(100) * 0.1) * 10 + np.random.randn(100) * 2,
            index=dates
        )
        return y

    @pytest.mark.slow
    def test_prophet_basic(self, sample_data):
        """Test basic Prophet fit and predict."""
        pytest.importorskip("prophet")

        from forecasting.statistical import ProphetForecaster

        model = ProphetForecaster()
        model.fit(sample_data)

        assert model.is_fitted
        predictions = model.predict(10)
        assert len(predictions) == 10

    @pytest.mark.slow
    def test_prophet_with_ci(self, sample_data):
        """Test Prophet predictions with confidence intervals."""
        pytest.importorskip("prophet")

        from forecasting.statistical import ProphetForecaster

        model = ProphetForecaster()
        model.fit(sample_data)

        mean, lower, upper = model.predict_with_ci(10)

        assert len(mean) == 10
        assert len(lower) == 10
        assert len(upper) == 10

    @pytest.mark.slow
    def test_prophet_requires_datetime_index(self):
        """Test that Prophet requires DatetimeIndex."""
        pytest.importorskip("prophet")

        from forecasting.statistical import ProphetForecaster

        model = ProphetForecaster()
        y = pd.Series([1, 2, 3, 4, 5])  # No DatetimeIndex

        with pytest.raises(ValueError):
            model.fit(y)

    @pytest.mark.slow
    def test_prophet_components(self, sample_data):
        """Test getting Prophet components."""
        pytest.importorskip("prophet")

        from forecasting.statistical import ProphetForecaster

        model = ProphetForecaster()
        model.fit(sample_data)

        components = model.get_components(10)

        assert len(components) == 10
        assert "trend" in components.columns


class TestModelComparison:
    """Tests for comparing different model types."""

    @pytest.mark.slow
    def test_baseline_vs_statistical(self):
        """Test that ARIMA outperforms baseline on appropriate data."""
        pytest.importorskip("statsmodels")

        from forecasting.baseline import NaiveForecaster
        from forecasting.statistical import ARIMAForecaster
        from forecasting.evaluation import ForecastEvaluator

        # Generate AR(1) data
        np.random.seed(42)
        n = 100
        phi = 0.8
        y = np.zeros(n)
        for i in range(1, n):
            y[i] = phi * y[i-1] + np.random.randn()

        y = pd.Series(y)

        # Split
        y_train = y.iloc[:80]
        y_test = y.iloc[80:]

        # Fit models
        naive = NaiveForecaster()
        naive.fit(y_train)
        naive_pred = naive.predict(20)

        arima = ARIMAForecaster(order=(1, 0, 0))
        arima.fit(y_train)
        arima_pred = arima.predict(20)

        # Evaluate
        evaluator = ForecastEvaluator()
        naive_rmse = evaluator.rmse(y_test.values, naive_pred)
        arima_rmse = evaluator.rmse(y_test.values, arima_pred)

        # ARIMA should at least be competitive
        assert arima_rmse < naive_rmse * 2
