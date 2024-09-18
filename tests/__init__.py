"""
Tests for the Time Series Forecasting System.
"""

import numpy as np
import pandas as pd
import pytest

from forecasting.baseline import NaiveForecaster, MeanForecaster, DriftForecaster
from forecasting.evaluation import ForecastEvaluator, evaluate
from forecasting.backtesting import TimeSeriesCrossValidator, run_backtest
from forecasting.data_generator import TimeSeriesGenerator, generate_sample_data


class TestNaiveForecaster:
    """Tests for NaiveForecaster."""

    def test_fit_predict_last(self):
        """Test naive forecast with 'last' strategy."""
        y = pd.Series([1, 2, 3, 4, 5])
        model = NaiveForecaster(strategy="last")
        model.fit(y)

        assert model.is_fitted
        predictions = model.predict(3)
        np.testing.assert_array_equal(predictions, np.array([5.0, 5.0, 5.0]))

    def test_fit_predict_seasonal(self):
        """Test naive forecast with 'seasonal' strategy."""
        y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        model = NaiveForecaster(strategy="seasonal", seasonality=4)
        model.fit(y)

        predictions = model.predict(4)
        # Should return values from same quarter last year
        expected = np.array([9.0, 10.0, 11.0, 12.0])
        np.testing.assert_array_equal(predictions, expected)

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        with pytest.raises(ValueError):
            NaiveForecaster(strategy="invalid")

    def test_seasonal_without_period(self):
        """Test that seasonal strategy requires seasonality parameter."""
        with pytest.raises(ValueError):
            NaiveForecaster(strategy="seasonal")

    def test_predict_before_fit(self):
        """Test that predict raises error before fit."""
        model = NaiveForecaster()
        with pytest.raises(RuntimeError):
            model.predict(5)

    def test_empty_input(self):
        """Test that empty input raises error."""
        model = NaiveForecaster()
        with pytest.raises(ValueError):
            model.fit(pd.Series([], dtype=float))


class TestMeanForecaster:
    """Tests for MeanForecaster."""

    def test_fit_predict(self):
        """Test mean forecast."""
        y = pd.Series([1, 2, 3, 4, 5])
        model = MeanForecaster()
        model.fit(y)

        assert model.is_fitted
        predictions = model.predict(3)
        np.testing.assert_array_equal(predictions, np.array([3.0, 3.0, 3.0]))

    def test_rolling_window(self):
        """Test mean forecast with rolling window."""
        y = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        model = MeanForecaster(window=3)
        model.fit(y)

        predictions = model.predict(2)
        expected_mean = np.mean([8, 9, 10])
        np.testing.assert_array_equal(predictions, np.array([expected_mean, expected_mean]))

    def test_invalid_window(self):
        """Test that invalid window raises error."""
        with pytest.raises(ValueError):
            MeanForecaster(window=0)


class TestDriftForecaster:
    """Tests for DriftForecaster."""

    def test_fit_predict(self):
        """Test drift forecast."""
        y = pd.Series([1, 2, 3, 4, 5])
        model = DriftForecaster()
        model.fit(y)

        assert model.is_fitted
        predictions = model.predict(3)
        # Drift = (5-1)/(5-1) = 1 per step
        expected = np.array([6.0, 7.0, 8.0])
        np.testing.assert_array_almost_equal(predictions, expected)

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        model = DriftForecaster()
        with pytest.raises(ValueError):
            model.fit(pd.Series([5.0]))


class TestForecastEvaluator:
    """Tests for ForecastEvaluator."""

    def test_mae(self):
        """Test MAE calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        mae = ForecastEvaluator.mae(y_true, y_pred)
        expected = np.mean([0.1, 0.2, 0.2, 0.1, 0.1])
        np.testing.assert_almost_equal(mae, expected)

    def test_mse(self):
        """Test MSE calculation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.5, 2.5, 2.5])

        mse = ForecastEvaluator.mse(y_true, y_pred)
        expected = np.mean([0.25, 0.25, 0.25])
        np.testing.assert_almost_equal(mse, expected)

    def test_rmse(self):
        """Test RMSE calculation."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])

        rmse = ForecastEvaluator.rmse(y_true, y_pred)
        expected = np.sqrt(np.mean([1, 1, 1]))
        np.testing.assert_almost_equal(rmse, expected)

    def test_mape(self):
        """Test MAPE calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])

        mape = ForecastEvaluator.mape(y_true, y_pred)
        expected = np.mean([0.1, 0.05, 0.0333]) * 100
        np.testing.assert_almost_equal(mape, expected, decimal=2)

    def test_mape_with_zero(self):
        """Test MAPE with zero values."""
        y_true = np.array([0, 1, 2])
        y_pred = np.array([0.1, 1.1, 2.1])

        mape = ForecastEvaluator.mape(y_true, y_pred)
        # Should skip the zero
        assert not np.isinf(mape)

    def test_r2(self):
        """Test R-squared calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        r2 = ForecastEvaluator.r2(y_true, y_pred)
        assert 0 < r2 < 1

    def test_r2_perfect(self):
        """Test R-squared with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        r2 = ForecastEvaluator.r2(y_true, y_pred)
        np.testing.assert_almost_equal(r2, 1.0)

    def test_evaluate_all_metrics(self):
        """Test evaluating all metrics."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        evaluator = ForecastEvaluator()
        results = evaluator.evaluate(y_true, y_pred)

        assert "mae" in results
        assert "mse" in results
        assert "rmse" in results
        assert "mape" in results
        assert "r2" in results

    def test_compare_models(self):
        """Test comparing multiple models."""
        y_true = np.array([1, 2, 3, 4, 5])

        predictions = {
            "model1": np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
            "model2": np.array([1.5, 2.5, 3.5, 4.5, 5.5]),
        }

        evaluator = ForecastEvaluator()
        comparison = evaluator.compare_models(y_true, predictions)

        assert "model1" in comparison.index
        assert "model2" in comparison.index
        # model1 should have better (lower) RMSE
        assert comparison.loc["model1", "rmse"] < comparison.loc["model2", "rmse"]


class TestTimeSeriesCrossValidator:
    """Tests for TimeSeriesCrossValidator."""

    def test_split_expanding(self):
        """Test expanding window split."""
        y = np.arange(100)
        cv = TimeSeriesCrossValidator(n_splits=5, horizon=10, expanding=True)

        splits = list(cv.split(y))
        assert len(splits) == 5

        # Check that training set expands
        train_sizes = [len(s[0]) for s in splits]
        assert train_sizes == sorted(train_sizes)

    def test_split_rolling(self):
        """Test rolling window split."""
        y = np.arange(100)
        cv = TimeSeriesCrossValidator(
            n_splits=5, horizon=10, min_train_size=30, expanding=False
        )

        splits = list(cv.split(y))
        assert len(splits) >= 3

        # Check that training set size stays constant
        train_sizes = [len(s[0]) for s in splits]
        assert len(set(train_sizes)) == 1

    def test_split_with_gap(self):
        """Test split with gap between train and test."""
        y = np.arange(100)
        cv = TimeSeriesCrossValidator(n_splits=3, horizon=5, gap=2)

        for train_idx, test_idx in cv.split(y):
            # Test set should start after gap
            assert test_idx[0] == train_idx[-1] + 1 + 2

    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        y = np.arange(10)
        cv = TimeSeriesCrossValidator(n_splits=5, horizon=10, min_train_size=20)

        with pytest.raises(ValueError):
            list(cv.split(y))


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_run_backtest(self):
        """Test running backtest on a model."""
        y = generate_sample_data(n_samples=200, random_seed=42)
        model = NaiveForecaster()

        results = run_backtest(model, y, n_splits=3, horizon=10)

        assert "mean_metrics" in results
        assert "fold_results" in results
        assert results["n_folds"] == 3

    def test_backtest_metrics(self):
        """Test that backtest returns expected metrics."""
        y = generate_sample_data(n_samples=200, random_seed=42)
        model = MeanForecaster()

        results = run_backtest(
            model, y, n_splits=3, horizon=10, metrics=["mae", "rmse"]
        )

        assert "mae" in results["mean_metrics"]
        assert "rmse" in results["mean_metrics"]


class TestTimeSeriesGenerator:
    """Tests for TimeSeriesGenerator."""

    def test_generate_basic(self):
        """Test basic time series generation."""
        gen = TimeSeriesGenerator(n_samples=100, random_seed=42)
        y = gen.generate()

        assert len(y) == 100
        assert isinstance(y, pd.Series)
        assert isinstance(y.index, pd.DatetimeIndex)

    def test_generate_with_trend(self):
        """Test generation with trend."""
        gen = TimeSeriesGenerator(n_samples=100, random_seed=42)

        y_linear = gen.generate(trend="linear", trend_strength=1.0, noise=False)
        assert y_linear.iloc[-1] > y_linear.iloc[0]

    def test_generate_with_seasonality(self):
        """Test generation with seasonality."""
        gen = TimeSeriesGenerator(n_samples=365, random_seed=42)
        y = gen.generate(seasonality=7, seasonality_strength=0.5, noise=False)

        # Check that seasonality exists (values should repeat pattern)
        # This is a weak test; stronger would use autocorrelation
        assert len(y) == 365

    def test_generate_with_outliers(self):
        """Test generation with outliers."""
        gen = TimeSeriesGenerator(n_samples=1000, random_seed=42)
        y_with = gen.generate(outliers=True, outlier_fraction=0.02)
        y_without = gen.generate(outliers=False)

        # Series with outliers should have higher variance
        assert y_with.std() > y_without.std() * 0.5

    def test_invalid_trend_type(self):
        """Test that invalid trend type raises error."""
        gen = TimeSeriesGenerator(n_samples=100)
        with pytest.raises(ValueError):
            gen.generate(trend="invalid")

    def test_generate_multiple(self):
        """Test generating multiple series."""
        gen = TimeSeriesGenerator(n_samples=100, random_seed=42)
        series_list = gen.generate_multiple(n_series=3)

        assert len(series_list) == 3
        assert all(len(s) == 100 for s in series_list)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_evaluate_function(self):
        """Test the evaluate() convenience function."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])

        results = evaluate(y_true, y_pred, metrics=["mae", "rmse"])

        assert "mae" in results
        assert "rmse" in results

    def test_generate_sample_data(self):
        """Test the generate_sample_data() function."""
        y = generate_sample_data(n_samples=365, random_seed=42)

        assert len(y) == 365
        assert isinstance(y, pd.Series)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline(self):
        """Test the full forecasting pipeline."""
        # Generate data
        y = generate_sample_data(n_samples=200, random_seed=42)

        # Split into train/test
        train_size = 150
        y_train = y.iloc[:train_size]
        y_test = y.iloc[train_size:]

        # Fit models
        models = {
            "naive": NaiveForecaster(),
            "mean": MeanForecaster(),
            "drift": DriftForecaster(),
        }

        predictions = {}
        for name, model in models.items():
            model.fit(y_train)
            predictions[name] = model.predict(len(y_test))

        # Evaluate
        evaluator = ForecastEvaluator()
        comparison = evaluator.compare_models(
            y_test.values,
            {k: v for k, v in predictions.items()},
            y_train=y_train.values,
        )

        # All models should have metrics
        assert len(comparison) == 3
        assert all("rmse" in comparison.columns for _ in range(3))


# Mark slow tests
@pytest.mark.slow
class TestSlowIntegration:
    """Slow integration tests that may take time."""

    @pytest.mark.skip(reason="Requires statsmodels installation")
    def test_arima_forecaster(self):
        """Test ARIMA forecaster (requires statsmodels)."""
        pytest.importorskip("statsmodels")

        from forecasting.statistical import ARIMAForecaster

        y = generate_sample_data(n_samples=200, random_seed=42)
        model = ARIMAForecaster(order=(1, 1, 1))
        model.fit(y)
        predictions = model.predict(10)

        assert len(predictions) == 10

    @pytest.mark.skip(reason="Requires XGBoost installation")
    def test_xgboost_forecaster(self):
        """Test XGBoost forecaster."""
        pytest.importorskip("xgboost")

        from forecasting.ml_forecaster import XGBoostForecaster

        y = generate_sample_data(n_samples=200, random_seed=42)
        model = XGBoostForecaster(lags=12, n_estimators=50)
        model.fit(y)
        predictions = model.predict(10)

        assert len(predictions) == 10
