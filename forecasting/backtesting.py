"""
Backtesting framework for time series forecasting.

Provides tools for cross-validation and backtesting of forecasting models
with proper temporal ordering.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesCrossValidator:
    """
    Time series cross-validation for forecasting models.

    Implements various cross-validation strategies that respect
    temporal ordering of time series data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits/folds for cross-validation.
    horizon : int, default=1
        Forecast horizon (number of steps to predict).
    gap : int, default=0
        Gap between training and test sets.
    min_train_size : int, optional
        Minimum number of observations in training set.
    expanding : bool, default=True
        If True, training set expands with each split.
        If False, training set slides (rolling window).

    Examples
    --------
    >>> import numpy as np
    >>> from forecasting.backtesting import TimeSeriesCrossValidator
    >>> y = np.arange(100)
    >>> cv = TimeSeriesCrossValidator(n_splits=5, horizon=10)
    >>> for train_idx, test_idx in cv.split(y):
    ...     print(f"Train: {train_idx[0]}-{train_idx[-1]}, Test: {test_idx[0]}-{test_idx[-1]}")
    Train: 0-44, Test: 45-54
    Train: 0-54, Test: 55-64
    ...
    """

    def __init__(
        self,
        n_splits: int = 5,
        horizon: int = 1,
        gap: int = 0,
        min_train_size: Optional[int] = None,
        expanding: bool = True,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        if gap < 0:
            raise ValueError("gap cannot be negative")

        self.n_splits = n_splits
        self.horizon = horizon
        self.gap = gap
        self.min_train_size = min_train_size or horizon
        self.expanding = expanding

    def split(
        self, y: Union[np.ndarray, pd.Series]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate train/test indices for cross-validation.

        Parameters
        ----------
        y : np.ndarray or pd.Series
            Time series data.

        Yields
        ------
        train_idx : np.ndarray
            Training set indices.
        test_idx : np.ndarray
            Test set indices.
        """
        if isinstance(y, pd.Series):
            n_samples = len(y)
        else:
            n_samples = len(y)

        if n_samples < self.min_train_size + self.horizon + self.gap:
            raise ValueError(
                f"Not enough samples ({n_samples}) for min_train_size={self.min_train_size}, "
                f"horizon={self.horizon}, gap={self.gap}"
            )

        # Calculate split points
        if self.expanding:
            # Expanding window: train set grows
            available = n_samples - self.min_train_size - self.gap - self.horizon
            step = max(1, available // (self.n_splits - 1))

            for i in range(self.n_splits):
                train_end = self.min_train_size + i * step
                test_start = train_end + self.gap
                test_end = test_start + self.horizon

                if test_end > n_samples:
                    break

                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)

                yield train_idx, test_idx
        else:
            # Rolling window: fixed train set size
            train_size = self.min_train_size
            available = n_samples - train_size - self.gap - self.horizon
            step = max(1, available // (self.n_splits - 1))

            for i in range(self.n_splits):
                train_start = i * step
                train_end = train_start + train_size
                test_start = train_end + self.gap
                test_end = test_start + self.horizon

                if test_end > n_samples:
                    break

                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)

                yield train_idx, test_idx

    def get_n_splits(self, y: Optional[Union[np.ndarray, pd.Series]] = None) -> int:
        """Return the number of splitting iterations."""
        if y is None:
            return self.n_splits

        actual_splits = 0
        for _ in self.split(y):
            actual_splits += 1
        return actual_splits


class BacktestEngine:
    """
    Engine for running backtests on forecasting models.

    Provides utilities for backtesting models with various configurations
    and aggregating results.

    Parameters
    ----------
    cv : TimeSeriesCrossValidator
        Cross-validation strategy.
    metrics : list, optional
        Metrics to compute. If None, uses all available metrics.

    Examples
    --------
    >>> import numpy as np
    >>> from forecasting.backtesting import BacktestEngine, TimeSeriesCrossValidator
    >>> from forecasting.baseline import NaiveForecaster
    >>> cv = TimeSeriesCrossValidator(n_splits=3, horizon=5)
    >>> engine = BacktestEngine(cv)
    >>> y = np.random.randn(50).cumsum() + 100
    >>> model = NaiveForecaster()
    >>> results = engine.run_backtest(model, y)
    """

    def __init__(
        self,
        cv: TimeSeriesCrossValidator,
        metrics: Optional[List[str]] = None,
    ):
        self.cv = cv
        self.metrics = metrics
        self._results: List[Dict[str, Any]] = []

    def run_backtest(
        self,
        model: Any,
        y: Union[np.ndarray, pd.Series],
        fit_kwargs: Optional[Dict] = None,
        predict_kwargs: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest on a single model.

        Parameters
        ----------
        model : object
            Forecasting model with fit() and predict() methods.
        y : np.ndarray or pd.Series
            Time series data.
        fit_kwargs : dict, optional
            Additional arguments for model.fit().
        predict_kwargs : dict, optional
            Additional arguments for model.predict().

        Returns
        -------
        results : dict
            Dictionary with backtest results including:
            - 'fold_results': List of per-fold metrics
            - 'mean_metrics': Average metrics across folds
            - 'std_metrics': Standard deviation of metrics
            - 'predictions': All predictions combined
        """
        from forecasting.evaluation import ForecastEvaluator

        fit_kwargs = fit_kwargs or {}
        predict_kwargs = predict_kwargs or {}

        evaluator = ForecastEvaluator(self.metrics)
        fold_results = []
        all_predictions = []
        all_actuals = []

        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(y)):
            # Split data
            if isinstance(y, pd.Series):
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
            else:
                y_train = y[train_idx]
                y_test = y[test_idx]

            # Fit and predict
            try:
                model.fit(y_train, **fit_kwargs)
                predictions = model.predict(len(test_idx), **predict_kwargs)

                # Calculate metrics
                metrics = evaluator.evaluate(
                    y_test if isinstance(y_test, np.ndarray) else y_test.values,
                    predictions,
                    y_train=y_train if isinstance(y_train, np.ndarray) else y_train.values,
                )

                fold_results.append(
                    {
                        "fold": fold_idx,
                        "train_size": len(train_idx),
                        "test_size": len(test_idx),
                        "metrics": metrics,
                    }
                )

                all_predictions.extend(predictions)
                all_actuals.extend(
                    y_test if isinstance(y_test, np.ndarray) else y_test.values
                )

            except Exception as e:
                fold_results.append(
                    {
                        "fold": fold_idx,
                        "error": str(e),
                    }
                )

        # Aggregate results
        successful_folds = [r for r in fold_results if "error" not in r]

        if successful_folds:
            # Calculate mean and std of metrics
            metric_names = successful_folds[0]["metrics"].keys()
            mean_metrics = {}
            std_metrics = {}

            for metric in metric_names:
                values = [f["metrics"][metric] for f in successful_folds]
                values = [v for v in values if not np.isnan(v)]
                if values:
                    mean_metrics[metric] = np.mean(values)
                    std_metrics[metric] = np.std(values)
                else:
                    mean_metrics[metric] = np.nan
                    std_metrics[metric] = np.nan

            # Calculate overall metrics
            overall_metrics = evaluator.evaluate(
                np.array(all_actuals), np.array(all_predictions)
            )
        else:
            mean_metrics = {}
            std_metrics = {}
            overall_metrics = {}

        results = {
            "fold_results": fold_results,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "overall_metrics": overall_metrics,
            "predictions": np.array(all_predictions),
            "actuals": np.array(all_actuals),
            "n_folds": len(successful_folds),
        }

        self._results.append({"model": model.__class__.__name__, **results})

        return results

    def compare_models(
        self,
        models: Dict[str, Any],
        y: Union[np.ndarray, pd.Series],
        fit_kwargs: Optional[Dict] = None,
        predict_kwargs: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple models using backtesting.

        Parameters
        ----------
        models : dict
            Dictionary mapping model names to model instances.
        y : np.ndarray or pd.Series
            Time series data.
        fit_kwargs : dict, optional
            Additional arguments for model.fit().
        predict_kwargs : dict, optional
            Additional arguments for model.predict().

        Returns
        -------
        comparison : pd.DataFrame
            DataFrame with mean and std metrics for each model.
        """
        results = {}

        for name, model in models.items():
            try:
                backtest_result = self.run_backtest(
                    model, y, fit_kwargs, predict_kwargs
                )
                results[name] = {
                    **{f"mean_{k}": v for k, v in backtest_result["mean_metrics"].items()},
                    **{f"std_{k}": v for k, v in backtest_result["std_metrics"].items()},
                    "n_folds": backtest_result["n_folds"],
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        df = pd.DataFrame(results).T
        return df

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all stored backtest results."""
        return self._results


def run_backtest(
    model: Any,
    y: Union[np.ndarray, pd.Series],
    n_splits: int = 5,
    horizon: int = 1,
    gap: int = 0,
    expanding: bool = True,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run a quick backtest.

    Parameters
    ----------
    model : object
        Forecasting model with fit() and predict() methods.
    y : np.ndarray or pd.Series
        Time series data.
    n_splits : int
        Number of cross-validation splits.
    horizon : int
        Forecast horizon.
    gap : int
        Gap between train and test sets.
    expanding : bool
        Use expanding window if True, rolling window if False.
    metrics : list, optional
        Metrics to compute.

    Returns
    -------
    results : dict
        Backtest results.

    Examples
    --------
    >>> import numpy as np
    >>> from forecasting.backtesting import run_backtest
    >>> from forecasting.baseline import NaiveForecaster
    >>> y = np.random.randn(50).cumsum() + 100
    >>> model = NaiveForecaster()
    >>> results = run_backtest(model, y, n_splits=3, horizon=5)
    """
    cv = TimeSeriesCrossValidator(
        n_splits=n_splits,
        horizon=horizon,
        gap=gap,
        expanding=expanding,
    )

    engine = BacktestEngine(cv, metrics)
    return engine.run_backtest(model, y)
