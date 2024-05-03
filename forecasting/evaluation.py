"""
Evaluation metrics and tools for time series forecasting.

Provides comprehensive evaluation metrics for comparing forecasting models.
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


class ForecastEvaluator:
    """
    Comprehensive evaluator for time series forecasts.

    Provides multiple metrics for evaluating forecast accuracy and quality.

    Parameters
    ----------
    metrics : list, optional
        List of metrics to compute. If None, computes all available metrics.
        Available metrics: 'mae', 'mse', 'rmse', 'mape', 'smape', 'mase',
                          'r2', 'theil_u', 'bias'

    Examples
    --------
    >>> import numpy as np
    >>> from forecasting.evaluation import ForecastEvaluator
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    >>> evaluator = ForecastEvaluator()
    >>> evaluator.evaluate(y_true, y_pred)
    {'mae': 0.14, 'mse': 0.024, ...}
    """

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or self._get_all_metric_names()
        self._validate_metrics()

    @classmethod
    def _get_all_metric_names(cls) -> List[str]:
        """Return names of all available metrics."""
        return ["mae", "mse", "rmse", "mape", "smape", "mase", "r2", "theil_u", "bias"]

    def _validate_metrics(self) -> None:
        """Validate that all requested metrics are available."""
        invalid = set(self.metrics) - set(self._get_all_metric_names())
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}")

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or self._get_all_metric_names()
        self._validate_metrics()

    @classmethod
    def _get_all_metric_names(cls) -> List[str]:
        """Return names of all available metrics."""
        return ["mae", "mse", "rmse", "mape", "smape", "mase", "r2", "theil_u", "bias"]

    def _validate_metrics(self) -> None:
        """Validate that all requested metrics are available."""
        invalid = set(self.metrics) - set(self._get_all_metric_names())
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}")

    @staticmethod
    def _calc_mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        m: int = 1,
    ) -> float:
        """Calculate MASE."""
        if y_train is None:
            y_train = y_true

        if len(y_train) <= m:
            return np.inf

        mae = np.mean(np.abs(y_true - y_pred))
        mae_naive = np.mean(np.abs(y_train[m:] - y_train[:-m]))

        if mae_naive == 0:
            return np.inf

        return float(mae / mae_naive)

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not mask.any():
            return np.inf
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not mask.any():
            return np.inf
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)

    @staticmethod
    def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: Optional[np.ndarray] = None, m: int = 1) -> float:
        """Mean Absolute Scaled Error."""
        return ForecastEvaluator._calc_mase(y_true, y_pred, y_train, m)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Coefficient of Determination (R-squared)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - (ss_res / ss_tot))

    @staticmethod
    def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Theil's U Statistic."""
        numerator = np.sqrt(np.sum((y_true - y_pred) ** 2))
        denominator = np.sqrt(np.sum(y_true ** 2))
        if denominator == 0:
            return np.inf
        return float(numerator / denominator)

    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Bias Error."""
        return float(np.mean(y_pred - y_true))

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate forecasts using all specified metrics.

        Parameters
        ----------
        y_true : np.ndarray
            Actual values.
        y_pred : np.ndarray
            Predicted values.
        y_train : np.ndarray, optional
            Training data for MASE calculation.

        Returns
        -------
        results : dict
            Dictionary of metric names and values.
        """
        metric_calculators = {
            "mae": lambda: self.mae(y_true, y_pred),
            "mse": lambda: self.mse(y_true, y_pred),
            "rmse": lambda: self.rmse(y_true, y_pred),
            "mape": lambda: self.mape(y_true, y_pred),
            "smape": lambda: self.smape(y_true, y_pred),
            "mase": lambda: self.mase(y_true, y_pred, y_train),
            "r2": lambda: self.r2(y_true, y_pred),
            "theil_u": lambda: self.theil_u(y_true, y_pred),
            "bias": lambda: self.bias(y_true, y_pred),
        }

        results = {}
        for metric in self.metrics:
            try:
                results[metric] = round(metric_calculators[metric](), 6)
            except Exception:
                results[metric] = np.nan

        return results

    def compare_models(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        y_train: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple forecasting models.

        Parameters
        ----------
        y_true : np.ndarray
            Actual values.
        predictions : dict
            Dictionary mapping model names to predictions.
        y_train : np.ndarray, optional
            Training data for MASE calculation.

        Returns
        -------
        comparison : pd.DataFrame
            DataFrame with metrics for each model.
        """
        results = []

        for model_name, y_pred in predictions.items():
            metrics = self.evaluate(y_true, y_pred, y_train=y_train)
            metrics["model"] = model_name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df.set_index("model")
        return df


# Short metric descriptions for documentation
METRIC_DESCRIPTIONS = {
    "mae": "Mean Absolute Error - average magnitude of errors",
    "mse": "Mean Squared Error - penalizes larger errors more heavily",
    "rmse": "Root Mean Squared Error - in same units as data",
    "mape": "Mean Absolute Percentage Error - error as percentage",
    "smape": "Symmetric MAPE - handles zero values better",
    "mase": "Mean Absolute Scaled Error - compares to naive forecast",
    "r2": "R-squared - proportion of variance explained",
    "theil_u": "Theil's U - accuracy relative to naive forecast",
    "bias": "Mean Bias Error - tendency to over/under-forecast",
}


def evaluate(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    metrics: Optional[List[str]] = None,
    y_train: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, float]:
    """
    Convenience function to evaluate forecasts.

    Parameters
    ----------
    y_true : np.ndarray or pd.Series
        Actual values.
    y_pred : np.ndarray or pd.Series
        Predicted values.
    metrics : list, optional
        Metrics to compute. If None, computes all metrics.
    y_train : np.ndarray or pd.Series, optional
        Training data for MASE calculation.

    Returns
    -------
    results : dict
        Dictionary of metric names and values.

    Examples
    --------
    >>> import numpy as np
    >>> from forecasting.evaluation import evaluate
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
    >>> evaluate(y_true, y_pred, metrics=['mae', 'rmse'])
    {'mae': 0.14, 'rmse': 0.15491933...}
    """
    # Convert to numpy arrays
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_train is not None and isinstance(y_train, pd.Series):
        y_train = y_train.values

    # Validate inputs
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Compute metrics
    if metrics is None:
        metrics = ForecastEvaluator._get_all_metric_names()

    evaluator = ForecastEvaluator(metrics)
    results = evaluator.evaluate(y_true, y_pred, y_train=y_train)

    return results


def print_comparison(comparison_df: pd.DataFrame, sort_by: str = "rmse") -> None:
    """
    Print a formatted comparison of model performance.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame from ForecastEvaluator.compare_models().
    sort_by : str
        Metric to sort by (lower is better for all metrics except R2).
    """
    # Sort by the specified metric
    if sort_by in comparison_df.columns:
        if sort_by == "r2":
            # Higher is better for R2
            df = comparison_df.sort_values(sort_by, ascending=False)
        else:
            df = comparison_df.sort_values(sort_by)
    else:
        df = comparison_df

    print("\n" + "=" * 70)
    print("FORECAST MODEL COMPARISON")
    print("=" * 70)

    # Format and print
    print(df.round(4).to_string())

    # Find best model for each metric
    print("\n" + "-" * 70)
    print("Best Model per Metric:")

    for metric in df.columns:
        if metric == "r2":
            best = df[metric].idxmax()
            value = df[metric].max()
        else:
            best = df[metric].idxmin()
            value = df[metric].min()

        if not np.isnan(value):
            print(f"  {metric.upper():8s}: {best} ({value:.4f})")

    print("=" * 70)
