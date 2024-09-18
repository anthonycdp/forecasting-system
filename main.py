#!/usr/bin/env python3
"""
Time Series Forecasting Pipeline

Demonstrates a complete forecasting workflow including:
- Data generation/loading
- Model training and comparison
- Backtesting
- Evaluation and visualization

Usage:
    python main.py
"""

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from forecasting.baseline import NaiveForecaster, MeanForecaster, DriftForecaster
from forecasting.data_generator import TimeSeriesGenerator, generate_sample_data
from forecasting.evaluation import ForecastEvaluator, evaluate, print_comparison
from forecasting.backtesting import TimeSeriesCrossValidator, BacktestEngine, run_backtest


# Configuration Constants
class Config:
    """Centralized configuration for the forecasting pipeline."""
    DEFAULT_TEST_SIZE = 30
    DEFAULT_N_SPLITS = 5
    DEFAULT_HORIZON = 14
    SAMPLE_DATA_DAYS = 730  # 2 years
    TRAINING_DISPLAY_POINTS = 100
    XGBOOST_LAGS = 12
    XGBOOST_N_ESTIMATORS = 100
    LSTM_LAGS = 12
    LSTM_EPOCHS = 50
    FIGURE_DPI = 150


class ModelFactory:
    """Factory for creating forecasting models with dependency checking."""

    @staticmethod
    def create_all() -> Dict[str, object]:
        """Create all available forecasting models."""
        models = ModelFactory._create_baseline_models()
        models.update(ModelFactory._create_statistical_models())
        models.update(ModelFactory._create_ml_models())
        return models

    @staticmethod
    def _create_baseline_models() -> Dict[str, object]:
        return {
            "Naive": NaiveForecaster(strategy="last"),
            "Mean": MeanForecaster(),
            "Drift": DriftForecaster(),
        }

    @staticmethod
    def _create_statistical_models() -> Dict[str, object]:
        models = {}
        ModelFactory._try_add_model(
            models, "ARIMA(1,1,1)",
            lambda: ModelFactory._create_arima(),
            "statsmodels"
        )
        ModelFactory._try_add_model(
            models, "Prophet",
            lambda: ModelFactory._create_prophet(),
            "prophet"
        )
        return models

    @staticmethod
    def _create_ml_models() -> Dict[str, object]:
        models = {}
        ModelFactory._try_add_model(
            models, "XGBoost",
            lambda: ModelFactory._create_xgboost(),
            "xgboost"
        )
        ModelFactory._try_add_model(
            models, "LSTM",
            lambda: ModelFactory._create_lstm(),
            "tensorflow"
        )
        return models

    @staticmethod
    def _try_add_model(
        models: Dict[str, object],
        name: str,
        factory_func: callable,
        dependency_name: str
    ) -> None:
        """Safely add a model if its dependency is available."""
        try:
            models[name] = factory_func()
            logger.info(f"{name} model available")
        except ImportError:
            logger.warning(f"{dependency_name} not available - {name} model disabled")

    @staticmethod
    def _create_arima():
        from forecasting.statistical import ARIMAForecaster
        return ARIMAForecaster(order=(1, 1, 1))

    @staticmethod
    def _create_prophet():
        from prophet import Prophet
        from forecasting.statistical import ProphetForecaster
        return ProphetForecaster()

    @staticmethod
    def _create_xgboost():
        from forecasting.ml_forecaster import XGBoostForecaster
        return XGBoostForecaster(
            lags=Config.XGBOOST_LAGS,
            n_estimators=Config.XGBOOST_N_ESTIMATORS
        )

    @staticmethod
    def _create_lstm():
        import tensorflow as tf
        from forecasting.ml_forecaster import LSTMForecaster
        return LSTMForecaster(
            lags=Config.LSTM_LAGS,
            epochs=Config.LSTM_EPOCHS,
            verbose=0
        )


class ModelEvaluator:
    """Handles model fitting, prediction, and evaluation."""

    def __init__(self):
        self.evaluator = ForecastEvaluator()

    def fit_and_predict(
        self,
        model: object,
        y_train: pd.Series,
        horizon: int
    ) -> Optional[np.ndarray]:
        """Fit a model and generate predictions."""
        model.fit(y_train)
        return model.predict(horizon)

    def calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    def evaluate_predictions(
        self,
        y_test: np.ndarray,
        predictions: Dict[str, np.ndarray],
        y_train: np.ndarray
    ) -> pd.DataFrame:
        """Evaluate all model predictions."""
        return self.evaluator.compare_models(y_test, predictions, y_train=y_train)


class SingleSplitExperiment:
    """Runs a single train/test split experiment."""

    def __init__(self):
        self.model_evaluator = ModelEvaluator()

    def run(
        self,
        y: pd.Series,
        test_size: int
    ) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], pd.Series]:
        """Run the single split experiment."""
        y_train, y_test = self._split_data(y, test_size)
        logger.info(f"Training size: {len(y_train)}, Test size: {len(y_test)}")

        models = ModelFactory.create_all()
        predictions = self._fit_all_models(models, y_train, y_test, test_size)
        comparison_df = self.model_evaluator.evaluate_predictions(
            y_test.values, predictions, y_train.values
        )

        return comparison_df, predictions, y_test

    def _split_data(
        self,
        y: pd.Series,
        test_size: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Split data into training and test sets."""
        return y.iloc[:-test_size], y.iloc[-test_size:]

    def _fit_all_models(
        self,
        models: Dict[str, object],
        y_train: pd.Series,
        y_test: pd.Series,
        horizon: int
    ) -> Dict[str, np.ndarray]:
        """Fit all models and collect predictions."""
        predictions = {}
        for name, model in models.items():
            prediction = self._fit_single_model(name, model, y_train, y_test, horizon)
            if prediction is not None:
                predictions[name] = prediction
        return predictions

    def _fit_single_model(
        self,
        name: str,
        model: object,
        y_train: pd.Series,
        y_test: pd.Series,
        horizon: int
    ) -> Optional[np.ndarray]:
        """Fit a single model and return predictions."""
        logger.info(f"Fitting {name}...")
        try:
            prediction = self.model_evaluator.fit_and_predict(model, y_train, horizon)
            rmse = self.model_evaluator.calculate_rmse(y_test.values, prediction)
            logger.info(f"  {name} - RMSE: {rmse:.4f}")
            return prediction
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            return None


class BacktestExperiment:
    """Runs cross-validation backtesting experiment."""

    def run(
        self,
        y: pd.Series,
        n_splits: int,
        horizon: int
    ) -> pd.DataFrame:
        """Run the backtest experiment."""
        logger.info(f"Running backtest with {n_splits} splits, horizon={horizon}")

        models = ModelFactory.create_all()
        cv = TimeSeriesCrossValidator(
            n_splits=n_splits,
            horizon=horizon,
            min_train_size=len(y) // 4,
        )
        engine = BacktestEngine(cv)

        results = {}
        for name, model in models.items():
            results[name] = self._run_single_backtest(name, model, engine, y)

        return pd.DataFrame(results).T

    def _run_single_backtest(
        self,
        name: str,
        model: object,
        engine: BacktestEngine,
        y: pd.Series
    ) -> Dict:
        """Run backtest for a single model."""
        logger.info(f"Backtesting {name}...")
        try:
            result = engine.run_backtest(model, y)
            return self._extract_backtest_metrics(name, result)
        except Exception as e:
            logger.error(f"  {name} failed: {e}")
            return {"error": str(e)}

    def _extract_backtest_metrics(
        self,
        name: str,
        result: Dict
    ) -> Dict:
        """Extract metrics from backtest result."""
        if not result["mean_metrics"]:
            logger.error(f"  {name} - All folds failed")
            return {"error": "All folds failed"}

        logger.info(f"  {name} - Mean RMSE: {result['mean_metrics']['rmse']:.4f}")
        return {
            **{f"mean_{k}": v for k, v in result["mean_metrics"].items()},
            **{f"std_{k}": v for k, v in result["std_metrics"].items()},
        }


class VisualizationService:
    """Handles all plotting operations."""

    @staticmethod
    def plot_forecasts(
        y: pd.Series,
        predictions: Dict[str, np.ndarray],
        y_test: pd.Series,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot actual values and forecasts."""
        plt.figure(figsize=(14, 6))

        VisualizationService._plot_training_data(y, y_test)
        VisualizationService._plot_test_data(y_test)
        VisualizationService._plot_predictions(y_test, predictions)

        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title("Time Series Forecast Comparison")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        VisualizationService._save_or_show(output_path)

    @staticmethod
    def _plot_training_data(y: pd.Series, y_test: pd.Series) -> None:
        """Plot the training data portion."""
        n_points = min(Config.TRAINING_DISPLAY_POINTS, len(y) - len(y_test))
        train_data = y.iloc[-(n_points + len(y_test)) : -len(y_test)]
        plt.plot(train_data.index, train_data.values, "k-", label="Training Data", linewidth=1.5)

    @staticmethod
    def _plot_test_data(y_test: pd.Series) -> None:
        """Plot the test data points."""
        plt.plot(y_test.index, y_test.values, "ko", label="Actual", markersize=6)

    @staticmethod
    def _plot_predictions(y_test: pd.Series, predictions: Dict[str, np.ndarray]) -> None:
        """Plot all model predictions."""
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        for (name, pred), color in zip(predictions.items(), colors):
            plt.plot(y_test.index, pred, "-", label=name, linewidth=1.5, color=color)

    @staticmethod
    def plot_comparison_heatmap(
        comparison_df: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> None:
        """Plot a heatmap of model comparison results."""
        metrics_for_heatmap = ["mae", "rmse", "mape", "mase"]
        available_metrics = [m for m in metrics_for_heatmap if m in comparison_df.columns]

        if not available_metrics:
            logger.warning("No metrics available for heatmap")
            return

        data = comparison_df[available_metrics].copy()
        data = VisualizationService._normalize_metrics(data)

        plt.figure(figsize=(10, len(data) * 0.8 + 2))
        sns.heatmap(data, annot=True, cmap="RdYlGn_r", fmt=".3f", linewidths=0.5)
        plt.title("Model Comparison (Lower is Better)")
        plt.tight_layout()

        VisualizationService._save_or_show(output_path)

    @staticmethod
    def _normalize_metrics(data: pd.DataFrame) -> pd.DataFrame:
        """Normalize metrics to [0, 1] range for comparison."""
        for col in data.columns:
            min_val, max_val = data[col].min(), data[col].max()
            if max_val > min_val:
                data[col] = (data[col] - min_val) / (max_val - min_val)
        return data

    @staticmethod
    def _save_or_show(output_path: Optional[Path]) -> None:
        """Save figure to file or display it."""
        if output_path:
            plt.savefig(output_path, dpi=Config.FIGURE_DPI, bbox_inches="tight")
            logger.info(f"Figure saved to {output_path}")
        else:
            plt.show()
        plt.close()


class DataLoader:
    """Handles data loading and generation."""

    @staticmethod
    def load(data_source: str) -> pd.Series:
        """Load or generate time series data."""
        if data_source == "sample":
            return DataLoader._generate_sample_data()
        return DataLoader._load_from_file(data_source)

    @staticmethod
    def _generate_sample_data() -> pd.Series:
        """Generate synthetic sample data."""
        logger.info("Generating sample data...")
        y = generate_sample_data(n_samples=Config.SAMPLE_DATA_DAYS, random_seed=42)
        logger.info(f"Generated {len(y)} observations")
        return y

    @staticmethod
    def _load_from_file(filepath: str) -> pd.Series:
        """Load data from a CSV file."""
        logger.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)

        if "date" in df.columns and "value" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            y = df.set_index("date")["value"]
        else:
            y = df.iloc[:, -1]

        logger.info(f"Loaded {len(y)} observations")
        return y


class Pipeline:
    """Main forecasting pipeline orchestrator."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run(self) -> None:
        """Execute the full forecasting pipeline."""
        y = DataLoader.load(self.args.data)

        self._print_header(y)
        self._run_single_split_evaluation(y)
        self._run_cross_validation_evaluation(y)
        self._print_footer()

    def _print_header(self, y: pd.Series) -> None:
        """Print pipeline header information."""
        print("\n" + "=" * 70)
        print("TIME SERIES FORECASTING PIPELINE")
        print("=" * 70)
        print(f"\nData: {len(y)} observations from {y.index[0]} to {y.index[-1]}")
        print(f"Test size: {self.args.test_size}")
        print(f"Cross-validation: {self.args.n_splits} splits, {self.args.horizon}-step horizon")

    def _run_single_split_evaluation(self, y: pd.Series) -> None:
        """Run and display single split evaluation."""
        print("\n" + "-" * 70)
        print("SINGLE SPLIT EVALUATION")
        print("-" * 70)

        experiment = SingleSplitExperiment()
        comparison_df, predictions, y_test = experiment.run(y, self.args.test_size)

        print_comparison(comparison_df)

        if not self.args.no_plot:
            VisualizationService.plot_forecasts(
                y, predictions, y_test,
                output_path=self.output_dir / "forecasts.png"
            )
            VisualizationService.plot_comparison_heatmap(
                comparison_df,
                output_path=self.output_dir / "comparison_heatmap.png"
            )

        comparison_df.to_csv(self.output_dir / "single_split_results.csv")

    def _run_cross_validation_evaluation(self, y: pd.Series) -> None:
        """Run and display cross-validation evaluation."""
        print("\n" + "-" * 70)
        print("CROSS-VALIDATION EVALUATION")
        print("-" * 70)

        experiment = BacktestExperiment()
        backtest_comparison = experiment.run(y, self.args.n_splits, self.args.horizon)

        print("\nMean Metrics Across Folds:")
        mean_cols = [c for c in backtest_comparison.columns if c.startswith("mean_")]
        print(backtest_comparison[mean_cols].round(4).to_string())

        backtest_comparison.to_csv(self.output_dir / "backtest_results.csv")

    def _print_footer(self) -> None:
        """Print pipeline footer."""
        print("\n" + "=" * 70)
        print(f"Results saved to {self.output_dir}/")
        print("=" * 70)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    parser.add_argument(
        "--data",
        type=str,
        default="sample",
        help="Data source: 'sample' or path to CSV file",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=Config.DEFAULT_TEST_SIZE,
        help="Number of observations for testing",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=Config.DEFAULT_N_SPLITS,
        help="Number of cross-validation splits",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=Config.DEFAULT_HORIZON,
        help="Forecast horizon",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory for output files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plotting",
    )
    return parser.parse_args()


def main():
    """Main entry point for the forecasting pipeline."""
    args = parse_arguments()
    pipeline = Pipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
