"""
Time series data generation for testing and demonstration.

Provides tools for generating synthetic time series with various
patterns and characteristics.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class TimeSeriesGenerator:
    """
    Generator for synthetic time series data.

    Creates time series with configurable patterns including:
    - Trend (linear, exponential, or polynomial)
    - Seasonality (multiple seasonal patterns)
    - Noise (Gaussian, t-distributed)
    - Outliers
    - Changepoints

    Parameters
    ----------
    n_samples : int
        Number of time points to generate.
    start_date : str, default='2020-01-01'
        Start date for the time index.
    freq : str, default='D'
        Frequency of the time series (e.g., 'D' for daily, 'H' for hourly).

    Examples
    --------
    >>> from forecasting.data_generator import TimeSeriesGenerator
    >>> gen = TimeSeriesGenerator(n_samples=365)
    >>> y = gen.generate(trend='linear', seasonality=7)
    >>> y.shape
    (365,)
    """

    def __init__(
        self,
        n_samples: int,
        start_date: str = "2020-01-01",
        freq: str = "D",
        random_seed: Optional[int] = None,
    ):
        if n_samples < 1:
            raise ValueError("n_samples must be a positive integer")

        self.n_samples = n_samples
        self.start_date = start_date
        self.freq = freq
        self.random_seed = random_seed

        # Generate time index
        self.dates = pd.date_range(
            start=start_date, periods=n_samples, freq=freq
        )
        self.t = np.arange(n_samples)

    def generate(
        self,
        trend: Optional[str] = None,
        trend_strength: float = 1.0,
        seasonality: Optional[Union[int, List[int]]] = None,
        seasonality_strength: float = 1.0,
        noise: bool = True,
        noise_std: float = 0.1,
        noise_type: str = "gaussian",
        outliers: bool = False,
        outlier_fraction: float = 0.01,
        changepoints: bool = False,
        n_changepoints: int = 3,
        changepoint_strength: float = 0.5,
        base_level: float = 100.0,
    ) -> pd.Series:
        """
        Generate a synthetic time series.

        Parameters
        ----------
        trend : str, optional
            Type of trend: 'linear', 'exponential', 'polynomial', or None.
        trend_strength : float
            Strength of the trend component.
        seasonality : int or list, optional
            Seasonal period(s). Can be a single int or list of ints.
            For example, 7 for weekly, 365 for yearly in daily data.
        seasonality_strength : float
            Strength of the seasonal component.
        noise : bool
            Whether to add noise.
        noise_std : float
            Standard deviation of noise (relative to base level).
        noise_type : str
            Type of noise: 'gaussian' or 'student_t'.
        outliers : bool
            Whether to add outliers.
        outlier_fraction : float
            Fraction of points to be outliers.
        changepoints : bool
            Whether to add trend changepoints.
        n_changepoints : int
            Number of changepoints.
        changepoint_strength : float
            Strength of changepoint effects.
        base_level : float
            Base level of the time series.

        Returns
        -------
        y : pd.Series
            Generated time series with DatetimeIndex.
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        y = np.full(self.n_samples, base_level)

        # Add trend
        if trend is not None:
            y = y + self._generate_trend(
                trend, trend_strength, base_level, changepoints, n_changepoints, changepoint_strength
            )

        # Add seasonality
        if seasonality is not None:
            y = y + self._generate_seasonality(seasonality, seasonality_strength, base_level)

        # Add noise
        if noise:
            y = y + self._generate_noise(noise_std * base_level, noise_type)

        # Add outliers
        if outliers:
            y = self._add_outliers(y, outlier_fraction)

        return pd.Series(y, index=self.dates)

    def _generate_trend(
        self,
        trend_type: str,
        strength: float,
        base_level: float,
        changepoints: bool,
        n_changepoints: int,
        changepoint_strength: float,
    ) -> np.ndarray:
        """Generate trend component."""
        t_normalized = self.t / self.n_samples

        if trend_type == "linear":
            trend = strength * t_normalized * base_level
        elif trend_type == "exponential":
            trend = strength * (np.exp(t_normalized) - 1) * base_level
        elif trend_type == "polynomial":
            trend = strength * (t_normalized ** 2) * base_level
        else:
            raise ValueError(f"Unknown trend type: {trend_type}")

        # Add changepoints
        if changepoints and n_changepoints > 0:
            changepoint_indices = np.random.choice(
                range(self.n_samples // 4, 3 * self.n_samples // 4),
                size=min(n_changepoints, self.n_samples // 2),
                replace=False,
            )

            for idx in sorted(changepoint_indices):
                # Random change in trend direction
                change = np.random.uniform(-1, 1) * changepoint_strength * base_level
                trend[idx:] += change * np.linspace(0, 1, self.n_samples - idx)

        return trend

    def _generate_seasonality(
        self,
        periods: Union[int, List[int]],
        strength: float,
        base_level: float,
    ) -> np.ndarray:
        """Generate seasonal component."""
        if isinstance(periods, int):
            periods = [periods]

        seasonal = np.zeros(self.n_samples)

        for period in periods:
            # Random phase for each seasonal component
            phase = np.random.uniform(0, 2 * np.pi)
            # Amplitude proportional to period (longer periods = larger swings)
            amplitude = strength * base_level * np.sqrt(period / self.n_samples)
            seasonal += amplitude * np.sin(2 * np.pi * self.t / period + phase)

        return seasonal

    def _generate_noise(self, std: float, noise_type: str) -> np.ndarray:
        """Generate noise component."""
        if noise_type == "gaussian":
            return np.random.normal(0, std, self.n_samples)
        elif noise_type == "student_t":
            # Heavier tails
            return np.random.standard_t(df=3, size=self.n_samples) * std
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def _add_outliers(self, y: np.ndarray, fraction: float) -> np.ndarray:
        """Add outliers to the series."""
        y = y.copy()
        n_outliers = int(self.n_samples * fraction)

        if n_outliers > 0:
            outlier_indices = np.random.choice(
                self.n_samples, size=n_outliers, replace=False
            )
            outlier_magnitudes = np.random.choice([-1, 1], size=n_outliers) * np.random.uniform(
                3, 5, size=n_outliers
            ) * np.std(y)
            y[outlier_indices] += outlier_magnitudes

        return y

    def generate_multiple(
        self,
        n_series: int,
        **kwargs,
    ) -> List[pd.Series]:
        """
        Generate multiple time series.

        Parameters
        ----------
        n_series : int
            Number of series to generate.
        **kwargs : dict
            Arguments passed to generate().

        Returns
        -------
        series_list : list
            List of generated pd.Series.
        """
        series_list = []

        for i in range(n_series):
            if self.random_seed is not None:
                np.random.seed(self.random_seed + i)
            series_list.append(self.generate(**kwargs))

        return series_list


def generate_sample_data(
    n_samples: int = 365 * 2,
    freq: str = "D",
    start_date: str = "2020-01-01",
    name: str = "sample",
    random_seed: int = 42,
) -> pd.Series:
    """
    Generate sample time series data with common patterns.

    Parameters
    ----------
    n_samples : int
        Number of time points.
    freq : str
        Frequency string.
    start_date : str
        Start date.
    name : str
        Name for the series.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    y : pd.Series
        Generated sample data.

    Examples
    --------
    >>> from forecasting.data_generator import generate_sample_data
    >>> y = generate_sample_data(n_samples=365)
    >>> len(y)
    365
    """
    gen = TimeSeriesGenerator(
        n_samples=n_samples,
        start_date=start_date,
        freq=freq,
        random_seed=random_seed,
    )

    y = gen.generate(
        trend="linear",
        trend_strength=0.5,
        seasonality=[7, 365],  # Weekly and yearly
        seasonality_strength=0.3,
        noise=True,
        noise_std=0.02,
        changepoints=True,
        n_changepoints=2,
        changepoint_strength=0.2,
    )

    y.name = name
    return y


def generate_m4_style_data(
    n_samples: int,
    freq: str = "M",
    seasonal_period: int = 12,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """
    Generate time series similar to M4 competition data.

    The M4 competition featured time series with various seasonal patterns
    and trend characteristics.

    Parameters
    ----------
    n_samples : int
        Number of time points.
    freq : str
        Frequency string.
    seasonal_period : int
        Seasonal period (e.g., 12 for monthly, 4 for quarterly).
    random_seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    y : pd.Series
        Generated M4-style data.
    """
    gen = TimeSeriesGenerator(
        n_samples=n_samples,
        freq=freq,
        random_seed=random_seed,
    )

    y = gen.generate(
        trend="linear",
        trend_strength=0.3,
        seasonality=seasonal_period,
        seasonality_strength=0.4,
        noise=True,
        noise_std=0.03,
    )

    return y


def load_example_data() -> Tuple[pd.Series, str]:
    """
    Load an example dataset for demonstration.

    Returns
    -------
    y : pd.Series
        Example time series data.
    description : str
        Description of the dataset.
    """
    # Generate airline-style data (classic Box-Jenkins example)
    gen = TimeSeriesGenerator(
        n_samples=144,  # 12 years of monthly data
        start_date="1949-01-01",
        freq="M",
        random_seed=42,
    )

    # Generate with multiplicative-like seasonality (using exponential trend)
    y = gen.generate(
        trend="exponential",
        trend_strength=0.5,
        seasonality=12,
        seasonality_strength=0.15,
        noise=True,
        noise_std=0.02,
        base_level=100.0,
    )

    y.name = "passengers"

    description = """
    Airline Passengers Dataset (Synthetic Recreation)

    This synthetic dataset mimics the classic Box-Jenkins airline data,
    featuring exponential growth and multiplicative seasonality.

    - Period: 1949-01 to 1960-12 (144 months)
    - Frequency: Monthly
    - Pattern: Exponential trend with yearly seasonality
    """

    return y, description
