"""Sample datasets embedded as Python data for testing without external files."""

import pandas as pd
import numpy as np


def get_retail_sales_sample():
    """Get a small sample of retail sales data."""
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2022-12-31", freq="D")
    n = len(dates)

    trend = np.linspace(100, 130, n)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 5, n)

    sales = trend + weekly_seasonality + yearly_seasonality + noise
    return pd.Series(sales, index=dates, name="sales")


def get_website_traffic_sample():
    """Get a small sample of website traffic data."""
    np.random.seed(43)
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="H")
    n = len(dates)

    base_traffic = 1000
    daily_pattern = 300 * np.sin(2 * np.pi * np.arange(n) / 24 - np.pi / 2)
    noise = np.random.normal(0, 50, n)

    traffic = base_traffic + daily_pattern + noise
    traffic = np.maximum(traffic, 100)
    return pd.Series(traffic, index=dates, name="traffic")


# Make sample data available as DataFrames
SAMPLE_DATASETS = {
    "retail_sales": get_retail_sales_sample,
    "website_traffic": get_website_traffic_sample,
}
