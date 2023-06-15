"""
Sample data generation script for the forecasting system.

Creates CSV files with sample time series data for testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_sample_datasets():
    """Create multiple sample datasets for the forecasting system."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)

    # 1. Retail sales data (daily)
    np.random.seed(42)
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    n = len(dates)

    trend = np.linspace(100, 150, n)
    weekly_seasonality = 10 * np.sin(2 * np.pi * np.arange(n) / 7)
    yearly_seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = np.random.normal(0, 5, n)

    sales = trend + weekly_seasonality + yearly_seasonality + noise
    sales = pd.Series(sales, index=dates, name="sales")

    df_sales = pd.DataFrame({
        "date": dates,
        "sales": sales.values,
    })
    df_sales.to_csv(output_dir / "retail_sales.csv", index=False)
    print(f"Created retail_sales.csv ({len(df_sales)} rows)")

    # 2. Website traffic data (hourly)
    np.random.seed(43)
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="H")
    n = len(dates)

    base_traffic = 1000
    daily_pattern = 300 * np.sin(2 * np.pi * np.arange(n) / 24 - np.pi / 2)
    weekly_pattern = 200 * np.where(
        np.array([d.weekday() for d in dates]) < 5, 1, -0.5
    )
    trend = np.linspace(0, 500, n)
    noise = np.random.normal(0, 50, n)

    traffic = base_traffic + daily_pattern + weekly_pattern + trend + noise
    traffic = np.maximum(traffic, 100)  # Ensure positive
    traffic = pd.Series(traffic, index=dates, name="traffic")

    df_traffic = pd.DataFrame({
        "date": dates,
        "traffic": traffic.values,
    })
    df_traffic.to_csv(output_dir / "website_traffic.csv", index=False)
    print(f"Created website_traffic.csv ({len(df_traffic)} rows)")

    # 3. Stock price data (daily business days)
    np.random.seed(44)
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="B")  # Business days
    n = len(dates)

    returns = np.random.normal(0.001, 0.02, n)
    price = 100 * np.exp(np.cumsum(returns))

    df_stock = pd.DataFrame({
        "date": dates,
        "price": price,
    })
    df_stock.to_csv(output_dir / "stock_prices.csv", index=False)
    print(f"Created stock_prices.csv ({len(df_stock)} rows)")

    # 4. Energy consumption data (daily)
    np.random.seed(45)
    dates = pd.date_range("2022-01-01", "2023-12-31", freq="D")
    n = len(dates)

    base_consumption = 500
    yearly_seasonality = 150 * np.cos(2 * np.pi * np.arange(n) / 365)
    weekly_pattern = 50 * np.where(
        np.array([d.weekday() for d in dates]) < 5, 1, -1
    )
    temp_effect = -30 * np.cos(2 * np.pi * np.arange(n) / 365)  # Heating/cooling
    noise = np.random.normal(0, 30, n)

    consumption = base_consumption + yearly_seasonality + weekly_pattern + temp_effect + noise
    consumption = pd.Series(consumption, index=dates, name="consumption")

    df_energy = pd.DataFrame({
        "date": dates,
        "consumption": consumption.values,
    })
    df_energy.to_csv(output_dir / "energy_consumption.csv", index=False)
    print(f"Created energy_consumption.csv ({len(df_energy)} rows)")

    print(f"\nAll sample datasets saved to {output_dir}/")


if __name__ == "__main__":
    create_sample_datasets()
