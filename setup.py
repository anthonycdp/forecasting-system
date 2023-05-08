"""Setup configuration for the forecasting package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="forecasting",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive time series forecasting library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/forecasting-system",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "statistical": [
            "statsmodels>=0.13.0",
            "prophet>=1.1.0",
        ],
        "ml": [
            "xgboost>=1.5.0",
        ],
        "deep-learning": [
            "tensorflow>=2.8.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
)
