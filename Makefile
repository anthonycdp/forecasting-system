.PHONY: help install test lint format clean data run

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy

test:  ## Run tests
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage
	python -m pytest tests/ --cov=forecasting --cov-report=html --cov-report=term

test-slow:  ## Run tests including slow tests
	python -m pytest tests/ -v --runslow

lint:  ## Run linters
	flake8 forecasting/ tests/
	mypy forecasting/

format:  ## Format code
	black forecasting/ tests/
	isort forecasting/ tests/

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf output/
	rm -f .coverage

data:  ## Generate sample data
	python data/create_sample.py

run:  ## Run the main pipeline
	python main.py

run-custom:  ## Run with custom parameters (usage: make run-custom ARGS="--test-size 60")
	python main.py $(ARGS)

notebook:  ## Start Jupyter notebook
	jupyter notebook notebooks/

requirements:  ## Generate requirements.txt
	pip freeze > requirements.txt
