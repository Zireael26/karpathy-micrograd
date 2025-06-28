# Makefile for micrograd project

.PHONY: help install test clean lint run-examples setup dev-install

# Default target
help:
	@echo "Available commands:"
	@echo "  setup       - Initial project setup (install dependencies)"
	@echo "  install     - Install the package in development mode"
	@echo "  test        - Run all tests"
	@echo "  clean       - Clean up build artifacts"
	@echo "  lint        - Run linting tools"
	@echo "  examples    - Run all examples"
	@echo "  notebook    - Start Jupyter notebook"
	@echo "  dev-install - Install development dependencies"

# Initial setup
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Install package in development mode
install:
	pip install -e .

# Development installation with extra dependencies
dev-install:
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Linting
lint:
	flake8 micrograd/ tests/ examples/
	black --check micrograd/ tests/ examples/
	isort --check-only micrograd/ tests/ examples/

# Format code
format:
	black micrograd/ tests/ examples/
	isort micrograd/ tests/ examples/

# Run examples
examples:
	python examples/basic_operations.py
	python examples/neural_network.py

# Start Jupyter notebook
notebook:
	jupyter notebook

# Install package and run tests (CI-like)
ci: setup test

# Show project structure
structure:
	tree -I '__pycache__|*.pyc|.git|.venv|node_modules'
