.PHONY: help install lint format test run-local

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make lint        - Run Ruff linter"
	@echo "  make format      - Run Ruff formatter"
	@echo "  make test        - Run Pytest"
	@echo "  make run-local   - Run local training and submission pipeline"

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

test:
	.venv/bin/pytest tests/ -v

run-local:
	.venv/bin/python scripts/train.py
	.venv/bin/python scripts/inference.py
