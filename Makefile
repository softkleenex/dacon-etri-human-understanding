.PHONY: help install lint format test run-local build-kaggle-dataset kaggle-init-code kaggle-push-code kaggle-init-notebook kaggle-push-notebook kaggle-pull-output dacon-submit

KAGGLE_DATASET_DIR := build/kaggle_dataset
KAGGLE_NOTEBOOK_DIR ?= notebooks/kaggle_runner
KAGGLE_SLUG ?= softkleenex/dacon-etri-human-runner
PULL_DIR ?= ./submissions
AUTO_SUBMIT ?= true
COMPETITION_ID ?= 236690 # Update this to your actual Dacon competition ID if needed

help:
	@echo "Available commands:"
	@echo "  make install           - Install dependencies"
	@echo "  make lint              - Run Ruff linter"
	@echo "  make format            - Run Ruff formatter"
	@echo "  make test              - Run Pytest"
	@echo "  make run-local         - Run local training and submission pipeline"
	@echo "  make build-kaggle-dataset - Package source code for Kaggle Dataset"
	@echo "  make kaggle-init-code  - Initialize Kaggle dataset metadata"
	@echo "  make kaggle-push-code  - Push packaged source code to Kaggle"
	@echo "  make kaggle-pull-output- Pull outputs from Kaggle notebook"
	@echo "  make dacon-submit      - Submit the prediction to Dacon"

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e .
	@echo "Installing Dacon API from global backup..."
	.venv/bin/pip install ~/sysp/dacon_api/dacon_submit_api-0.0.4-py3-none-any.whl

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

test:
	PYTHONPATH=. .venv/bin/pytest tests/ -v

jupyter:
	.venv/bin/jupyter lab

run-local:
	.venv/bin/python notebooks/generate_notebooks.py

build-kaggle-dataset:
	@mkdir -p $(KAGGLE_DATASET_DIR)
	@rm -rf $(KAGGLE_DATASET_DIR)/*
	@zip -r $(KAGGLE_DATASET_DIR)/source.zip src conf pyproject.toml README.md
	@if [ -f dataset-metadata.json ]; then cp dataset-metadata.json $(KAGGLE_DATASET_DIR)/; fi
	@echo "Packaged project files into $(KAGGLE_DATASET_DIR)/"

kaggle-init-code: build-kaggle-dataset
	kaggle datasets init -p $(KAGGLE_DATASET_DIR)
	cp $(KAGGLE_DATASET_DIR)/dataset-metadata.json ./
	@echo "Created dataset-metadata.json. Please edit it in the root directory and then run 'make kaggle-push-code'."

kaggle-push-code: build-kaggle-dataset
	@if [ ! -f dataset-metadata.json ]; then echo "Error: dataset-metadata.json not found in root. Run make kaggle-init-code first."; exit 1; fi
	@.venv/bin/python scripts/pre_flight_check.py
	kaggle datasets version -p $(KAGGLE_DATASET_DIR) -m "Update source code"

kaggle-init-notebook:
	@mkdir -p $(KAGGLE_NOTEBOOK_DIR)
	kaggle kernels init -p $(KAGGLE_NOTEBOOK_DIR)
	@echo "Created kernel-metadata.json in $(KAGGLE_NOTEBOOK_DIR). Please edit it and then run 'make kaggle-push-notebook'."

kaggle-push-notebook:
	@if [ "$(AUTO_SUBMIT)" = "false" ]; then \
		.venv/bin/python scripts/pre_flight_check.py --notebook_dir $(KAGGLE_NOTEBOOK_DIR) --research; \
		kaggle kernels push -p $(KAGGLE_NOTEBOOK_DIR); \
		echo "Waiting for Research Kaggle kernel to finish..."; \
		.venv/bin/python -u scripts/kaggle_manager.py --slug $(KAGGLE_SLUG) --out_dir $(PULL_DIR); \
	else \
		.venv/bin/python scripts/pre_flight_check.py --notebook_dir $(KAGGLE_NOTEBOOK_DIR); \
		kaggle kernels push -p $(KAGGLE_NOTEBOOK_DIR); \
		echo "Waiting for Submission Kaggle kernel to finish..."; \
		.venv/bin/python -u scripts/kaggle_manager.py --slug $(KAGGLE_SLUG) --out_dir $(PULL_DIR) --submit; \
	fi

kaggle-pull-output:
	kaggle kernels output $(KAGGLE_SLUG) -p $(PULL_DIR) --force
	@echo "Downloaded outputs to $(PULL_DIR) directory."

kaggle-wait-and-pull:
	@echo "Waiting for Kaggle kernel $(KAGGLE_SLUG) to finish..."
	@while [ "$$(kaggle kernels status $(KAGGLE_SLUG) | grep -c 'RUNNING')" -gt 0 ]; do \
		echo "Kernel is still running... sleeping for 60 seconds."; \
		sleep 60; \
	done; \
	echo "Kernel finished. Pulling output..."; \
	kaggle kernels output $(KAGGLE_SLUG) -p $(PULL_DIR)
	@echo "Downloaded outputs to $(PULL_DIR) directory."

ensemble:
	.venv/bin/python scripts/ensemble.py

SUB_PATH ?= "submissions/submission_ensemble_latest.csv"
SUB_MEMO ?= "Automated submission"

dacon-submit:
	@.venv/bin/python -c "import os; \
from dacon_submit_api import dacon_submit_api; \
dacon_token = os.environ.get('DACON_TOKEN'); \
team_name = os.environ.get('DACON_TEAMNAME'); \
sub_path = '$(SUB_PATH)'; \
sub_memo = '$(SUB_MEMO)'; \
assert dacon_token and team_name, 'Error: DACON_TOKEN or DACON_TEAMNAME not set. Please export them in your shell.'; \
print(f'Submitting {sub_path} to Dacon...'); \
result = dacon_submit_api.post_submission_file( \
    file_path=sub_path, \
    token=dacon_token, \
    cpt_id='$(strip $(COMPETITION_ID))', \
    team_name=team_name, \
    memo=sub_memo \
); \
print('Result:', result);"
