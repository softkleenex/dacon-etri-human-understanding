# Notebooks

This directory contains Jupyter notebooks used for the competition.

## Structure

- `01-EDA/`: For local or Kaggle-based Exploratory Data Analysis. Keep these notebooks clean and documented.
- `99-kaggle_runner/`: Contains the main notebook(s) intended to be executed on Kaggle Kernels for heavy training or final inference.

## Kaggle Workflow

To use the `99-kaggle_runner` effectively:

1. **Initialize Kernel Metadata**:
   Run `make kaggle-init-notebook` to generate a `kernel-metadata.json` file in `notebooks/99-kaggle_runner/`. Edit this file to set your username and kernel name.
2. **Push Source Code**:
   If your notebook depends on the local `src/` directory, initialize a dataset metadata there (`make kaggle-init-code`) and push it to Kaggle (`make kaggle-push-code`).
3. **Attach Code to Notebook**:
   In your Kaggle kernel settings (via web UI or API), attach the dataset you just uploaded. The template `main.ipynb` shows how to add this dataset to your Python path.
4. **Push Notebook**:
   Run `make kaggle-push-notebook` to trigger a run on Kaggle's servers.
5. **Retrieve Output**:
   Once finished, run `make kaggle-pull-output` to download the trained models and submission files.