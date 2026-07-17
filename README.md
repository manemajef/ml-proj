# Course-Drop Prediction

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](<#>) [![Package Manager](https://img.shields.io/badge/uv-Fast%20Setup-D97706?style=flat-square)](<#>) [![Interactive Report](https://img.shields.io/badge/GitHub%20Pages-Interactive%20Notebook-blue?style=flat-square)](https://manemajef.github.io/ml-proj/) [![Status](https://img.shields.io/badge/Status-Submitted-success?style=flat-square)](<#>)

This repository contains the project submission for the Introduction to Machine Learning course within the Digital Science for High-Tech program (Department of Engineering) at Tel Aviv University (Spring 2026). The project applies the CRISP-DM methodology to predict the probability that a B2B course registration will be cancelled (`Dropped_Course`) prior to the course start date. The official guidelines and grading criteria are detailed in the [project instructions](<References/instructions.md>).

## Quick Navigation

- **Interactive Presentations**
  - [Interactive Notebook (HTML)](https://manemajef.github.io/ml-proj/notebook.html) – Full exploratory analysis, validation diagnostics, and model tuning.
  - [HTML Overview (No Code)](https://manemajef.github.io/ml-proj/overview.html) – High-level summary of findings and output figures.
  - [Markdown Notebook Export](<docs/notebook.md>) – Static markdown version.
- **Code & Scripts**
  - [Jupyter Notebook (ipynb)](<notebook.ipynb>) – Submitted notebook file (and the [Marimo script version](<notebook.py>)).
  - [Inference Pipeline (pipeline.py)](<pipeline.py>) – Production script replicating the final predictions.

## Project Overview

The prediction task is structured as a future-window forecasting problem. The dataset represents historical course bookings, where the validation and test sets are chronologically separated from the training data.

### Technical Summary

- **Categorical Normalization**: Consolidated 2,670 raw categorical levels (containing typographic variations, casing noise, and special characters) into 201 clean levels.
- **Validation Strategy**: Used adversarial validation to identify temporal and distribution drift between the training and test sets (adversarial AUC of 0.935). Consequently, a chronological split (training on 2015-2016, validating on 2017) was used instead of random cross-validation.
- **Feature Engineering**: Formulated domain-specific indicators including cancellation histories (`prev_drop_rate`), lab assignment matches (`got_requested_lab`), group composition features, and linear time indices (`days_since_epoch`). High-cardinality identifiers (Agent, Country, Company) were represented using frequency encoding.
- **Model Blend**: Evaluated Logistic Regression (AUC 0.881), Multi-Layer Perceptron (AUC 0.877), and gradient boosted tree architectures. The final pipeline implements a rank-average ensemble of XGBoost, LightGBM, and CatBoost (validation AUC of 0.9159, Average Precision of 0.897).
- **Submission Details**: The current scored submission achieves a pre-submission test AUC score of 0.889314.

## Repository Structure

- [submission](./submission) - Files submitted for grading:
  - [Group_27_Notebook.ipynb](<submission/Group_27_Notebook.ipynb>) - Jupyter notebook containing EDA, training, and evaluation.
  - [Group_27_Submission.csv](<submission/Group_27_Submission.csv>) - Pre-submission prediction CSV.
  - [submission-report.md](<submission/submission-report.md>) - Project report (written in Hebrew).
  - [Train_Data.csv](<submission/Train_Data.csv>) and [Test_Data_No_Target.csv](<submission/Test_Data_No_Target.csv>) - Source datasets.
- [docs](./docs) - Documentation and interactive views:
  - [index.html](<docs/index.html>) - Landing page for GitHub Pages.
  - [notebook.html](https://manemajef.github.io/ml-proj/notebook.html) - Interactive notebook export.
  - [overview.html](https://manemajef.github.io/ml-proj/overview.html) - HTML overview containing prose and figures without code blocks.
  - [notebook.md](<docs/notebook.md>) - Markdown copy of the notebook.
- [versions](./versions) - Archived iterations and previous pipeline/notebook versions (`v1`, `v2`, `v3`, `ron_version`).
- [pipeline.py](<pipeline.py>) - Script containing the inference pipeline.
- [pyproject.toml](<pyproject.toml>) and [uv.lock](<uv.lock>) - Python dependencies managed via `uv`.
- [References](./References) - Original assignments, instructions, and course materials.

## Setup and Execution

The project requires Python 3.13 and uses `uv` for dependency management.

### Environment Setup

Sync the dependencies and build the virtual environment:

```bash
uv sync
```

### Running the Notebook

Marimo version:

```bash
uv run marimo edit notebook.py
```

Jupyter version:

```bash
uv run --with jupyter jupyter notebook notebook.ipynb
```

### Running the Pipeline

To run a dry-run check of the pipeline without writing predictions:

```bash
uv run python pipeline.py
```

To run the pipeline and output a prediction file:

```bash
uv run python pipeline.py --write --out data/tmp_submission.csv
```
