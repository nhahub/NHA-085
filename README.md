# Rossmann Sales Forecasting with ML

This repository contains a Jupyter notebook (`Rossman_Sales_Forecasting with ML.ipynb`) that performs EDA, preprocessing, and forecasting (SARIMAX, XGBoost, LightGBM) on the Rossmann dataset. Command-line helper scripts are provided so you can reproduce model training and prediction outside the notebook.

Quick reproduction (Windows PowerShell)

1. Create a Python virtual environment and install dependencies (unpinned):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2. (Optional) For bit-for-bit reproducibility install pinned versions:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-frozen.txt
```

3. Train SARIMAX (weekly aggregated) or LightGBM:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --data-dir dataset --out models/sarimax_artifacts.joblib --model sarimax

# or
.\.venv\Scripts\python.exe -m scripts.train --data-dir dataset --out models/lgbm_artifacts.joblib --model lgbm
```

4. Predict using the trained artifact:

```powershell
.\.venv\Scripts\python.exe -m scripts.predict --artifacts models/sarimax_artifacts.joblib --data dataset/test.csv --out results/sarimax_predictions.csv
```

Important notes before publishing
- The repository expects the dataset directory `dataset/` to contain `train.csv`, `store.csv`, and `test.csv`. The dataset is not included here. Provide instructions or a download link in a private/internal repo if the dataset is not public.
- `requirements-frozen.txt` pins exact versions from a tested environment (created here). Use it to reproduce installs.
- Add `sample_data/` if you want CI to run a full smoke test; otherwise CI must be configured to skip expensive steps.

See `README_DEPLOY.md` for additional deploy/run instructions and `scripts/` for CLI helpers.
