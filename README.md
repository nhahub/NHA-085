# Rossmann Sales Forecasting with ML

This repository contains a Jupyter notebook (`Rossman_Sales_Forecasting with ML.ipynb`) that performs EDA, preprocessing, and forecasting (SARIMAX, XGBoost, LightGBM) on the Rossmann dataset. Command-line helper scripts are provided so you can reproduce model training and prediction outside the notebook. A small Dash UI (`app.py`) lets you run SARIMAX forecasts and view/download predictions.

Quick reproduction (Windows PowerShell)

1. Create a Python virtual environment and install dependencies (recommended)

Open a terminal in the repository root (the folder that contains this `README.md`). The examples below are explicit for Windows PowerShell and also show equivalent cross-platform notes.

PowerShell (explicit venv + install):

```powershell
# From the repository root
python -m venv .venv

# Activate in PowerShell (if blocked run the Set-ExecutionPolicy line first)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .venv\Scripts\Activate.ps1

# upgrade pip and install dependencies using the venv python
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Alternative activation methods

- cmd.exe: `.venv\Scripts\activate.bat` then run `pip install -r requirements.txt`
- macOS / Linux (bash/zsh): `python -m venv .venv; source .venv/bin/activate; python -m pip install -r requirements.txt`

2. (Optional) For exact reproducibility install pinned versions:

```powershell
& ".venv\Scripts\python.exe" -m pip install -r requirements-frozen.txt
```

Running the Dash UI (predictions)

1. Ensure you have a SARIMAX artifact available at `models/sarimax_artifacts.joblib`. If you trained a model with `scripts/train.py` it will produce such an artifact.

2. Start the Dash app (from the repo root, venv activated):

```powershell
# Start app using the venv python (preferred)
& ".venv\Scripts\python.exe" app.py

# If you prefer to activate the venv first, use the PowerShell activation shown above, then:
# python app.py
```

3. Open your browser at http://127.0.0.1:8050

Using the UI

- Use the "Use dataset/test.csv" checkbox to load the built-in sample (`dataset/test.csv`) and press "Forecast".
- Or upload your own CSV with a `Date` column (and numeric columns to aggregate). Set the forecast horizon (weeks) and press "Forecast".
- If forecasting succeeds, predictions are saved to `results/sarimax_ui_predictions.csv` and the interactive plot will show historical and forecasted values.

CLI prediction (alternative)

You can also run predictions from the command line with `scripts/predict.py`:

```powershell
.venv\Scripts\python.exe scripts\predict.py --artifacts models\sarimax_artifacts.joblib --data dataset\test.csv --out results\sarimax_predictions.csv
```

Notes & troubleshooting

- The Dash app expects the artifact to contain a statsmodels SARIMAX results object stored under the `results` key (see `models/sarimax_artifacts.joblib`). If the artifact is missing or contains a different model type the UI will display an error message.
- If PowerShell refuses to run the activation script, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same session (temporary change) and then run `.venv\Scripts\Activate.ps1`.
- If port 8050 is already in use, change the port in `app.py` (the `APP.run(..., port=XXXX)` call) and restart.

See `README_DEPLOY.md` for additional deploy/run instructions and `scripts/` for CLI helpers.

MLflow (optional)

This project can optionally log training runs to MLflow. Install MLflow in your environment (using the venv python):

```powershell
& ".venv\Scripts\python.exe" -m pip install mlflow
```

Then run training with MLflow enabled (example for LightGBM):

```powershell
& ".venv\Scripts\python.exe" -m scripts.train --data-dir dataset --out models/lgbm_artifacts.joblib --model lgbm --mlflow --mlflow-experiment rossmann
```

The script will create an MLflow run, log parameters and RMSE, and upload the generated artifact (`.joblib`) to the run's artifacts.

Recommended: start MLflow UI with a sqlite backend (avoids the filesystem backend warning and persists runs across sessions):

```powershell
& ".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# then open http://127.0.0.1:5000 in your browser
```

If you prefer the default filesystem store you can still run `mlflow ui`, but you may see a FutureWarning recommending a DB-backed store.
