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

This project supports MLflow for experiment tracking and model artifact storage. The code does two things:

- When you train with `scripts.train --mlflow`, the training run will log params, validation RMSE and (where supported) training traces. For LightGBM we enable MLflow autologging so you also get iteration-level metrics and lineage.
- There are helper scripts to retroactively log an existing joblib artifact as an MLflow model (`scripts/log_lgbmlflow.py`) and to inspect the artifacts in a given experiment (`scripts/list_mlflow_artifacts.py`).

Install MLflow into your virtual environment:

```powershell
& ".venv\Scripts\python.exe" -m pip install mlflow
```

Recommended local server (persist runs, avoid future filesystem-only warnings)
1) Start MLflow **server** or **UI** backed by sqlite (recommended for local experiments):

```powershell
# Run the MLflow server (recommended).
# Replace <REPO_ROOT> with the absolute path to this repository if you want a specific artifact root.
& ".venv\Scripts\python.exe" -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root file:///%CD%/mlruns --host 127.0.0.1 --port 5000

# OR (simpler) run the built-in UI which defaults to filesystem tracking:
& ".venv\Scripts\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Then open http://127.0.0.1:5000 in your browser
```

Important: if you start a dedicated MLflow server process (the `mlflow server` command above), set the tracking URI for training processes so autologging and artifact uploads write back to the running server (PowerShell example):

```powershell
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
```

Training with MLflow enabled
---------------------------
Use `scripts.train` with `--mlflow` to run a training job that logs metrics + artifacts. Choose an experiment name via `--mlflow-experiment`.

LightGBM (recommended path — autolog + pyfunc):

```powershell
& ".venv\Scripts\python.exe" -m scripts.train --data-dir dataset --out models/lgb_artifacts.joblib --model lgbm --mlflow --mlflow-experiment rossmann
```

SARIMAX (weekly-aggregated; will attempt to log a pyfunc wrapper):

```powershell
& ".venv\Scripts\python.exe" -m scripts.train --data-dir dataset --out models/sarimax_artifacts.joblib --model sarimax --mlflow --mlflow-experiment rossmann
```

Notes:
- Training with LGBM logs hyperparameters, RMSE and (when MLflow is available) LightGBM autologging captures iteration-level metrics and lineage.
- For LGBM the training logic will attempt to log both a native model flavor (sklearn/lightgbm) and a `pyfunc` wrapper so the model can be loaded consistently via `mlflow.pyfunc.load_model()`.
- For SARIMAX the training flow logs a joblib artifact and attempts to register a pyfunc wrapper (`scripts.sarimax_pyfunc.SarimaxPyfuncModel`) under the run as `model_pyfunc`.

Retroactive logging / helper scripts
-----------------------------------
If you already have a joblib artifact (e.g., `models/lgb_artifacts.joblib`) and want to create a proper MLflow run with a model artifact and AM-like metadata, use the helper:

```powershell
# Log an existing LGBM joblib artifact to MLflow and attempt to add a pyfunc model
& ".venv\Scripts\python.exe" -m scripts.log_lgbmlflow models/lgb_artifacts.joblib --experiment rossmann

# If you have a validation file (CSV) containing a `Sales` column, log and compute RMSE too:
& ".venv\Scripts\python.exe" -m scripts.log_lgbmlflow models/lgb_artifacts.joblib --experiment rossmann --val-data dataset/test.csv
```

Inspect artifacts for an experiment (quick):

```powershell
& ".venv\Scripts\python.exe" -m scripts.list_mlflow_artifacts
# By default the script scans the "rossmann" experiment. Edit the script or call it from Python if you want different experiment names.
```

Why we log pyfunc models
------------------------
Pyfunc wrappers (via `mlflow.pyfunc.log_model`) capture prediction-time behavior in a single object (preprocessing + model). This helps avoid the inference mismatches that can happen when model artifacts and preprocessing are stored separately (scaler mismatch, feature-order issues). Both `scripts.train` and `scripts/log_lgbmlflow.py` attempt to log a pyfunc `model_pyfunc` artifact when possible.

Troubleshooting and tips
------------------------
- If autologging emits warnings about validating the input example, it means the sample use-case (input_example) does not match the model's expected feature vector; ensure the `input_example` you provide matches the model's `feature_columns` order and types.
- If you see errors while autologging that mention artifact store/tracking URI differences, set `MLFLOW_TRACKING_URI` to the running server's URL (see above) so artifacts are uploaded correctly to the server.
- For production use, consider using a remote artifact store (S3/Azure/GCS) and a database-backed tracking store for MLflow.
