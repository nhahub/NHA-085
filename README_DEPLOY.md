# Deploy / Run instructions (local)

This project is presented as a Jupyter notebook. The supplied helper scripts let you run training and prediction from the command line.

Prerequisites
- Python 3.9+ recommended (this repository was developed and tested on Python 3.11 — 3.9+ should work but 3.11 is recommended)

Install dependencies (PowerShell example; explicit venv python usage is recommended):

```powershell
# From repo root
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .venv\Scripts\Activate.ps1
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Train
- Train a LightGBM model using the notebook preprocessing:

```powershell
& ".venv\Scripts\python.exe" -m scripts.train --data-dir dataset --out models/lgb_artifacts.joblib --model lgbm
```

- Or train SARIMAX (weekly-aggregated) matching the notebook's SARIMA flow:

```powershell
& ".venv\Scripts\python.exe" -m scripts.train --data-dir dataset --out models/sarimax_artifacts.joblib --model sarimax
```

Predict
- Run predictions (writes CSV with a `prediction` column):

```powershell
& ".venv\Scripts\python.exe" -m scripts.predict --artifacts models/lgb_artifacts.joblib --data dataset/test.csv --out results/predictions.csv
```

Notes / contract
-- The notebook uses `Date` as an index, sentinel `CompetitionDistance=200000`, `PromoInterval` expansion, IterativeImputer and `pd.get_dummies(..., prefix=categorical_cols)`. The scripts mirror these choices to remain compatible.
-- If you change sentinel values or encoding, update both notebook and scripts.

Notes
- Prefer invoking modules with `-m` (as shown) so the package import shim in `scripts/__init__.py` works correctly when running from the repo root.
- If you want MLflow experiment tracking, see `README.md` for a recommended `mlflow ui` invocation that uses sqlite to persist runs.