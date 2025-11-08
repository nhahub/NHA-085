# Deploy / Run instructions (local)

This project is presented as a Jupyter notebook. The supplied helper scripts let you run training and prediction from the command line.

Prerequisites
- Python 3.9+ recommended
- Install dependencies (PowerShell example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
```

Train
- Train a LightGBM model using the notebook preprocessing:

```powershell
python scripts/train.py --data-dir dataset --out models/lgb_artifacts.joblib --model lgbm
```

- Or train SARIMAX (weekly-aggregated) matching the notebook's SARIMA flow:

```powershell
python scripts/train.py --data-dir dataset --out models/sarimax_artifacts.joblib --model sarimax
```

Predict
- Run predictions (writes CSV with a `prediction` column):

```powershell
python scripts/predict.py --artifacts models/lgb_artifacts.joblib --data dataset/test.csv --out results/predictions.csv
```

Notes / contract
- The notebook uses `Date` as an index, sentinel `CompetitionDistance=200000`, `PromoInterval` expansion, IterativeImputer and `pd.get_dummies(..., prefix=categorical_cols)`. The scripts mirror these choices to remain compatible.
- If you change sentinel values or encoding, update both notebook and scripts.

Want me to also:
- create a `requirements-frozen.txt` with exact pinned versions, or
- convert notebooks into a runnable package (src/ + CLI entrypoints)?
