## Copilot instructions for Rossman Sales Forecasting (notebook project)

This repository is a Jupyter/VS Code notebook-based forecasting project using the Rossmann dataset. The goal of this file is to provide concise, actionable guidance so an AI coding agent (or contributor) can be productive immediately.

Key facts (quick):
- Main analysis and pipeline: `Rossman_Sales_Forecasting with ML.ipynb` (root). Run this notebook to reproduce EDA, preprocessing, baseline SARIMAX and tree-based models.
- Data files expected under `dataset/`:
  - `dataset/train.csv` (Date parsed as index in notebook)
  - `dataset/store.csv`
  - `dataset/test.csv`

Project-specific patterns you must follow or preserve:
- Date handling: the notebook reads train with `parse_dates=['Date'], index_col='Date'` and immediately derives Year/Month/Day/WeekofYear. Keep this indexing approach when adding time-series code.
- Missing-value strategy:
  - `train.fillna(0, inplace=True)` is used for the `train` dataframe early in the notebook (explicit sentinel fill).
  - `CompetitionDistance` missing values are filled with `200000` as a sentinel; downstream code assumes sentinel means "no competition" and sets related fields to 0.
  - `PromoInterval` missing values are filled with `'0'` and then expanded into month columns `Promo_Jan`...`Promo_Dec`.
  - IterativeImputer from sklearn.experimental is used for `CompetitionOpenSinceMonth/Year` and `Promo2SinceWeek/Year`.
- Categorical encoding: `pd.get_dummies(store_merged, columns=categorical_cols, prefix=categorical_cols)` where `categorical_cols = ['StoreType','Assortment','StateHoliday']` — replicate this exact pattern when adding features or training code for consistency with downstream column naming.
- Scaling: MinMaxScaler is applied to the numeric column list `numeric_cols` (see the notebook for the exact list). When adding numeric features, include them in `numeric_cols` before scaling.
- Feature selection for models: the notebook creates X by dropping `['Store','Sales','Customers','SalesPerCustomer']`. Be careful if you change these names.

Models & evaluation notes (what the notebook already does):
- SARIMAX: trained on weekly-aggregated numeric `train_arima` (resampled `'W'`) using statsmodels' `statespace.SARIMAX`. Notebook searches pdq/seasonal_pdq then fits `(1,1,1)x(1,1,1,12)` as example.
- XGBoost: uses `xgb.DMatrix` and `xgb.train` with params `{'max_depth':6, 'eta':0.3, 'objective':'reg:linear'}` and early stopping; predictions computed from DMatrix (note: newer xgboost versions deprecate `reg:linear` — keep this line as-is if mirroring notebook behavior).
- LightGBM: uses `lgb.train` with an `lgb.Dataset`, early stopping (100 rounds) and `log_evaluation(period=100)` callbacks.

Repro steps (how to run locally):
1. Open the notebook `Rossman_Sales_Forecasting with ML.ipynb` in Jupyter or VS Code Notebook.
2. Ensure required packages are installed. The notebook contains inline installs for some libs (e.g., `!pip install dash`, `!pip install lightgbm`). Recommended env install (run in your Python environment):

   pip install pandas numpy matplotlib seaborn plotly dash scikit-learn statsmodels xgboost lightgbm

3. Run the top cells to load data. If `dataset/` is not present, point notebook to the correct data location or add dataset files.

Debugging & behavior to preserve:
- If you change the index strategy (no longer using Date as index), update every cell that uses `.resample('W')`, `.index.year`, `.index.month`, or `.index.isocalendar().week`.
- The sentinel `200000` for CompetitionDistance is relied on later to set related features to 0 — do not replace it silently. If you change the sentinel, update all referencing cells.
- The notebook uses `pd.get_dummies(..., prefix=categorical_cols)` which creates column names like `StoreType_a`. When adding checks or new models, reference these prefixed column names.

Files to inspect for examples:
- `Rossman_Sales_Forecasting with ML.ipynb` — entire pipeline (EDA → preprocessing → SARIMAX → XGBoost → LightGBM). See the cells that define `numeric_cols` and `categorical_cols` and the sections labeled "preprocessing", "SARIMAX", "XGBoost", and "LightGBM".

When making PRs or code changes:
- Keep changes minimal to the notebook's data handling contract: date-as-index, sentinel values, and the get_dummies prefixing.
- Add a short note in the notebook (markdown) close to any change that alters a sentinel, scaling, or encoding choice.

If anything is unclear or you want the agent to expand this into a `requirements.txt` or helper scripts (data loader, train.py), tell me which artifact to create next and I will draft it.
