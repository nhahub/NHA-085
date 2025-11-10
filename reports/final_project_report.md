# Final Project Report — Sales Forecasting and Demand Prediction

## Summary

This project predicts future sales and demand using historical store-level data (Rossmann dataset). The work includes exploratory data analysis (EDA), preprocessing, model development (SARIMAX and tree-based models), evaluation, an interactive Dash UI for forecasts, and MLOps scaffolding including optional MLflow integration and monitoring documentation.

## Data & Preprocessing

Source files: dataset/train.csv, dataset/store.csv, dataset/test.csv.
Date is parsed as the index and time-based features (year, month, day, week) are derived. Missing values are handled with explicit sentinels (e.g., CompetitionDistance=200000) and imputation for selected fields. Categorical variables were one-hot encoded where appropriate and numeric features scaled when used in tree-based pipelines.

## Exploratory Data Analysis (EDA)

The Jupyter notebook `Rossman_Sales_Forecasting with ML.ipynb` contains EDA including seasonal plots, holiday/promotion effects, and correlation analysis. Visualizations highlight weekly and monthly seasonality and promotional impacts on sales.

## Models & Evaluation

Implemented models: SARIMAX (weekly aggregated), LightGBM and XGBoost pipelines. Training and evaluation code live in `scripts/train.py` and the notebook. Evaluation metrics used: MAE, MSE, RMSE. Model artifacts are saved under `models/` (for example `models/sarimax_artifacts.joblib`).

## Deployment & MLOps

An interactive Dash UI (`app.py`) enables uploading test data and generating forecasts using the SARIMAX artifact. MLflow support was added to `scripts/train.py` to optionally log parameters, metrics, and artifacts. Monitoring guidance and instrumentation snippets are in `docs/monitoring_setup.md`. A lightweight report generator and utilities exist under `scripts/`.

## Artifacts

Repo artifacts include: models/sarimax_artifacts.joblib, mlruns/ (MLflow artifacts), results/sarimax_ui_predictions.csv (UI output). Ensure trained LightGBM/XGBoost artifacts are saved to `models/` after training.

## Recommendations & Next Steps

1) Export a polished Final Presentation (.pptx).
2) Save cleaned dataset artifacts to dataset/ (cleaned_train.csv).
3) Containerize the Dash app (add Dockerfile) for easy deployment.
4) Wrap SARIMAX with an MLflow pyfunc flavor for uniform serving.
5) Add small smoke tests to `tests/` to verify prediction scripts programmatically.

## References (technical terms)

ARIMA/SARIMAX: https://www.statsmodels.org/stable/tsa.html
ETS: https://otexts.com/fpp3/ets.html
LSTM: https://keras.io/guides/what_is_keras/
LightGBM: https://lightgbm.readthedocs.io/
XGBoost: https://xgboost.readthedocs.io/
MLflow: https://mlflow.org/
DVC: https://dvc.org/
Dash/Plotly: https://dash.plotly.com/
Flask: https://flask.palletsprojects.com/
FastAPI: https://fastapi.tiangolo.com/
Prometheus: https://prometheus.io/
Grafana: https://grafana.com/
Loki: https://grafana.com/oss/loki/
Alertmanager: https://prometheus.io/docs/alerting/latest/alertmanager/
joblib: https://joblib.readthedocs.io/
pandas/numpy/scikit-learn: https://pandas.pydata.org/ https://numpy.org/ https://scikit-learn.org/
ReportLab: https://www.reportlab.com/


