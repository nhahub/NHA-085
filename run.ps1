# Simple Windows PowerShell helper to create venv, install requirements and run training
param(
    [string]$Action = "help"
)

switch ($Action) {
    "setup" {
        python -m venv .venv
        .\.venv\Scripts\Activate
        pip install -r requirements.txt
        break
    }
    "train" {
        .\.venv\Scripts\Activate
        python scripts/train.py --data-dir dataset --out models/lgb_artifacts.joblib
        break
    }
    "predict" {
        .\.venv\Scripts\Activate
        python scripts/predict.py --artifacts models/lgb_artifacts.joblib --data dataset/test.csv --out results/predictions.csv
        break
    }
    default {
        Write-Host "Usage: .\run.ps1 setup|train|predict"
    }
}
