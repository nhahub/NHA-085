"""CLI training script for Rossmann forecasting (LightGBM or SARIMAX)

Usage:
    python -m scripts.train --data-dir dataset --out models/sarimax_artifacts.joblib --model sarimax

This script uses the preprocessing in `scripts/data_loader.py`. It trains either a
LightGBM (sklearn API) or a SARIMAX model (statsmodels) and persists artifacts with joblib.
"""

import argparse
from pathlib import Path
import joblib

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.api as sm

from scripts.data_loader import load_raw, build_merged, get_feature_matrix


def train_and_save(data_dir: str, out_path: str, test_size: float = 0.2, random_state: int = 42, model_type: str = "lgbm"):
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train, store, _test = load_raw(data_dir)

    if model_type.lower() == "lgbm":
        merged = build_merged(train, store)
        X, y, scaler, feature_columns = get_feature_matrix(merged)

        # If y is None (e.g., when training on test-only), raise
        if y is None:
            raise RuntimeError("No Sales column found for training target")

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        model = lgb.LGBMRegressor(
            objective="regression",
            metric="rmse",
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=1000,
            random_state=random_state,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=100,
            verbose=100,
        )

        preds = model.predict(X_val, num_iteration=getattr(model, "best_iteration_", None))
        rmse = sqrt(mean_squared_error(y_val, preds))
        print(f"Validation RMSE: {rmse:.4f}")

        # Save artifacts: model, scaler, feature_columns
        artifacts = {
            "model_type": "lgbm",
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns,
        }
        joblib.dump(artifacts, out_path)
        print(f"Saved LGBM model artifacts to {out_path}")

    elif model_type.lower() == "sarimax":
        # SARIMAX is trained on weekly-aggregated sales (matching the notebook)
        numeric_train = train.select_dtypes(include="number")
        train_arima = numeric_train.resample("W").mean()
        if "Sales" not in train_arima.columns:
            raise RuntimeError("Sales column not found for SARIMAX training")

        series = train_arima["Sales"].dropna()
        if len(series) < 10:
            raise RuntimeError("Not enough data to train SARIMAX")

        split_idx = int(len(series) * (1 - test_size))
        train_series = series.iloc[:split_idx]
        val_series = series.iloc[split_idx:]

        # Default orders from notebook; these can be parameterized further
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

        print("Fitting SARIMAX on weekly-aggregated sales...")
        mod = sm.tsa.statespace.SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                                        enforce_stationarity=False, enforce_invertibility=False)
        results = mod.fit(disp=False)

        # Evaluate on validation period
        pred = results.get_prediction(start=val_series.index[0], end=val_series.index[-1], dynamic=False)
        pred_mean = pred.predicted_mean
        rmse = sqrt(mean_squared_error(val_series, pred_mean))
        print(f"SARIMAX validation RMSE: {rmse:.4f}")

        artifacts = {
            "model_type": "sarimax",
            "results": results,
            "order": order,
            "seasonal_order": seasonal_order,
            "train_end": train_series.index[-1],
        }
        joblib.dump(artifacts, out_path)
        print(f"Saved SARIMAX artifacts to {out_path}")

    else:
        raise ValueError("Unsupported model_type. Choose 'lgbm' or 'sarimax'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--out", default="models/lgb_artifacts.joblib", help="Output artifact path")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "sarimax"], help="Model type to train")
    args = parser.parse_args()

    train_and_save(args.data_dir, args.out, args.test_size, args.random_state, model_type=args.model)


if __name__ == "__main__":
    main()
