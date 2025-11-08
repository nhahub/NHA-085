"""Load model artifacts and run predictions on the test set (or arbitrary CSV).

Usage examples:
    python scripts/predict.py --artifacts models/lgb_artifacts.joblib --data dataset/test.csv --out predictions.csv
    python scripts/predict.py --artifacts models/lgb_artifacts.joblib --data dataset/test.csv --out results/pred.csv --data-is-test
"""
import argparse
from pathlib import Path
import joblib
import pandas as pd

from scripts.data_loader import load_raw, build_merged, get_feature_matrix


def predict(artifacts_path: str, data_path: str, out_path: str, data_is_test: bool = False):
    artifacts_path = Path(artifacts_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    artifacts = joblib.load(artifacts_path)
    model_type = artifacts.get("model_type", "lgbm")

    if model_type == "lgbm":
        model = artifacts["model"]
        scaler = artifacts.get("scaler")
        feature_columns = artifacts.get("feature_columns")
    elif model_type == "sarimax":
        results = artifacts.get("results")
        order = artifacts.get("order")
        seasonal_order = artifacts.get("seasonal_order")
        train_end = artifacts.get("train_end")
    else:
        raise RuntimeError("Unsupported model_type in artifacts")

    # If the user passes a csv, load and prepare with the same pipeline
    data_path = Path(data_path)
    if model_type == "lgbm":
        if data_path.name == "test.csv" or data_is_test:
            # load raw test + store and merge similarly to train
            # We rely on the dataset structure in dataset/ (test.csv + store.csv)
            df_train, store, df_test = load_raw(data_path.parent)
            merged = build_merged(df_train, store)
            # Use merged rows corresponding to test dates/stores where possible
            X, y, _, _ = get_feature_matrix(merged)
        else:
            df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
            # Minimal preprocessing: merge with store if available next to CSV
            store_csv = data_path.parent / "store.csv"
            if store_csv.exists():
                store = pd.read_csv(store_csv)
                merged = pd.merge(df, store, on="Store", how="left")
            else:
                merged = df
            X, y, _, _ = get_feature_matrix(merged)

    elif model_type == "sarimax":
        # For SARIMAX we trained on weekly-aggregated sales.
        # If user provides a dataset folder with test.csv, load and resample weekly to get forecast range.
        if data_path.is_file() and data_path.name == "test.csv":
            df_test = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
            numeric_test = df_test.select_dtypes(include="number")
            test_arima = numeric_test.resample("W").mean()
            # If there are no weekly rows, fallback to forecasting next N periods equal to number of test rows
            if test_arima.shape[0] == 0:
                steps = len(df_test)
                pred = results.get_forecast(steps=steps)
                pred_mean = pred.predicted_mean
                out = pd.DataFrame({"prediction": pred_mean})
                out.to_csv(out_path)
                print(f"Saved SARIMAX forecast (steps={steps}) to {out_path}")
                return
            else:
                start = test_arima.index[0]
                end = test_arima.index[-1]
                pred = results.get_prediction(start=start, end=end, dynamic=False)
                pred_mean = pred.predicted_mean
                out = pd.DataFrame({"prediction": pred_mean})
                out.to_csv(out_path)
                print(f"Saved SARIMAX weekly predictions to {out_path}")
                return
        else:
            # If user passed a different CSV, try to read and resample weekly
            df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
            numeric_test = df.select_dtypes(include="number")
            test_arima = numeric_test.resample("W").mean()
            if test_arima.shape[0] == 0:
                steps = len(df)
                pred = results.get_forecast(steps=steps)
                pred_mean = pred.predicted_mean
                out = pd.DataFrame({"prediction": pred_mean})
                out.to_csv(out_path)
                print(f"Saved SARIMAX forecast (steps={steps}) to {out_path}")
                return
            start = test_arima.index[0]
            end = test_arima.index[-1]
            pred = results.get_prediction(start=start, end=end, dynamic=False)
            pred_mean = pred.predicted_mean
            out = pd.DataFrame({"prediction": pred_mean})
            out.to_csv(out_path)
            print(f"Saved SARIMAX weekly predictions to {out_path}")
            return
    else:
        df = pd.read_csv(data_path, parse_dates=["Date"], index_col="Date")
        # Minimal preprocessing: merge with store if available next to CSV
        store_csv = data_path.parent / "store.csv"
        if store_csv.exists():
            store = pd.read_csv(store_csv)
            merged = pd.merge(df, store, on="Store", how="left")
        else:
            merged = df
        X, y, _, _ = get_feature_matrix(merged)

    # Align features to what the model was trained on
    missing = [c for c in feature_columns if c not in X.columns]
    for c in missing:
        X[c] = 0
    X = X[feature_columns]

    preds = model.predict(X)
    results = merged.copy()
    results["prediction"] = preds
    results.to_csv(out_path)
    print(f"Saved predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", required=True, help="Path to joblib artifact created by train.py")
    parser.add_argument("--data", required=True, help="CSV to run predictions on (path) or folder containing test.csv + store.csv")
    parser.add_argument("--out", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--data-is-test", action="store_true", help="Treat the data path as the test dataset folder")
    args = parser.parse_args()

    predict(args.artifacts, args.data, args.out, args.data_is_test)


if __name__ == "__main__":
    main()
