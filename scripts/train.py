"""CLI training script for Rossmann forecasting (LightGBM or SARIMAX)

Usage:
    python -m scripts.train --data-dir dataset --out models/sarimax_artifacts.joblib --model sarimax

This script uses the preprocessing in `scripts/data_loader.py`. It trains either a
LightGBM (sklearn API) or a SARIMAX model (statsmodels) and persists artifacts with joblib.
"""

import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path so `python scripts/train.py` (script mode)
# can import sibling `scripts.*` modules. When executing a file directly,
# Python sets sys.path[0] to the `scripts/` directory which prevents
# `import scripts.data_loader` from resolving. Insert repo root at front.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import joblib
import json
import tempfile
import os

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

import statsmodels.api as sm

from scripts.data_loader import load_raw, build_merged, get_feature_matrix


def _ensure_mlflow():
    try:
        import mlflow  # noqa: F401
    except Exception as e:
        raise RuntimeError("MLflow requested but not installed. Install with: pip install mlflow")


def train_and_save(data_dir: str, out_path: str, test_size: float = 0.2, random_state: int = 42,
                   model_type: str = "lgbm", use_mlflow: bool = False, mlflow_experiment: str = "rossmann"):
    data_dir = Path(data_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train, store, _test = load_raw(data_dir)

    def _run_mlflow_run(run_fn, params: dict, metrics: dict, artifacts: list):
        """Helper to run an mlflow run and log params/metrics/artifacts."""
        import mlflow
        # Try to set the requested experiment. If the experiment has been deleted or
        # is otherwise unavailable, try to restore or create a fresh experiment so
        # the run does not fail due to a missing experiment resource.
        try:
            mlflow.set_experiment(mlflow_experiment)
        except Exception as ex_set_exp:
            try:
                # Attempt to restore deleted experiment if possible, or create a new one
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                exp = client.get_experiment_by_name(mlflow_experiment)
                if exp is not None and getattr(exp, 'lifecycle_stage', None) == 'deleted' and hasattr(client, 'restore_experiment'):
                    try:
                        client.restore_experiment(exp.experiment_id)
                        mlflow.set_experiment(mlflow_experiment)
                    except Exception:
                        # If restore fails, fall back to creating a new experiment name
                        fallback_name = f"{mlflow_experiment}_auto"
                        try:
                            client.create_experiment(fallback_name)
                            mlflow.set_experiment(fallback_name)
                        except Exception:
                            mlflow.set_experiment("default")
                else:
                    # Experiment doesn't exist or can't be restored. Create an experiment
                    try:
                        client.create_experiment(mlflow_experiment)
                        mlflow.set_experiment(mlflow_experiment)
                    except Exception:
                        # Last resort: use default experiment
                        mlflow.set_experiment("default")
            except Exception:
                # If anything goes wrong while handling experiments, fall back to default
                try:
                    mlflow.set_experiment("default")
                except Exception:
                    # If even default fails (very unusual), re-raise the original exception
                    raise ex_set_exp
        with mlflow.start_run():
            for k, v in params.items():
                mlflow.log_param(k, v)
            # execute training/eval logic provided by caller; if run_fn raises we'll
            # let it propagate so the script logs the error and the caller can inspect
            # stdout/stderr — we still ensured the MLflow experiment is available
            run_fn()
            for k, v in metrics.items():
                try:
                    mlflow.log_metric(k, float(v))
                except Exception:
                    pass
            for a in artifacts:
                try:
                    mlflow.log_artifact(str(a))
                except Exception:
                    pass

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

        # Only enable MLflow autologging for LightGBM if MLflow logging was requested
        # (autolog creates run context during fit; enabling it unconditionally can create
        # accidental runs which may end up marked as FAILED if an unrelated error occurs).
        if use_mlflow:
            try:
                import mlflow.lightgbm as _ml_lgb
                _ml_lgb.autolog()
            except Exception:
                # If MLflow is not available or autolog not supported, fall back silently.
                pass

        # Fit model robustly across different LightGBM versions.
        fit_succeeded = False
        fit_errors = []
        # Try common fit signatures in order of preference
        fit_attempts = [
            lambda: model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=100),
            lambda: model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]),
            lambda: model.fit(X_train, y_train),
        ]

        for attempt in fit_attempts:
            try:
                attempt()
                fit_succeeded = True
                break
            except TypeError as te:
                fit_errors.append(str(te))
                continue
            except Exception as e:
                # For non-TypeErrors, record and continue to try fallbacks
                fit_errors.append(str(e))
                continue

        if not fit_succeeded:
            raise RuntimeError(f"All LGBM fit attempts failed. Errors: {fit_errors}")

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

        # Optionally log to MLflow
        if use_mlflow:
            _ensure_mlflow()
            import mlflow
            import mlflow.sklearn

            # Prepare extra artifacts: scaler and feature_columns
            temp_files = []
            try:
                if scaler is not None:
                    scaler_path = Path(tempfile.mkstemp(suffix="_scaler.joblib")[1])
                    joblib.dump(scaler, scaler_path)
                    temp_files.append(scaler_path)
                # write feature columns as json
                if feature_columns is not None:
                    feat_path = Path(tempfile.mkstemp(suffix="_feature_columns.json")[1])
                    with open(feat_path, "w", encoding="utf-8") as f:
                        json.dump(list(feature_columns), f)
                    temp_files.append(feat_path)

                def _train_fn():
                    # Attempt to log the trained model into the MLflow run.
                    # Try mlflow.sklearn.log_model first (works for sklearn-like objects),
                    # then try mlflow.lightgbm.log_model for LightGBM native logging,
                    # and finally fall back to uploading the joblib artifact.
                    input_example = None
                    try:
                        # use a small sample of validation features as an input example
                        input_example = X_val.iloc[:3]
                    except Exception:
                        input_example = None

                    # Log sklearn flavor with input example/signature where possible
                    from mlflow.models.signature import infer_signature
                    try:
                        signature = None
                        try:
                            signature = infer_signature(X_val.iloc[:5], model.predict(X_val.iloc[:5]))
                        except Exception:
                            signature = None

                        mlflow.sklearn.log_model(model, "model", input_example=input_example, signature=signature)
                        print("mlflow.sklearn.log_model succeeded")
                    except Exception as e1:
                        print("mlflow.sklearn.log_model failed:", e1)
                        try:
                            # import via importlib to avoid rebinding the name `mlflow` in this local scope
                            import importlib
                            mlflow_lgb = importlib.import_module("mlflow.lightgbm")
                            lg = getattr(model, "booster_", model)
                            mlflow_lgb.log_model(lg, artifact_path="model")
                            print("mlflow.lightgbm.log_model succeeded")
                        except Exception as e2:
                            print("mlflow.lightgbm.log_model failed:", e2)
                            try:
                                # fallback: upload the joblib artifact created by the script
                                mlflow.log_artifact(str(out_path))
                                print("Logged joblib artifact as fallback")
                            except Exception as e3:
                                print("Fallback mlflow.log_artifact failed:", e3)

                    # Attempt to also log a pyfunc that wraps the saved joblib artifact for consistent inference
                    try:
                        from scripts.lgbm_pyfunc import LgbmPyfuncModel
                        # Infer input signature using X_val where possible
                        try:
                            signature = infer_signature(X_val.iloc[:5], model.predict(X_val.iloc[:5]))
                        except Exception:
                            signature = None

                        mlflow.pyfunc.log_model(
                            artifact_path="model_pyfunc",
                            python_model=LgbmPyfuncModel(),
                            artifacts={"lgb_artifacts": str(out_path)},
                            signature=signature,
                            input_example=input_example,
                        )
                        print("Logged LGBM as mlflow.pyfunc model (model_pyfunc)")
                    except Exception as e:
                        print("mlflow.pyfunc.log_model for LGBM failed:", e)

                params = {
                    "model_type": "lgbm",
                    "num_leaves": getattr(model, "num_leaves", None),
                    "learning_rate": getattr(model, "learning_rate", None),
                    "n_estimators": getattr(model, "n_estimators", None),
                    "random_state": random_state,
                    "test_size": test_size,
                }
                metrics = {"rmse": rmse}

                # artifacts to upload to mlflow: main joblib + any temp files created
                mlflow_artifacts = [out_path] + temp_files
                _run_mlflow_run(_train_fn, params, metrics, mlflow_artifacts)
            finally:
                # cleanup temp files
                for tf in temp_files:
                    try:
                        os.unlink(tf)
                    except Exception:
                        pass

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
        # Enable MLflow autologging for statsmodels SARIMAX if MLflow logging was requested
        # (not all MLflow installs include a statsmodels autolog impl; this is best-effort)
        if use_mlflow:
            try:
                import mlflow.statsmodels as _ml_stats
                _ml_stats.autolog()
            except Exception:
                # If statsmodels autolog isn't available, continue without error
                pass
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

        # Optionally log to MLflow
        if use_mlflow:
            _ensure_mlflow()
            import mlflow
            import mlflow.pyfunc

            # Try to log a pyfunc-wrapped SARIMAX model for uniform MLflow serving.
            # This requires the `scripts.sarimax_pyfunc.SarimaxPyfuncModel` class to be importable.
            try:
                from scripts.sarimax_pyfunc import SarimaxPyfuncModel

                def _train_fn():
                    # Attempt to log a pyfunc model for SARIMAX with input_example and
                    # an inferred signature where possible (mirrors the LGBM logging flow).
                    try:
                        input_example = None
                        signature = None
                        # Use a small sample from val_series if present to build an input example
                        try:
                            if val_series is not None and len(val_series) > 0:
                                # For pyfunc SARIMAX the model expects a DataFrame placeholder
                                # with a DatetimeIndex matching number of steps; create a small
                                # DataFrame with 3 rows (or fewer) carrying the same index type
                                sample_len = min(3, len(val_series))
                                input_example = val_series.iloc[:sample_len].to_frame()
                                # infer signature by using the forecast from the fitted results
                                try:
                                    sig_y = results.get_forecast(steps=len(input_example)).predicted_mean
                                    from mlflow.models.signature import infer_signature
                                    signature = infer_signature(input_example, sig_y)
                                except Exception:
                                    signature = None
                        except Exception:
                            input_example = None

                        mlflow.pyfunc.log_model(
                            artifact_path="model_pyfunc",
                            python_model=SarimaxPyfuncModel(),
                            artifacts={"sarimax_artifacts": str(out_path)},
                            signature=signature,
                            input_example=input_example,
                        )
                    except Exception as e:
                        print("mlflow.pyfunc.log_model failed:", e)
                        # fallback to logging the raw joblib artifact
                        try:
                            mlflow.log_artifact(str(out_path))
                        except Exception as e2:
                            print("mlflow.log_artifact failed:", e2)

            except Exception as ex:
                print("Failed to import SARIMAX pyfunc wrapper; falling back to artifact upload:", ex)

                def _train_fn():
                    try:
                        mlflow.log_artifact(str(out_path))
                    except Exception as e:
                        print("mlflow.log_artifact failed:", e)

            params = {
                "model_type": "sarimax",
                "order": str(order),
                "seasonal_order": str(seasonal_order),
                "random_state": random_state,
                "test_size": test_size,
            }
            metrics = {"rmse": rmse}
            _run_mlflow_run(_train_fn, params, metrics, [out_path])

    else:
        raise ValueError("Unsupported model_type. Choose 'lgbm' or 'sarimax'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset directory")
    parser.add_argument("--out", default="models/lgb_artifacts.joblib", help="Output artifact path")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model", default="lgbm", choices=["lgbm", "sarimax"], help="Model type to train")
    parser.add_argument("--mlflow", action="store_true", help="Enable MLflow logging for this run")
    parser.add_argument("--mlflow-experiment", default="rossmann", help="MLflow experiment name to use when logging")
    args = parser.parse_args()

    train_and_save(
        args.data_dir,
        args.out,
        args.test_size,
        args.random_state,
        model_type=args.model,
        use_mlflow=args.mlflow,
        mlflow_experiment=args.mlflow_experiment,
    )


if __name__ == "__main__":
    main()
