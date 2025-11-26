"""Helper: load a saved LGBM joblib artifact and log the sklearn model to MLflow.

Usage:
    python -m scripts.log_lgbmlflow models/lgb_artifacts_mlflow.joblib --experiment rossmann

This will create a new MLflow run and log the sklearn model under the run's `model` artifact
so it shows up as a model in the MLflow UI. It also uploads scaler and feature_columns artifacts.
"""
import argparse
import joblib
import pandas as pd
import subprocess
from pathlib import Path
import json
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="Path to the joblib artifact containing the LGBM model")
    parser.add_argument("--val-data", help="Optional CSV file with validation features and true target to compute metrics (Sales column)")
    parser.add_argument("--experiment", default="rossmann", help="MLflow experiment name to use")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        raise SystemExit(f"Artifact not found: {artifact_path}")

    artifacts = joblib.load(artifact_path)
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")
    feature_columns = artifacts.get("feature_columns")

    if model is None:
        raise SystemExit("No 'model' object found in the provided artifact")

    try:
        import mlflow
        import mlflow.sklearn
        import mlflow.pyfunc
        from mlflow.models.signature import infer_signature
    except Exception as e:
        raise SystemExit("MLflow is required to run this helper. Install with: pip install mlflow")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_param("source_artifact", str(artifact_path.name))
        # Log the sklearn model so it appears in the MLflow UI as a proper model artifact
        try:
            # Attempt to log the sklearn flavor and also calculate a model signature
            X_example = None
            if feature_columns:
                # make a tiny input example of zeros using feature_columns
                try:
                    X_example = pd.DataFrame([{c: 0 for c in feature_columns} for _ in range(3)])
                except Exception:
                    X_example = None

            signature = None
            try:
                if X_example is not None:
                    preds_example = model.predict(X_example)
                    signature = infer_signature(X_example, preds_example)
            except Exception:
                signature = None

            mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_example, signature=signature)
            print("Logged sklearn model to MLflow under artifact 'model'")
        except Exception as e:
            print("mlflow.sklearn.log_model failed:", e)

        # If a validation dataset is provided, try to compute predictions and RMSE
        if args.val_data:
            val_path = Path(args.val_data)
            if val_path.exists():
                try:
                    df_val = pd.read_csv(val_path, parse_dates=["Date"], index_col="Date", low_memory=False)
                except Exception:
                    try:
                        df_val = pd.read_csv(val_path, low_memory=False)
                    except Exception:
                        df_val = None

                if df_val is not None:
                    # Attempt to align validation data with feature_columns and predict
                    try:
                        X_val = None
                        if feature_columns:
                            X_val = df_val.reindex(columns=feature_columns, fill_value=0)
                        else:
                            X_val = df_val.select_dtypes(include=["number"]).fillna(0)

                        preds = None
                        try:
                            preds = model.predict(X_val)
                        except Exception:
                            # try using booster if present
                            lg = getattr(model, "booster_", model)
                            preds = lg.predict(X_val)

                        # If Sales present in val, compute rmse
                        if "Sales" in df_val.columns:
                            from sklearn.metrics import mean_squared_error
                            import math
                            y_true = df_val["Sales"].astype(float)
                            # attempt to align lengths
                            min_len = min(len(y_true), len(preds))
                            rmse = math.sqrt(mean_squared_error(y_true.iloc[:min_len], preds[:min_len]))
                            mlflow.log_metric("rmse", float(rmse))
                            print(f"Logged RMSE from validation file: {rmse:.4f}")
                        # Save predictions for traceability
                        pred_df = pd.DataFrame({"prediction": preds}, index=X_val.index if hasattr(X_val, "index") else None)
                        tmp_pred = Path(tempfile.mkstemp(suffix="_preds.csv")[1])
                        pred_df.to_csv(tmp_pred)
                        mlflow.log_artifact(str(tmp_pred))
                        try:
                            tmp_pred.unlink()
                        except Exception:
                            pass
                    except Exception as e:
                        print("Failed to compute predictions on validation data:", e)

        # upload scaler and feature columns as artifacts for reproducibility
        temp_files = []
        try:
            if scaler is not None:
                tmp_scaler = Path(tempfile.mkstemp(suffix="_scaler.joblib")[1])
                joblib.dump(scaler, tmp_scaler)
                mlflow.log_artifact(str(tmp_scaler))
                temp_files.append(tmp_scaler)
            if feature_columns is not None:
                tmp_feats = Path(tempfile.mkstemp(suffix="_feature_columns.json")[1])
                with open(tmp_feats, "w", encoding="utf-8") as f:
                    json.dump(list(feature_columns), f)
                mlflow.log_artifact(str(tmp_feats))
                temp_files.append(tmp_feats)
        finally:
            for tf in temp_files:
                try:
                    tf.unlink()
                except Exception:
                    pass

        # Log environment (pip freeze) for reproducibility
        try:
            tmp_reqs = Path(tempfile.mkstemp(suffix="_requirements.txt")[1])
            try:
                out = subprocess.check_output(["pip", "freeze"], universal_newlines=True)
                tmp_reqs.write_text(out, encoding="utf-8")
                mlflow.log_artifact(str(tmp_reqs))
            finally:
                try:
                    tmp_reqs.unlink()
                except Exception:
                    pass
        except Exception:
            # Don't fail the run if pip freeze not available
            pass

        # Also try to register a pyfunc wrapper for consistent inference if wrapper exists
        try:
            from scripts.lgbm_pyfunc import LgbmPyfuncModel
            try:
                mlflow.pyfunc.log_model(
                    artifact_path="model_pyfunc",
                    python_model=LgbmPyfuncModel(),
                    artifacts={"lgb_artifacts": str(artifact_path)},
                    signature=signature,
                    input_example=X_example,
                )
                print("Logged pyfunc model as 'model_pyfunc'")
            except Exception as e:
                print("mlflow.pyfunc.log_model failed:", e)
        except Exception:
            # wrapper missing — not a hard failure
            pass

        print(f"MLflow run created: {run.info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
