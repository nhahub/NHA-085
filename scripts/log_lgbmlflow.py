"""Helper: load a saved LGBM joblib artifact and log the sklearn model to MLflow.

Usage:
    python -m scripts.log_lgbmlflow models/lgb_artifacts_mlflow.joblib --experiment rossmann

This will create a new MLflow run and log the sklearn model under the run's `model` artifact
so it shows up as a model in the MLflow UI. It also uploads scaler and feature_columns artifacts.
"""
import argparse
import joblib
from pathlib import Path
import json
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="Path to the joblib artifact containing the LGBM model")
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
    except Exception as e:
        raise SystemExit("MLflow is required to run this helper. Install with: pip install mlflow")

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_param("source_artifact", str(artifact_path.name))
        # Log the sklearn model so it appears in the MLflow UI as a proper model artifact
        try:
            mlflow.sklearn.log_model(model, artifact_path="model")
            print("Logged sklearn model to MLflow under artifact 'model'")
        except Exception as e:
            print("mlflow.sklearn.log_model failed:", e)

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

        print(f"MLflow run created: {run.info.run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")


if __name__ == "__main__":
    main()
