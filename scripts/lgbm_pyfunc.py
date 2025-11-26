import joblib
import pandas as pd
import numpy as np
import mlflow.pyfunc


class LgbmPyfuncModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper for LGBM artifacts saved as a joblib dict.

    Expected artifact layout (joblib dict):
      - 'model': trained lgb model or wrapper
      - 'scaler': optional fitted scaler used at training time
      - 'feature_columns': list of column names used as features during training

    The wrapper aligns incoming DataFrame columns with `feature_columns`, fills
    missing columns with zeros, applies scaler if present, and calls model.predict.
    """

    def load_context(self, context):
        # context.artifacts mapping is provided by mlflow when logging the pyfunc.
        art_path = None
        # prefer explicit key
        if "lgb_artifacts" in getattr(context, "artifacts", {}):
            art_path = context.artifacts["lgb_artifacts"]
        else:
            for k in getattr(context, "artifacts", {}).keys():
                if k.lower().startswith("lgb") or k.lower().endswith("artifacts"):
                    art_path = context.artifacts[k]
                    break

        if art_path is None:
            raise RuntimeError("lgb artifact not found in context.artifacts; expected key 'lgb_artifacts'")

        self._artifacts = joblib.load(art_path)
        self._model = self._artifacts.get("model")
        self._scaler = self._artifacts.get("scaler")
        self._feature_columns = list(self._artifacts.get("feature_columns") or [])

        if self._model is None:
            raise RuntimeError("Loaded LGB artifact does not contain 'model'")

    def _align_input(self, df: pd.DataFrame) -> pd.DataFrame:
        # Ensure a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Create a copy with feature columns in order; missing -> fill 0; extra -> drop
        if self._feature_columns:
            # Add any missing columns with zeros
            for col in self._feature_columns:
                if col not in df.columns:
                    df[col] = 0

            aligned = df.reindex(columns=self._feature_columns, fill_value=0)
        else:
            # No saved feature columns; keep all numeric columns
            aligned = df.select_dtypes(include=[np.number]).fillna(0)

        return aligned

    def _apply_scaler(self, X: pd.DataFrame) -> np.ndarray:
        if self._scaler is None:
            return X.values
        try:
            # Some scalers (like sklearn's) expect feature order matching training.
            return self._scaler.transform(X)
        except Exception:
            # As fallback attempt to align with scaler feature names if present
            try:
                names = getattr(self._scaler, "feature_names_in_", None)
                if names is not None:
                    X2 = X.reindex(columns=names, fill_value=0)
                    return self._scaler.transform(X2)
            except Exception:
                pass
            # last resort
            return X.values

    def predict(self, context, model_input):
        # Accept DataFrame-like input. Ensure pandas DataFrame
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        if model_input.shape[0] == 0:
            return pd.DataFrame({"prediction": []})

        X_aligned = self._align_input(model_input)
        X_prepared = self._apply_scaler(X_aligned)

        # Use sklearn-compatible predict interface
        try:
            preds = self._model.predict(X_prepared)
        except Exception:
            # Some LightGBM objects have a `booster_` that requires `predict` on the booster
            try:
                lg = getattr(self._model, "booster_", self._model)
                preds = lg.predict(X_prepared)
            except Exception as e:
                raise RuntimeError(f"Model prediction failed: {e}")

        # Return a DataFrame with predictions and align index with input
        out = pd.Series(preds, index=model_input.index if hasattr(model_input, "index") else None)
        return pd.DataFrame({"prediction": out})
