import joblib
import pandas as pd
import numpy as np

class SarimaxPyfuncModel:
    """Light wrapper implementing the mlflow.pyfunc PythonModel contract for SARIMAX results.

    This class is intentionally lightweight and only depends on joblib/pandas/numpy.
    The loader expects an artifact key named 'sarimax_artifacts' which points to a
    joblib file produced by `scripts/train.py` containing a dict with keys including
    'results' (the fitted SARIMAXResults) and 'train_end' (last training timestamp).
    """

    def load_context(self, context):
        # context.artifacts is a dict mapping the artifact name to local path
        art_path = None
        # prefer explicit key
        if "sarimax_artifacts" in getattr(context, "artifacts", {}):
            art_path = context.artifacts["sarimax_artifacts"]
        else:
            # fallback: try common keys
            for k in getattr(context, "artifacts", {}).keys():
                if k.lower().startswith("sarimax") or k.lower().endswith("artifacts"):
                    art_path = context.artifacts[k]
                    break

        if art_path is None:
            raise RuntimeError("sarimax artifact not found in context.artifacts; expected key 'sarimax_artifacts'")

        # load joblib artifact
        self._artifacts = joblib.load(art_path)
        # extract fitted results
        self._results = self._artifacts.get("results")
        self._train_end = self._artifacts.get("train_end")

        if self._results is None:
            raise RuntimeError("Loaded SARIMAX artifact does not contain 'results' object")

    def predict(self, context, model_input):
        """Predict method called by mlflow.pyfunc.

        Behavior:
        - If model_input is a pandas DataFrame, perform an n-step forecast where n = len(model_input).
        - Return a pandas.DataFrame with a single column 'prediction' and an index matching
          model_input.index when it's a DatetimeIndex; otherwise a RangeIndex is used.
        """
        # ensure pandas
        if not isinstance(model_input, pd.DataFrame):
            # try to coerce
            model_input = pd.DataFrame(model_input)

        n_steps = len(model_input)
        if n_steps == 0:
            return pd.DataFrame({"prediction": []})

        # Use SARIMAXResults.get_forecast(steps=n_steps) for out-of-sample predictions
        # This avoids index alignment complexity; we'll attach an index after.
        forecast_obj = self._results.get_forecast(steps=n_steps)
        preds = forecast_obj.predicted_mean

        # If input index is datetime-like, create a date index starting after train_end with weekly freq
        try:
            input_index = model_input.index
            if isinstance(input_index, pd.DatetimeIndex):
                # Attempt to infer frequency; default to weekly to match training
                freq = input_index.freq or pd.infer_freq(input_index)
                if freq is None:
                    # default to weekly as the models were trained on weekly resampled data
                    freq = "W"
                start = pd.to_datetime(self._train_end) + pd.tseries.frequencies.to_offset(freq)
                new_index = pd.date_range(start=start, periods=n_steps, freq=freq)
                preds.index = new_index
            else:
                # keep default RangeIndex
                preds.index = pd.RangeIndex(start=0, stop=n_steps, step=1)
        except Exception:
            preds.index = pd.RangeIndex(start=0, stop=n_steps, step=1)

        return pd.DataFrame({"prediction": preds})
