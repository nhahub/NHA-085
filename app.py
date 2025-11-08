"""Simple Dash UI to forecast using saved SARIMAX artifact.

Run with:
    .\.venv\Scripts\python.exe app.py

Features:
- Upload a CSV with a `Date` column (or use `dataset/test.csv`).
- Choose forecast horizon (weeks) and run SARIMAX forecast using saved artifacts in `models/`.
- Shows an interactive time-series plot and allows downloading predictions.
"""
import io
from pathlib import Path
import base64

import pandas as pd
import joblib
import plotly.graph_objs as go

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State


APP = Dash(__name__)
APP.title = "Rossmann SARIMAX Forecast"

ARTIFACT_PATH = Path("models/sarimax_artifacts.joblib")


def load_artifact(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def parse_uploaded(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")), parse_dates=["Date"], index_col="Date")
            return df
        else:
            return None
    except Exception as e:
        print(e)
        return None


APP.layout = html.Div(className="container", children=[
    html.Header(className="header", children=[
        html.Div(className="brand", children=[
            html.H1("Rossmann SARIMAX Forecast"),
            html.P("Interactive forecasting UI — upload data or use the sample test set")
        ])
    ]),

    html.Main(className="main-card", children=[
        html.Section(className="controls", children=[
            html.Div(className="control-item sample-check", children=[
                dcc.Checklist(id="use-sample", options=[{"label": "Use dataset/test.csv", "value": "sample"}], value=[])
            ]),

            html.Div(className="control-item upload", children=[
                dcc.Upload(id="upload-data",
                           children=html.Div([html.Span("Drag & drop or "), html.A("select a CSV file")]),
                           className="upload-drop",
                           multiple=False),
            ]),

            html.Div(className="control-item horizon", children=[
                html.Label("Forecast horizon (weeks):", htmlFor="horizon"),
                dcc.Input(id="horizon", type="number", value=4, min=1, step=1, className="input-number"),
                html.Button("Forecast", id="forecast-btn", n_clicks=0, className="btn btn-primary"),
            ])
        ]),

        html.Div(id="status", className="status"),

        html.Section(className="graph-card", children=[
            dcc.Loading(id="loading", children=[dcc.Graph(id="ts-graph")], type="default")
        ]),

        html.Div(id="download-link", className="download")
    ]),

    html.Footer(className="footer", children=[
        html.Small("Built with Dash — SARIMAX artifacts expected at models/sarimax_artifacts.joblib")
    ])
])


@APP.callback(
    Output("status", "children"),
    Input("forecast-btn", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("use-sample", "value"),
    State("horizon", "value"),
)
def run_forecast(n_clicks, contents, filename, use_sample, horizon):
    if n_clicks is None or n_clicks == 0:
        return ""

    artifacts = load_artifact(ARTIFACT_PATH)
    if artifacts is None or artifacts.get("model_type") != "sarimax":
        return "SARIMAX artifact not found at models/sarimax_artifacts.joblib"

    results = artifacts.get("results")
    if results is None:
        return "SARIMAX results not found in artifact"

    # Load data
    if use_sample and "sample" in use_sample:
        sample_path = Path("dataset/test.csv")
        if not sample_path.exists():
            return "dataset/test.csv not found"
        df = pd.read_csv(sample_path, parse_dates=["Date"], index_col="Date")
    elif contents is not None and filename is not None:
        df = parse_uploaded(contents, filename)
        if df is None:
            return "Failed to parse uploaded file. Ensure a CSV with a 'Date' column." 
    else:
        return "No input data provided. Upload a CSV or check 'Use dataset/test.csv'."

    # Resample weekly and compute forecast
    numeric = df.select_dtypes(include="number")
    weekly = numeric.resample("W").mean()
    steps = int(horizon) if horizon is not None else 4

    try:
        pred = results.get_forecast(steps=steps)
        pred_mean = pred.predicted_mean
    except Exception as e:
        return f"Forecasting failed: {e}"

    # Store predictions to a CSV
    out = pd.DataFrame({"prediction": pred_mean})
    out_path = Path("results/sarimax_ui_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)

    # Prepare graph data and update a hidden div via client side (we only set status here)
    return f"Forecast complete — saved to {out_path.as_posix()} (horizon={steps} weeks)"


@APP.callback(
    Output("ts-graph", "figure"),
    Input("status", "children")
)
def update_graph(status_text):
    # When status updates, read predictions and show simple plot
    traces = []
    # historical
    sample_path = Path("dataset/test.csv")
    if sample_path.exists():
        try:
            df = pd.read_csv(sample_path, parse_dates=["Date"], index_col="Date")
            numeric = df.select_dtypes(include="number")
            weekly = numeric.resample("W").mean()
            if "Sales" in weekly.columns:
                traces.append(go.Scatter(x=weekly.index, y=weekly["Sales"], name="Historical Sales"))
        except Exception:
            pass

    pred_path = Path("results/sarimax_ui_predictions.csv")
    if pred_path.exists():
        try:
            p = pd.read_csv(pred_path, parse_dates=[0], index_col=0)
            traces.append(go.Scatter(x=p.index, y=p["prediction"], name="Forecast", line=dict(dash="dash")))
        except Exception:
            pass

    if not traces:
        fig = go.Figure()
        fig.update_layout(title="No data to display")
        return fig

    fig = go.Figure(data=traces)
    fig.update_layout(title="Historical and Forecasted Sales", xaxis_title="Date", yaxis_title="Value")
    return fig


if __name__ == "__main__":
    # Dash v3 replaced `run_server` with `run`. Support both to be compatible
    # with older and newer Dash versions.
    runner = getattr(APP, "run", None)
    if callable(runner):
        runner(debug=True, host="127.0.0.1", port=8050)
    else:
        APP.run_server(debug=True, host="127.0.0.1", port=8050)
