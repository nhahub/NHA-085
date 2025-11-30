# Deployment structure and folder descriptions

This document maps the repository structure to the items used during deployment and MLOps. It explains what each top-level folder contains, why it's important for deployment, and where runtime configuration or artifacts live.

> Quick note: keep README_DEPLOY.md and this file aligned — README_DEPLOY.md contains deployment steps and environment configuration, while this document focuses only on the structural mapping of files & folders.

---

## Top-level files

- `app.py` 🔧
  - Dash/Plotly application entrypoint (production UI). Loads model artifacts and serves the forecast UI.
  - Important for deployment: update model paths and environment variables here if packaging the app into a container or serverless function.

- `run.ps1` / project-level PowerShell scripts ⚙️
  - Dev helper for launching the app or environment on Windows.

- `README.md` / `README_DEPLOY.md` 📚
  - Documentation and deployment instructions. `README_DEPLOY.md` contains step-by-step deploy notes and should be the primary reference during rollout.

- `requirements.txt` / `requirements-frozen.txt` 🧩
  - Python dependency lists used to create reproducible environments (venv, Docker images, CI pipelines).

- `.venv/` (local) 🧪
  - Local virtual environment used during development. Not used directly in deployment (Docker/CI should re-create dependencies based on requirements files).

---

## Folders (what they contain and deployment relevance)

- `dataset/` 🗂
  - Contains input CSVs (`train.csv`, `test.csv`, `store.csv`) used for model training and example validation.
  - Not usually packaged for production, but critical for reproducing training runs and unit tests.

- `scripts/` 🔧
  - Core training, prediction, logging and helper scripts used during model development and MLOps.
  - Key scripts and their roles:
    - `train.py` — CLI to train either SARIMAX or LightGBM models and log runs to MLflow (used in CI for retraining and production runbooks).
    - `predict.py` — command-line inference helper used by the UI or backfill jobs.
    - `data_loader.py` — consistent dataset loading + preprocessing logic shared by notebook and scripts (keep this file stable, used by pipelines).
    - `lgbm_pyfunc.py` / `sarimax_pyfunc.py` — MLflow pyfunc wrappers for consistent serving of LightGBM & SARIMAX models in MLflow-based deployments.
    - `log_lgbmlflow.py`, `list_mlflow_artifacts.py` — utilities to retroactively log model artifacts to MLflow and to inspect experiment artifacts.

- `models/` 🧾
  - Serialized model artifacts (joblib), e.g. `lgb_artifacts.joblib`, `sarimax_artifacts.joblib`.
  - Production packaging: models stored here are the local persisted artifacts — during deployment you may prefer to fetch artifacts from MLflow or a model registry instead of shipping files with the image.

- `mlruns/` and `mlflow.db` 🧭
  - Local MLflow tracking store (runs and experiment metadata). For production, configure a central MLflow tracking/registry or S3/remote storage rather than the local folder.

- `artifacts/` / `results/` 🎯
  - Generated artifacts, reports, prediction CSVs, or model outputs created during training, evaluation, or UI inference (e.g., `results/sarimax_ui_predictions.csv`).
  - Helpful for lightweight deployments or demos; in production a storage backend (S3, Azure Blob) is recommended.

- `reports/` 📝
  - Human-readable documentation, result summaries, and exported deliverables (e.g., Word/PDF reports used by stakeholders).

- `assets/` / `style.css` 🎨
  - UI assets used by the Dash app; static files included with the dashboard.

- `docs/` 📁
  - Longer-form documentation and deployment guides. This repository now contains `docs/deployment_structure.md` (this file) and other supporting docs.

- `results/` 📈
  - Saved predictions, CSV outputs, and model evaluation outputs used by the UI or for debugging.

---

## Deployment & runtime notes

- Environment & dependencies
  - Use `requirements.txt` for lightweight environments and `requirements-frozen.txt` for fully pinned builds (production Docker images, reproducible pipelines).
  - Validate Python version in the runtime environment (the project used Python 3.11 in development).

- Model storage & serving
  - Prefer storing production-ready model artifacts in a model registry (MLflow Model Registry or cloud-based model store) rather than shipping `models/` in deployment images.
  - The `scripts/*.py` pyfunc wrappers are provided to ensure consistent behavior when loading artifacts from MLflow.

- CI/CD & Infrastructure
  - Continuous training: use `scripts/train.py` in scheduled jobs or CI pipelines; configure MLflow experiment names, remote artifact stores, and model promotion to registry.
  - For the Dash app, create a container that installs `requirements.txt`, loads the selected model (from `models/` or MLflow), and exposes `app.py` on the desired port.