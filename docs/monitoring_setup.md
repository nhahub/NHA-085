# Monitoring Setup for Rossmann Sales Forecasting

This document explains a recommended monitoring and observability setup for the project: components, metrics to collect, dashboards to build, alerting rules, and operational runbook items.

## Goals
- Detect model/data drift and performance regressions.
- Monitor service health, latency, and availability.
- Alert engineers on high-severity incidents.
- Provide dashboards for model owners and SREs.

## Recommended stack (local/dev-friendly and production)
- Metrics: Prometheus (scraping target metrics endpoints)
- Dashboards: Grafana
- Logs: Loki (Grafana Loki) or ElasticSearch + Kibana
- Alerting: Prometheus Alertmanager
- Model artifacts and experiment tracking: MLflow (with file store or remote tracking + S3 artifact storage)
- (Optional) Data validation: Great Expectations for batch checks

## What to instrument
1. Service / App metrics (Dash app)
   - process_up (bool)
   - request_count (per endpoint)
   - request_latency_seconds (histogram)
   - request_errors_total (5xx/4xx)

2. Model & prediction metrics
   - predictions_count
   - prediction_latency_seconds (histogram)
   - prediction_mean, prediction_p50/p95 (summary stats)
   - model_version (label)

3. Data-quality metrics
   - input_row_count
   - percent_missing_columns
   - schema_mismatch_count
   - feature_distribution_* (e.g., histograms or sketches for numeric features)

4. Model quality metrics (if labels available)
   - rolling_rmse (or other loss metrics computed daily/weekly)
   - bias_metrics (mean prediction - mean actual)

5. Infrastructure metrics
   - CPU, memory, disk usage for host/container
   - process restart count

## Exporters & Libraries
- Python Prometheus client: `prometheus_client` for instrumenting the Dash app and helper scripts.
- Expose `/metrics` endpoint on the app that Prometheus scrapes.
- If using Gunicorn/WSGI front, use a metrics middleware or an exporter sidecar.

## Prometheus scraping configuration (example)
- Add a job in `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'rossmann_app'
    static_configs:
      - targets: ['rossmann-app:8000']  # change to your host:port
```

## Example metrics to emit from Python (prometheus_client)
```python
from prometheus_client import Counter, Histogram, Gauge

REQUEST_COUNT = Counter('request_count', 'Total request count', ['endpoint', 'method', 'status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
PREDICTIONS = Counter('predictions_count', 'Number of predictions made')
MODEL_VERSION = Gauge('model_version', 'Model version', [])
```

Wrap prediction endpoints with timers and increment counters accordingly.

## Grafana dashboards (recommended panels)
- App health overview (up, CPU, memory, request rate, errors)
- Prediction overview (prediction rate, latency histogram)
- Data quality (missing rates, schema mismatch alerts)
- Model performance (rolling RMSE, bias, calibration)

## Alerting (Alertmanager rules examples)
- High error rate:
  - If request_errors_total / request_count > 5% for 5 minutes -> P1
- Prediction latency spike:
  - If p95(prediction_latency_seconds) > 2s for 10 minutes -> P2
- Model performance degradation:
  - If rolling_rmse increases by > 20% compared to baseline for 24h -> P2
- Data drift:
  - If KL divergence or a simple distribution shift metric for a key feature > threshold -> P2

## Deployment checklist
- Add `/metrics` to app and ensure local Prometheus can scrape it.
- Deploy Prometheus + Grafana (docker-compose for dev or helm charts to Kubernetes).
- Create Grafana dashboards and import panels for the app and model metrics.
- Configure Alertmanager and receiver channels (Slack, email, PagerDuty).

## Runbook snippets
- Service down: Check host/container logs, restart service, verify `/metrics` and health endpoints.
- High prediction error: verify recent training runs in MLflow, check input distributions, roll back to previous artifact if necessary.
- Data drift: freeze incoming data sample, run local EDA comparing to training distribution, open an incident and escalate.

## Automation & Maintenance
- Add a periodic job to compute batch metrics (daily) and push to Prometheus pushgateway or ingest into time-series store.
- Automate retraining pipelines when drift and performance triggers are satisfied (carefully gated with human approval).

## References
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/
- MLflow: https://mlflow.org/
- Great Expectations: https://greatexpectations.io/

