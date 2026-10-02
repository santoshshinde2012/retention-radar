# Running everything on free platforms

The whole pipeline runs on a laptop, GitHub Actions and a free Streamlit host. No paid
service or credit card is needed. Related: [getting-started.md](../getting-started.md) ·
[architecture.md](../architecture.md).

## The steps

| Step | Command | Output |
|------|---------|--------|
| Generate | `python -m retention_radar.cli.generate_data` | `data/raw/renewals_all.csv`, `renewals_t7.csv`, `subscribers/*.json` |
| Train | `python -m retention_radar.cli.train` | ladder, Optuna XGBoost, calibrator |
| Evaluate | `python -m retention_radar.cli.evaluate` | plots, τ, `models/metrics.json` |
| Latency | `python -m retention_radar.cli.benchmark` | latency block in `metrics.json` |
| Packet | `python -m retention_radar.cli.single_record --user santosh` | `artifacts/santosh_decision_packet.json` |
| Drift | `python -m retention_radar.cli.drift_check` | z-scores against `feature_stats.json` |
| Slices | `python -m retention_radar.cli.slice_metrics` | metrics by `plan_tier` |
| Explain | `python -m retention_radar.cli.explain` | XGBoost gain importance |
| UI | `streamlit run app/streamlit_app.py` | serve-only UI |

All of it at once: `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh`.

Serving never trains. Streamlit, `infer` and `single_record` only load `models/`.

## 1. Laptop

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
pytest -q
streamlit run app/streamlit_app.py
```

Python 3.12 or newer. With an older Python, pip resolves an older XGBoost and the
published numbers no longer reproduce.

Smoke run with the CI settings, leaving `models/` alone:

```bash
RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke N_USERS=800 N_OPTUNA_TRIALS=5 \
  CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

After a full run you should have `models/*.joblib`, `models/metrics.json`,
`artifacts/*.png`, `artifacts/santosh_decision_packet.json` and
`artifacts/arjun_decision_packet.json`.

## 2. GitHub Actions

Workflow: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). Two jobs run on
every push and pull request to `main` (ubuntu-latest, Python 3.12).

`test`:

1. Install requirements, the package and ruff.
2. Copy the committed bundle to `.ci_seed42_fixtures/`.
3. `pytest tests/test_seed42_canary.py tests/test_artifact_dir_isolation.py` on the
   committed bundle (Santosh and Arjun must score as published).
4. `ruff check src/ tests/ app/`.
5. `./scripts/run_all.sh` with `N_USERS=800`, `N_OPTUNA_TRIALS=5`, into `artifacts/smoke/`.
6. `pytest -q`.
7. `python -m retention_radar.cli.drift_check --strict --z-threshold 3.0` on the smoke run.

`e2e-local` clones local-data-lakehouse and runs `make e2e-local`: exact reproduction of
`models/metrics.json`, full pytest, the use-case pack, live API and Streamlit, the
lakehouse run, and a check that committed files were not modified. The lakehouse step
needs the lakehouse repo's v2 export.

CI metrics from the smoke run are not the published numbers. Nothing is uploaded as an
artifact; the job log is the record. On a fork, enable Actions under Settings → Actions.

## 3. Streamlit Community Cloud

Cloud does not run `run_all.sh`. It needs the committed bundle:

| File | Role |
|------|------|
| `models/churn_xgb.joblib` | tuned XGBoost |
| `models/calibrator.joblib` | Platt calibrator |
| `models/feature_names.json` | the 22 feature names in order |
| `models/feature_stats.json` | outlier flags, drift reference, cohort percentiles when the table is absent |
| `models/metrics.json` | Methodology and Benchmarks tabs |

`data/raw/subscribers/*.json` and `data/use_cases/` are committed and feed the sidebar.
`data/raw/renewals_t7.csv` is gitignored, so cohort percentiles come from
`feature_stats.json` on Cloud. Steps: [deploy-later.md](deploy-later.md).

## 4. Hugging Face Spaces

Same idea: host the trained app, do not train on the Space.

1. Create a Space with SDK Streamlit and hardware CPU basic.
2. Point it at this repo or copy the files.
3. Space README header:

```yaml
---
title: Retention Radar
sdk: streamlit
sdk_version: 1.63.0
app_file: app/streamlit_app.py
python_version: 3.12
---
```

4. Include `requirements.txt`, `app/`, `src/`, `models/`, `data/raw/subscribers/`,
   `data/use_cases/`, and `docs/` if you want the doc tabs.

## 5. Lakehouse

```bash
./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse
```

Builds gold from events, syncs it into `data/external/` and runs the pipeline with
`CHURN_DATA_SOURCE=lakehouse` under `artifacts/lakehouse_run/`. Requires the lakehouse
repo's v2 export. CI's `test` job never needs it. Details:
[data-foundation-lakehouse.md](../data/data-foundation-lakehouse.md).

## Not required

Databricks, SageMaker, Azure ML, Vertex, hosted MLflow or W&B, feature stores, Kafka,
Kubernetes, production auth. The FastAPI app (`serving/api.py`) runs locally with no auth
and is not a Cloud target.

## Where the commands land

| Command | Code |
|---------|------|
| `cli.generate_data` | `src/retention_radar/data/generate.py` |
| `cli.train` | `src/retention_radar/training/train.py` |
| `cli.evaluate` | `src/retention_radar/evaluation/evaluate.py` |
| `cli.infer` | `src/retention_radar/serving/infer.py` |
| `cli.single_record` | `src/retention_radar/serving/packet.py` |
| `cli.batch_score` | `src/retention_radar/serving/batch_score.py` |
| `cli.action_log` | `src/retention_radar/serving/action_log.py` |
| `cli.outcomes` | `src/retention_radar/serving/outcomes.py` |
| `cli.drift_check` | `src/retention_radar/serving/drift.py` |
| Streamlit | `app/streamlit_app.py` |
