# Getting started — 10-minute Retention Radar path

Human-in-the-loop churn ranking for a fictional AI platform. Synthetic data only (seed **42**). No real PII. Scores go to a human (`auto_action: none`).

**This repo** = public source code + benchmarks + results analysis.  
Articles are authored separately (**internal**); this repo is the public code home.  
**Data SoR / foundation:** [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse)

## 1. Clone and install

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
# If you skip editable install:
# export PYTHONPATH="$(pwd)/src"
```

Requires **Python 3.11+**. The committed `models/calibrator.joblib` was trained with scikit-learn **1.9.1**, so use this runtime to install dependencies and load the serving bundle. Or run `make setup` (venv + requirements + editable install).

**macOS note:** XGBoost / LightGBM need OpenMP (CatBoost is usually fine via pip). If `pip install` or import fails with `libomp`, install once:

```bash
brew install libomp
```

Env template: [`.env.example`](../.env.example) (`CHURN_DATA_SOURCE`, `N_USERS`, `N_OPTUNA_TRIALS`).

## 2. Run the full pipeline

```bash
chmod +x scripts/run_all.sh
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Faster smoke: `N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh`

Cite **`models/metrics.json`**. Narrative: [results/BENCHMARKS.md](../results/BENCHMARKS.md).

## 3. Score Santosh

```bash
make infer
# equivalent:
# PYTHONPATH=src python -m retention_radar.cli.single_record --user santosh --out artifacts/santosh_decision_packet.json
```

Expect raw ≈ **0.043**, calibrated ≈ **0.017**, band **low**, HITL **monitor** (`auto_action: none`). See [results/SANTOSH_ANALYSIS.md](../results/SANTOSH_ANALYSIS.md).

## 4. UI and tests

```bash
streamlit run app/streamlit_app.py
# or: make ui

CHURN_DATA_SOURCE=synthetic PYTHONPATH=src pytest -q
# or: make test
```

## Dual path: synthetic vs lakehouse

| Path | Command |
|------|---------|
| Synthetic (default / CI) | `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh` |
| Lakehouse gold | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` |

Lakehouse E2E sets `RETENTION_RADAR_ARTIFACT_DIR=artifacts/lakehouse_run` so committed `models/` and `docs/MODEL_CARD.md` stay untouched. Dual-world cite: [`results/lakehouse-e2e-summary.json`](../results/lakehouse-e2e-summary.json).

Sync only: `./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export`

More: [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [e2e-free-platforms.md](guides/e2e-free-platforms.md) · [../README.md](../README.md)

## FOSS production-shaped extras (optional)

| Piece | Command |
|-------|---------|
| Batch gold scores | `python -m retention_radar.cli.batch_score --csv data/external/churn_user_features.csv` |
| HITL review log | `python -m retention_radar.cli.hitl_log --from-packet artifacts/santosh_decision_packet.json --reviewer you --action-taken monitor` |
| Thin local API | `uvicorn retention_radar.serving.api:app --app-dir src` → `POST /v1/churn/score` |
| Outcome write-back | `python -m retention_radar.cli.hitl_outcomes --labels data/raw/users.csv` → `artifacts/hitl_outcomes.{csv,json}` |

These start a **predict → act → outcome** loop: scores and human review rows are first-class. **Outcome write-back** joins review rows to labels observed after the review (`user_id, churned[, observed_at]`) and reports observed churn per band and per action taken. It is descriptive, not an uplift estimate. `auto_action` stays `none`. Live Streamlit demo URL: **TBD**.

