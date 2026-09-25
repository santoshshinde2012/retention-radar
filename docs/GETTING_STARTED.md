# Getting started — 10-minute Retention Radar path

Human-in-the-loop churn ranking for a fictional AI platform. Synthetic data only (seed **42**). No real PII. Scores go to a human (`auto_action: none`).

**This repo** = public source code + benchmarks + results analysis.  
Articles are authored separately (**internal**); this repo is the public code home.  
**Data SoR / foundation:** [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse)

## 1. Clone and install

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
# If you skip editable install:
# export PYTHONPATH="$(pwd)/src"
```

Requires **Python 3.12+**. The committed seed-42 bundle was trained with XGBoost **3.4.1** (needs Python 3.12) and scikit-learn **1.9.1**; `requirements.txt` pins both so a local retrain reproduces `models/metrics.json` exactly (`make reproduce` checks it). Or run `make setup` (venv + requirements + editable install).

**macOS note:** XGBoost / LightGBM need OpenMP (CatBoost is usually fine via pip). If `pip install` or import fails with `libomp`, install once:

```bash
brew install libomp
```

Env template: [`.env.example`](../.env.example) (`CHURN_DATA_SOURCE`, `N_USERS`, `N_OPTUNA_TRIALS`).

## 1b. Verify everything end to end (one command)

```bash
make e2e-local
```

Runs, in order: environment check (Python 3.12 + pinned libs) → ruff → seed-42 canary → **exact reproduction** of `models/metrics.json` in an isolated `artifacts/local_e2e/repro/` → full pytest (including headless Streamlit and the API) → every serve surface on the committed bundle (infer, decision packet, batch score, HITL log, HITL outcomes, strict drift, live `uvicorn`, live Streamlit) → lakehouse gold E2E compared with [`results/lakehouse-e2e-summary.json`](../results/lakehouse-e2e-summary.json) when [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) is cloned beside this repo (or `LAKEHOUSE_ROOT=...`) → a check that committed `models/`, `docs/`, `results/`, `data/raw/` were not modified. About 3 minutes on a laptop CPU; the same target runs in CI.

## 2. Run the full pipeline

```bash
chmod +x scripts/run_all.sh
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Faster smoke (committed `models/` untouched): `RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh`. Without `RETENTION_RADAR_ARTIFACT_DIR`, `run_all.sh` retrains **into** `models/` and replaces the published bundle.

Cite **`models/metrics.json`**. Narrative: [results/BENCHMARKS.md](../results/BENCHMARKS.md).

## 3. Score Santosh

```bash
make infer
# equivalent:
# PYTHONPATH=src python -m retention_radar.cli.single_record --user santosh --out artifacts/santosh_decision_packet.json
```

Expect raw ≈ **0.043**, calibrated ≈ **0.016**, band **low**, HITL **monitor** (`auto_action: none`). See [results/SANTOSH_ANALYSIS.md](../results/SANTOSH_ANALYSIS.md).

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

## The service end to end, on real use-case data

[`data/use_cases/`](../data/use_cases/README.md) holds real seed-42 records for every serving path (test-split rows, plus the Santosh hero from the validation split), with the result the committed bundle gives each one:

| Scenario | Band | Suggested action |
|----------|------|------------------|
| Steady power user (Santosh) · Usage dip, still healthy | low | monitor |
| New free trial hitting friction | low | nurture / check-in |
| Borderline: friction just under τ (reviewer overrides to outreach) | medium | nurture / check-in |
| Payment failures plus support load | medium | retention outreach (human review) |
| Gone dark · Enterprise renewal at risk | high | escalate |
| 4 invalid records (unknown plan, missing NPS, rate > 1, text in a number) | — | **hold: fix input data** (never scored or queued) |

```bash
make use-cases   # weekly batch → ranked queue + rejects → packets → held records → reviews → day-30 outcomes
```

| Piece | Command |
|-------|---------|
| Batch → review queue | `python -m retention_radar.cli.batch_score --csv data/use_cases/weekly_batch.csv --out artifacts/use_cases/queue.csv` → sorted by calibrated risk (`rank` 1 first), invalid rows in `queue_rejected.csv` |
| One decision packet | `python -m retention_radar.cli.single_record --json data/use_cases/records/gone_dark.json` → `artifacts/<user_id>_decision_packet.json` (exit 1 + `hold` for invalid input) |
| HITL review log | one row: `python -m retention_radar.cli.hitl_log --from-packet artifacts/santosh_decision_packet.json --reviewer you --action-taken monitor` · bulk (all-or-nothing, re-runs skip already-logged reviews): `--from-scores artifacts/use_cases/queue.csv --decisions data/use_cases/review_decisions.csv --log artifacts/use_cases/hitl_review_log.csv` |
| Outcome write-back | `python -m retention_radar.cli.hitl_outcomes --log artifacts/use_cases/hitl_review_log.csv --labels data/use_cases/labels_day30.csv` |
| Thin local API | `uvicorn retention_radar.serving.api:app --app-dir src` → `POST /v1/churn/score` (422 + hold reason for invalid input), `POST /v1/churn/batch`, `POST /v1/churn/reviews` |
| UI | `make ui` → sidebar **Use case** picker loads each scenario exactly as the API scores it |

This is the **predict → act → outcome** loop: scores, human review rows and later labels are first-class. **Outcome write-back** joins review rows to labels observed after the review (`user_id, churned[, observed_at]`) and reports observed churn per band and per action taken. It is descriptive, not an uplift estimate. `auto_action` stays `none`. Live Streamlit demo URL: **TBD**.
