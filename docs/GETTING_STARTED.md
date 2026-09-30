# Getting started

Retention Radar scores paying subscribers of a monthly AI coding assistant plan seven
days before renewal (T-7) and suggests one approved playbook, a holdout, or nothing.
The data is synthetic (seed 42). The service never sends anything: `auto_action` is
always `none`. Background: [USE_CASE.md](USE_CASE.md).

## 1. Install

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Python 3.12 or newer is required. The committed bundle was trained with XGBoost 3.4.1 and
scikit-learn 1.9.1; `requirements.txt` pins both so a retrain reproduces
`models/metrics.json` (`make reproduce` checks this). `make setup` does the same install.

On macOS, if XGBoost or LightGBM fail to import with a `libomp` error: `brew install libomp`.

Environment variables are listed in [`.env.example`](../.env.example)
(`CHURN_DATA_SOURCE`, `N_USERS`, `N_OPTUNA_TRIALS`).

## 2. Run the pipeline

```bash
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Generate → ingest → train → evaluate → slices → explain → benchmark → drift check →
decision packets for Maya and Arjun. Without `RETENTION_RADAR_ARTIFACT_DIR` it retrains
into `models/` and replaces the committed bundle. For a fast smoke run that leaves
`models/` alone:

```bash
RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke N_USERS=800 N_OPTUNA_TRIALS=5 \
  CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Cite `models/metrics.json`. Explanation of the numbers: [results/BENCHMARKS.md](../results/BENCHMARKS.md).

## 3. Score one subscriber

```bash
make infer
# same as:
python -m retention_radar.cli.single_record --user maya --out artifacts/maya_decision_packet.json
```

Maya: calibrated P(lapse) 0.288, band medium, action `limit_reset`. `--user arjun` gives
0.025, low, `no_action`. Walkthrough: [results/WORKED_EXAMPLES.md](../results/WORKED_EXAMPLES.md).

## 4. UI and tests

```bash
make ui     # Streamlit, committed models only
make test   # pytest with CHURN_DATA_SOURCE=synthetic
```

## 5. Verify everything

```bash
make e2e-local
```

In order: environment check, ruff, the seed-42 check on the committed bundle, exact reproduction of
`models/metrics.json` in `artifacts/local_e2e/repro/`, full pytest, the use-case pack
through every CLI, a live `uvicorn` and a live Streamlit server, the lakehouse path when
[local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) is
cloned beside this repo (or `LAKEHOUSE_ROOT=...`), and a check that committed `models/`,
`docs/`, `results/` and `data/raw/` were not modified. CI runs the same target.

## 6. One renewal day, end to end

[`data/use_cases/`](../data/use_cases/README.md) is one day of T-7 renewals built from
test-split subscribers: eight scenarios (one per path through the policy), four invalid
records, a daily batch, the messaging tool's send export and the renewal outcomes.

```bash
make use-cases
```

| Step | Command |
|------|---------|
| Batch → action queue | `python -m retention_radar.cli.batch_score --csv data/use_cases/daily_t7_batch.csv --out artifacts/use_cases/queue.csv` (invalid rows go to `queue_rejected.csv`) |
| One decision packet | `python -m retention_radar.cli.single_record --json data/use_cases/records/overage_shock.json` (exit 1 and `hold: fix input data` for invalid input) |
| Import what was sent | `python -m retention_radar.cli.action_log --from-scores artifacts/use_cases/queue.csv --decisions data/use_cases/actions_taken.csv --log artifacts/use_cases/action_log.csv` |
| Lift vs holdout | `python -m retention_radar.cli.outcomes --log artifacts/use_cases/action_log.csv --labels data/use_cases/renewal_outcomes.csv` |
| API | `uvicorn retention_radar.serving.api:app --app-dir src` → `POST /v1/churn/score`, `/v1/churn/batch`, `/v1/churn/actions` |
| UI | `make ui`, then pick a subscriber in the sidebar |

The action log records what was actually done, including holdout rows and suppressed
sends. `cli.outcomes` joins it to renewal outcomes observed after the action and reports
lapse rate per band and lift per playbook against the holdout, with a Newcombe interval.
It gives no verdict while either group has fewer than 30 subscribers. In the pack, the
outcomes are simulated from the playbook effects assumed in `config.PLAYBOOKS`, so the
lift it reports is not evidence that any playbook works.

## Data sources

| Path | Command |
|------|---------|
| Synthetic (default, CI) | `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh` |
| Lakehouse | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` (needs its v2 renewal export) |

The lakehouse run writes only under `artifacts/lakehouse_run/`. Details:
[data/data-foundation-lakehouse.md](data/data-foundation-lakehouse.md).

More: [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) · [ARCHITECTURE.md](ARCHITECTURE.md) ·
[guides/e2e-free-platforms.md](guides/e2e-free-platforms.md)
