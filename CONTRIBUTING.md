# Contributing

Thanks for improving **Retention Radar**, a FOSS AI-platform churn teaching repo.  
Repo: [https://github.com/santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

Prefer focused PRs: code + benchmarks + results analysis. Articles are authored separately (internal); this repo is the public code home.

## Setup

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Optional informational pins: requirements.lock (CI still uses requirements.txt)
# If you skip editable install: export PYTHONPATH="$(pwd)/src"
```

## Run the full pipeline

```bash
chmod +x scripts/run_all.sh
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Primary citation file: **`models/metrics.json`**.

Faster smoke (committed `models/` untouched): `RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh`. Without `RETENTION_RADAR_ARTIFACT_DIR`, `run_all.sh` retrains **into** `models/` and replaces the published bundle.

## Refresh model card & data dictionary

```bash
python -m retention_radar.cli.docs_gen
```

Writes `docs/MODEL_CARD.md` and `docs/data/data-dictionary.md`.

## Tests

```bash
export PYTHONPATH="$(pwd)/src"
export CHURN_DATA_SOURCE=synthetic
pytest -q
```

**PRs should keep pytest green.** Pin `CHURN_DATA_SOURCE=synthetic` so a local
`data/external/` lakehouse export cannot change what CI measures.

Lakehouse E2E (optional; trains under `artifacts/lakehouse_run/`, but refreshes the committed `results/lakehouse-e2e-summary.json`):

```bash
./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse
git checkout -- results/lakehouse-e2e-summary.json   # unless you mean to publish it
```

## Style notes

- FOSS only — no paid SaaS in the core path. See [docs/guides/e2e-free-platforms.md](docs/guides/e2e-free-platforms.md).
- Synthetic data disclaimer stays visible; HITL only (`auto_action: none`) in `src/retention_radar/serving/policy.py`.
- New logic goes in `src/retention_radar/`; CLI entrypoints live under `src/retention_radar/cli/` (`python -m retention_radar.cli.*`).
- Do not claim production ROI or fairness audits from this teaching repo.
- Prefer editing generated docs via `docs_gen` so they stay in sync with `models/metrics.json`.

## License

MIT © Santosh Shinde — see [LICENSE](LICENSE).
