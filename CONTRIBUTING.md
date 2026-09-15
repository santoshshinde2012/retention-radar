# Contributing

Thanks for improving **Retention Radar**, a FOSS AI-platform churn teaching repo.  
Repo: [https://github.com/santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

Prefer focused PRs: code + engineering docs only (no Medium article trees).

## Setup

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional informational pins: requirements.lock (CI still uses requirements.txt)
export PYTHONPATH="$(pwd)"
```

## Run the full pipeline

```bash
chmod +x scripts/run_all.sh
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Primary citation file: **`models/metrics.json`**.

Faster smoke: `N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh`

## Refresh model card & data dictionary

```bash
python -m src.docs_gen
```

Writes `MODEL_CARD.md` and `docs/data-dictionary.md`.

## Tests

```bash
export PYTHONPATH="$(pwd)"
export CHURN_DATA_SOURCE=synthetic
pytest -q
```

**PRs should keep pytest green.** Pin `CHURN_DATA_SOURCE=synthetic` so a local
`data/external/` lakehouse export cannot change what CI measures.

Lakehouse E2E (optional, overwrites `models/`):

```bash
./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse
git checkout -- models/ MODEL_CARD.md docs/data-dictionary.md
```

## Style notes

- FOSS only — no paid SaaS in the core path. See [docs/e2e-free-platforms.md](docs/e2e-free-platforms.md).
- Synthetic data disclaimer stays visible; HITL only (`auto_action: none`) in `src/retention_radar/serving/policy.py`.
- New logic goes in `src/retention_radar/`; keep `src/*.py` shims so `python -m src.*` stays stable.
- Do not claim production ROI or fairness audits from this teaching repo.
- Prefer editing generated docs via `docs_gen` so they stay in sync with `models/metrics.json`.

## License

MIT © Santosh Shinde — see [LICENSE](LICENSE).
