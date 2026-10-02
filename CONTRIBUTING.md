# Contributing

Repo: [github.com/santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar).
Keep pull requests focused: code, benchmarks, results.

## Setup

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.lock` is a snapshot for reference; CI installs from `requirements.txt`.

## Run the pipeline

```bash
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

This retrains into `models/` and replaces the committed bundle. For a quick run that
leaves `models/` alone:

```bash
RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke N_USERS=800 N_OPTUNA_TRIALS=5 \
  CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Cite `models/metrics.json`.

## Regenerate the model card and data dictionary

```bash
python -m retention_radar.cli.docs_gen
```

Writes `docs/model-card.md` and `docs/data/data-dictionary.md`. Edit those through
`docs_gen` rather than by hand, so they match `models/metrics.json`.

## Tests

```bash
CHURN_DATA_SOURCE=synthetic pytest -q     # or: make test
```

Pull requests must keep the suite green. `CHURN_DATA_SOURCE=synthetic` stops a local
`data/external/` export from changing what is tested. `make e2e-local` runs the full
check that CI runs.

The lakehouse run (optional) trains under `artifacts/lakehouse_run/` but rewrites the
committed `results/lakehouse_e2e_summary.json`, and needs the lakehouse repo's v2 export:

```bash
./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse
git checkout -- results/lakehouse_e2e_summary.json   # unless you mean to publish it
```

## Conventions

- Open-source tools only in the core path. See [docs/guides/e2e-free-platforms.md](docs/guides/e2e-free-platforms.md).
- Keep the synthetic-data notice visible.
- The service never sends anything: `auto_action` stays `none` in `src/retention_radar/serving/policy.py`.
- Playbook effects in `config.PLAYBOOKS` are assumptions; do not describe them as measured.
- Code goes in `src/retention_radar/`; CLI entry points in `src/retention_radar/cli/`.
- No ROI or fairness claims.
- Checklist: [docs/guides/best-practices.md](docs/guides/best-practices.md).

## License

MIT © Santosh Shinde, see [LICENSE](LICENSE).
