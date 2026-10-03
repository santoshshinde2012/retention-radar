# Folder structure

Uses a src layout for the package and the cookiecutter-data-science `data/` layers.

```text
retention-radar/
├── README.md, LICENSE, CHANGELOG.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, Makefile
├── pyproject.toml, requirements.txt, requirements.lock, runtime.txt, packages.txt
├── .env.example, .editorconfig, .pre-commit-config.yaml, .streamlit/config.toml
├── .github/          # ci.yml, dependabot.yml, issue and pull-request templates
├── configs/
│   ├── schemas/user_record.schema.json   # the 24-field T-7 record
│   ├── action_log.schema.json
│   └── templates/action_log.csv
├── src/retention_radar/
│   ├── config.py, protocols.py, docs_gen.py
│   ├── cli/          # python -m retention_radar.cli.<name>
│   ├── data/         # generate, ingest, use_cases
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   └── serving/      # infer, scoring, explain, packet, policy, batch_score, action_log, outcomes, api, drift
├── app/streamlit_app.py
├── scripts/          # run_all, run_use_cases, run_lakehouse_e2e, sync_lakehouse_exports, run_local_e2e,
│                     # check_docs (naming + links), fix_macos_libomp (called by make setup)
├── data/
│   ├── raw/          # renewals_t7.csv, renewals_all.csv (generated, gitignored); subscribers/santosh.json, arjun.json
│   ├── use_cases/    # one renewal day: scenarios, invalid records, daily batch, send export, outcomes
│   ├── external/     # lakehouse gold exports (gitignored)
│   ├── interim/      # empty
│   └── processed/    # empty
├── models/           # committed seed-42 serve bundle
├── results/          # benchmarks.md, worked-examples.md, lakehouse-consume-e2e.md, analysis.json,
│                     # lakehouse_e2e_summary.json, sample packet, plots/
├── artifacts/        # runtime output, gitignored
├── notebooks/        # exploration only
├── docs/             # use case, guides, architecture, model card, data, case study
└── tests/
```

## Notes

| Path | Purpose |
|------|---------|
| `src/retention_radar/cli/` | Thin entry points: `train`, `evaluate`, `infer`, `single_record`, `batch_score`, `action_log`, `outcomes`, `drift_check`, `build_use_cases`, `check_reproduction`, `docs_gen`, … |
| `app/` | Serve only; never trains |
| `models/` | Committed so the UI can run on a free host without training |
| `results/` | Committed analysis and charts. `artifacts/` is what a run writes; `make docs-results` copies plots into `results/plots/` |
| `data/use_cases/` | Committed; rebuilt by `python -m retention_radar.cli.build_use_cases` |
| `notebooks/` | Exploration; import the package, keep logic in `src/` |

Generated and scratch: `artifacts/`, `data/raw/*.csv`, `data/external/*`, Optuna databases,
`__pycache__`.

Not used here: Airflow, DVC, MLflow, production auth.
