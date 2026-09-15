# Folder structure — Retention Radar

Teaching / ML use-case layout. **Code + benchmarks + results analysis** live here (public code home).  
Articles are authored separately (**internal**); readers should not be pointed at an articles GitHub repo.  
Data foundation / SoR → [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse).

```text
retention-radar/
├── README.md, LICENSE, CHANGELOG.md, CONTRIBUTING.md, Makefile
├── pyproject.toml, requirements.txt, requirements.lock, runtime.txt, .gitignore
├── .github/workflows/ci.yml
├── src/retention_radar/           # ONLY Python package under src/
│   ├── __init__.py
│   ├── config.py, protocols.py, docs_gen.py
│   ├── cli/                       # train, evaluate, infer, single_record, …
│   ├── data/, features/, training/, evaluation/, serving/
├── app/streamlit_app.py
├── scripts/                       # shell orchestration only
├── schemas/user_record.schema.json
├── data/raw/, data/external/
├── models/                        # seed-42 serve bundle (joblibs + metrics)
├── results/                       # BENCHMARKS, SANTOSH_ANALYSIS, plots/, samples
├── artifacts/                     # runtime-only (.gitkeep committed; * gitignored)
├── docs/
│   ├── README.md                  # this index
│   ├── GETTING_STARTED.md
│   ├── FOLDER_STRUCTURE.md
│   ├── ARCHITECTURE.md
│   ├── MODEL_CARD.md
│   ├── guides/
│   ├── data/
│   └── case-study/
└── tests/
```

## Folder purposes

| Path | Purpose |
|------|---------|
| `src/retention_radar/` | Product package (SOLID boundaries) |
| `src/retention_radar/cli/` | CLI entrypoints (`python -m retention_radar.cli.*`) |
| `app/` | Streamlit — **serve-only** |
| `scripts/` | E2E + lakehouse sync helpers (`make run` / `make run-lakehouse`) |
| `models/` | Seed-42 serve bundle — **commit** for Cloud |
| `results/` | Committed analysis + charts (not runtime) |
| `artifacts/` | Local evaluate / infer dumps — **never commit** |
| `docs/` | Engineering docs + model card + nested guides |
| `data/external/` | Lakehouse gold sync (CSVs gitignored) |

**Product vs scratch:** package + `app/` + committed `models/` + `results/` are product. `artifacts/`, generated `data/raw/users.csv`, Optuna DBs, and `__pycache__` are scratch.

**Charts:** `artifacts/` is what evaluate writes at runtime. `results/plots/` is the committed pack. Refresh with `make docs-results`.

More: [GETTING_STARTED.md](GETTING_STARTED.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [../results/README.md](../results/README.md)
