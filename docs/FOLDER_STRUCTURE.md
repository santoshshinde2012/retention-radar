# Folder structure — Retention Radar

Teaching / ML use-case layout aligned with **cookiecutter-data-science** data layers and **src-layout** packaging.  
**Code + benchmarks + results analysis** live here (public code home).  
Articles are authored separately (**internal**); readers should not be pointed at an articles GitHub repo.  
Data foundation / SoR → [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse).

```text
retention-radar/
├── README.md, LICENSE, CHANGELOG.md, CONTRIBUTING.md, Makefile
├── pyproject.toml, requirements.txt, requirements.lock, runtime.txt
├── .env.example, .gitignore
├── .github/workflows/ci.yml
├── configs/                         # production convention: contracts / config
│   ├── README.md
│   ├── hitl_review_log.schema.json
│   ├── templates/hitl_review_log.csv
│   └── schemas/
│       └── user_record.schema.json
├── src/retention_radar/             # ONLY Python package under src/
│   ├── __init__.py
│   ├── config.py, protocols.py, docs_gen.py
│   ├── cli/                         # train, evaluate, infer, batch_score, hitl_log, …
│   ├── data/, features/, training/, evaluation/, serving/  # + thin FastAPI
├── app/streamlit_app.py             # Streamlit serve-only
├── scripts/                         # shell orchestration only
├── data/
│   ├── README.md                    # raw / interim / processed / external roles
│   ├── raw/                         # Santosh JSON; users.csv (gitignored)
│   ├── interim/                     # CDS parity (empty teaching path)
│   ├── processed/                   # CDS parity (features usually in-memory)
│   └── external/                    # lakehouse gold sync
├── models/                          # seed-42 serve bundle (joblibs + metrics)
├── results/                         # ≡ reports/ in CDS templates (name kept for dig-deeper URLs)
│   ├── README.md, BENCHMARKS.md, SANTOSH_ANALYSIS.md
│   └── plots/                       # ≡ reports/figures
├── artifacts/                       # runtime-only (.gitkeep; * gitignored)
├── notebooks/                       # exploration only; import package
├── docs/
│   ├── README.md
│   ├── GETTING_STARTED.md
│   ├── FOLDER_STRUCTURE.md
│   ├── ARCHITECTURE.md
│   ├── MODEL_CARD.md
│   ├── guides/
│   ├── data/
│   └── case-study/
└── tests/
```

## Alignment notes

| Convention | How we follow it |
|------------|------------------|
| **src-layout** | Package only under `src/retention_radar/`; `pip install -e .` |
| **CDS data layers** | `data/{raw,interim,processed,external}/` with README |
| **configs/** | JSON Schema + config home (not scattered at repo root) |
| **notebooks/** | Exploration only — no production train/serve logic |
| **results/** (not `reports/`) | Same role as CDS `reports/` + `reports/figures/` → `results/plots/`; **name kept** so public article dig-deeper links to `results/BENCHMARKS.md` stay valid |
| **Out of scope** | Airflow, DVC, MLflow, production auth / CRM write-back |
| **Optional teaching serve** | Thin local FastAPI (`serving/api.py`) — no auth |

## Folder purposes

| Path | Purpose |
|------|---------|
| `src/retention_radar/` | Product package (SOLID boundaries) |
| `src/retention_radar/cli/` | CLI entrypoints (`python -m retention_radar.cli.*`) |
| `configs/schemas/` | User-record JSON Schema (`USER_RECORD_SCHEMA_PATH`) |
| `configs/templates/` | HITL review-log CSV header template |
| `serving/api.py` | Optional thin local FastAPI (`POST /v1/churn/score`) |
| `app/` | Streamlit — **serve-only** |
| `scripts/` | E2E + lakehouse sync helpers (`make run` / `make run-lakehouse`) |
| `models/` | Seed-42 serve bundle — **commit** for Cloud |
| `results/` | Committed analysis + charts (not runtime) |
| `artifacts/` | Local evaluate / infer dumps — **never commit** |
| `notebooks/` | Ad-hoc exploration; import the package |
| `docs/` | Engineering docs + model card + nested guides |
| `data/external/` | Lakehouse gold sync (CSVs gitignored) |

**Product vs scratch:** package + `app/` + committed `models/` + `results/` are product. `artifacts/`, generated `data/raw/users.csv`, Optuna DBs, and `__pycache__` are scratch.

**Charts:** `artifacts/` is what evaluate writes at runtime. `results/plots/` is the committed pack. Refresh with `make docs-results`.

More: [GETTING_STARTED.md](GETTING_STARTED.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [../results/README.md](../results/README.md)
