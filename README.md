# Retention Radar

Human-in-the-loop churn ranking for a fictional AI platform — ranks quiet fade-out risk and returns a flight checklist for a human. It does **not** auto-cancel anyone.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](runtime.txt)
[![pytest](https://img.shields.io/badge/pytest-21%20pass-brightgreen.svg)](#quick-start)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**GitHub:** [santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

## What this is / What this is not

| Is | Is not |
|----|--------|
| FOSS teaching codebase for train → evaluate → serve | Production CRM or billing integration |
| Synthetic users only (seed **42**); no real PII | A claim of production ROI or fairness audit |
| HITL policy (`auto_action: none`) | Automated account cancellation |
| Source of truth for **code + benchmarks + results analysis** | Article / Medium home (authored separately, internal) |
| Cookiecutter-data-science + src-layout conventions | Airflow / DVC / MLflow / FastAPI stack |

**Seed 42 reference (one record):** validate → 22 features → raw **0.043** → calibrated **0.017** → band **low** → SHAP → HITL **monitor** (`auto_action: none`). Honest ladder (test AUC from [`models/metrics.json`](models/metrics.json)): LogReg **0.872** / CatBoost **0.872** / XGB Optuna **0.870** / RF **0.868** / LightGBM **0.865**; Brier **0.138 → 0.106**. Serving hero = calibrated XGB.

## Repository map

| Repo | Role |
|------|------|
| **This repo (`retention-radar`)** | **Public** use-case **source code + benchmarks + results analysis** |
| Articles (authored separately) | Medium series is written in an **internal** workspace — not a public reader destination; **this repo is the public code home** |
| [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) | **Data foundation / SoR** (SILO · N=5000 gold) → sync into `data/external/` |

Dual-world notes: [docs/data/data-foundation-lakehouse.md](docs/data/data-foundation-lakehouse.md). Published ladder stays on the synthetic generator.

## Requirements

- **Python 3.11+** (`runtime.txt` pins `python-3.11`)
- Linux, macOS, or Windows (WSL recommended on Windows)
- CPU-only; no GPU required
- **macOS:** `brew install libomp` if XGBoost/LightGBM fail to load OpenMP (`libomp`); CatBoost is pip-only on most Macs

## Installation

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
# If you skip editable install, set:
# export PYTHONPATH="$(pwd)/src"
```

Optional: `make setup` (venv + `pip install -r requirements.txt` + `pip install -e .`).  
Env template: [`.env.example`](.env.example) (`CHURN_DATA_SOURCE`, `N_USERS`, `N_OPTUNA_TRIALS`).

## Quick start

```bash
chmod +x scripts/run_all.sh
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

1. Cite holdout metrics from **`models/metrics.json`**.
2. Read the narrative in [results/BENCHMARKS.md](results/BENCHMARKS.md) and [results/SANTOSH_ANALYSIS.md](results/SANTOSH_ANALYSIS.md).
3. Launch the UI: `streamlit run app/streamlit_app.py` (or `make ui`).
4. Run tests: `CHURN_DATA_SOURCE=synthetic PYTHONPATH=src pytest -q` (or `make test`).

Faster smoke: `N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh`.

Score the Santosh record: `make infer` (writes `artifacts/santosh_decision_packet.json`).

### Make targets

```bash
make setup          # venv + pip install -r requirements.txt + pip install -e .
make run            # synthetic run_all
make run-lakehouse  # lakehouse E2E (needs sibling lakehouse checkout)
make test           # CHURN_DATA_SOURCE=synthetic PYTHONPATH=src pytest -q
make infer          # Santosh decision packet
make ui             # Streamlit
make docs-results   # copy artifact PNGs → results/plots/
```

## Dual data path (synthetic vs lakehouse)

| Path | How | Notes |
|------|-----|-------|
| **Synthetic (default / CI)** | `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh` | Seed-42 teaching ladder; committed `models/` bundle |
| **Lakehouse gold** | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` | **Overwrites** `models/` and regenerated docs |
| **Auto** | `CHURN_DATA_SOURCE=auto` (default in code) | Prefers `data/external/` when present |

**Warning:** Lakehouse E2E overwrites published seed-42 artifacts. Restore before committing:

```bash
git checkout -- models/ docs/MODEL_CARD.md docs/data/data-dictionary.md
```

Sync exports only (no retrain):

```bash
./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export
```

## Project structure

Production-style layout (cookiecutter-data-science data layers + src-layout package). Teaching FOSS scope — no Airflow/DVC/MLflow/FastAPI.

```text
retention-radar/
├── README.md, LICENSE, CHANGELOG.md, CONTRIBUTING.md, Makefile
├── pyproject.toml, requirements.txt, requirements.lock, runtime.txt
├── .env.example, .gitignore
├── .github/workflows/ci.yml
├── configs/                         # contracts & config home
│   ├── README.md
│   └── schemas/
│       └── user_record.schema.json
├── src/retention_radar/             # ONLY Python package (src-layout)
│   ├── config.py, protocols.py, docs_gen.py
│   ├── cli/                         # train, evaluate, infer, single_record, …
│   ├── data/, features/, training/, evaluation/, serving/
├── app/                             # Streamlit (serve-only)
├── scripts/                         # shell orchestration
├── data/
│   ├── README.md                    # raw / interim / processed / external
│   ├── raw/                         # Santosh JSON; users.csv (gitignored)
│   ├── interim/                     # CDS parity (empty)
│   ├── processed/                   # CDS parity (empty; features in-memory)
│   └── external/                    # lakehouse gold sync
├── models/                          # seed-42 serve bundle + metrics.json
├── results/                         # ≡ reports/ in CDS templates (name kept for article URLs)
│   ├── README.md, BENCHMARKS.md, SANTOSH_ANALYSIS.md
│   └── plots/                       # ≡ reports/figures
├── artifacts/                       # runtime dumps (gitignored)
├── notebooks/                       # exploration only; import package
├── docs/                            # architecture, model card, guides/, data/, case-study/
└── tests/
```

Full map: [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md).

## Results & benchmarks

| Doc | Contents |
|-----|----------|
| [results/BENCHMARKS.md](results/BENCHMARKS.md) | Honest ladder, calibration, latency, τ — cite [`models/metrics.json`](models/metrics.json) |
| [results/SANTOSH_ANALYSIS.md](results/SANTOSH_ANALYSIS.md) | Single-record outcome (raw / calibrated / HITL) |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use + metrics from `models/metrics.json` |
| [docs/guides/ALGORITHM_LANDSCAPE.md](docs/guides/ALGORITHM_LANDSCAPE.md) | Ladder IN vs deferred (TabPFN, survival, conformal, …) |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | 10-minute path |

## Development

```bash
make setup && make test
python -m retention_radar.cli.docs_gen   # refresh model card / data dictionary after a train
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep PRs on the synthetic path so CI metrics stay comparable.

## Related projects

- Articles are authored separately (**internal**); this repo is the public code home for Medium readers
- [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) — SILO gold / data SoR (foundation)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — SOLID train/serve boundary
- [docs/guides/BEST_PRACTICES.md](docs/guides/BEST_PRACTICES.md)
- [docs/guides/e2e-free-platforms.md](docs/guides/e2e-free-platforms.md)

## License

MIT © Santosh Shinde — see [LICENSE](LICENSE).
