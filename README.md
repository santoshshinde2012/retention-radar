# Retention Radar

Human-in-the-loop churn ranking for a fictional AI platform. Ranks quiet fade-out risk and returns a flight checklist for a human — it does **not** auto-cancel anyone.

**GitHub:** [santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

## What this is / What this is not

| Is | Is not |
|----|--------|
| FOSS teaching codebase for train → evaluate → serve | Production CRM or billing integration |
| Synthetic users only (seed **42**); no real PII | A claim of production ROI or fairness audit |
| HITL policy (`auto_action: none`) | Automated account cancellation |
| Source of truth for **code + benchmarks + results analysis** | Article / Medium home (see related projects) |

**Seed 42 reference (one record):** validate → 22 features → raw **0.043** → calibrated **0.017** → band **low** → SHAP → HITL **monitor** (`auto_action: none`). Honest ladder (test AUC): LogReg **0.872** / RF **0.868** / XGB **0.870** / LightGBM **0.865**; Brier **0.138 → 0.106**.

## Repository map

| Repo | Role |
|------|------|
| **This repo (`retention-radar`)** | **Public** use-case **source code + benchmarks + results analysis** |
| Articles (authored separately) | Medium series is written in an **internal** workspace — not a public reader destination; **this repo is the public code home** |
| [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) | **Data foundation / SoR** (SILO · N=5000 gold) → sync into `data/external/` |

Dual-world notes: [docs/data-foundation-lakehouse.md](docs/data-foundation-lakehouse.md). Published ladder stays on the synthetic generator.

## Features

- End-to-end pipeline: generate/load → features → train (LogReg, RF, XGBoost, LightGBM, Optuna) → calibrate → evaluate → serve
- Committed seed-42 model bundle under `models/` for offline serve and Cloud demos
- Single-record decision packet (Santosh hero JSON) with SHAP drivers and HITL action
- Streamlit serve-only UI
- Dual data path: synthetic (CI / published metrics) or lakehouse gold sync
- Benchmarks and analysis under `results/`

## Requirements

- **Python 3.11+** (`runtime.txt` pins `python-3.11`)
- Linux, macOS, or Windows (WSL recommended on Windows)
- CPU-only; no GPU required

## Installation

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

Optional: `make setup` creates the venv and installs requirements.

## Quick start

```bash
chmod +x scripts/run_all.sh
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

1. Cite holdout metrics from **`models/metrics.json`**.
2. Read the narrative in [results/BENCHMARKS.md](results/BENCHMARKS.md) and [results/SANTOSH_ANALYSIS.md](results/SANTOSH_ANALYSIS.md).
3. Launch the UI: `streamlit run app/streamlit_app.py` (or `make ui`).
4. Run tests: `CHURN_DATA_SOURCE=synthetic pytest -q` (or `make test`).

Faster smoke: `N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh`.

Score the Santosh record: `make infer` (writes `artifacts/santosh_decision_packet.json`).

## Dual data path (synthetic vs lakehouse)

| Path | How | Notes |
|------|-----|-------|
| **Synthetic (default / CI)** | `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh` | Seed-42 teaching ladder; committed `models/` bundle |
| **Lakehouse gold** | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` | **Overwrites** `models/` and regenerated docs |
| **Auto** | `CHURN_DATA_SOURCE=auto` (default in code) | Prefers `data/external/` when present |

**Warning:** Lakehouse E2E overwrites published seed-42 artifacts. Restore before committing:

```bash
git checkout -- models/ docs/MODEL_CARD.md docs/data-dictionary.md
```

Sync exports only (no retrain):

```bash
./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export
```

## Project layout

| Path | Role |
|------|------|
| `src/retention_radar/` | Packages: data, features, training, evaluation, serving |
| `src/*.py` | Shims (`python -m src.train`, `src.infer`, …) |
| `app/` | Streamlit (serve-only) |
| `scripts/` | `run_all`, lakehouse sync / E2E |
| `data/raw/` | Santosh JSON (+ generated `users.csv`, gitignored) |
| `data/external/` | Lakehouse gold sync (CSVs gitignored) |
| `models/` | Seed-42 serve bundle (`joblib` + `metrics.json`) |
| `results/` | Benchmarks + Santosh analysis + committed plots |
| `artifacts/` | Runtime plots + Santosh packet (gitignored) |
| `docs/` | Architecture, model card, dictionary, getting started |

Full map: [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md).

## Results & benchmarks

| Doc | Contents |
|-----|----------|
| [results/BENCHMARKS.md](results/BENCHMARKS.md) | Honest ladder, calibration, latency, τ |
| [results/SANTOSH_ANALYSIS.md](results/SANTOSH_ANALYSIS.md) | Single-record outcome (raw / calibrated / HITL) |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use + metrics from `models/metrics.json` |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | 10-minute path |

## Development

```bash
make setup          # venv + pip
make run            # synthetic run_all
make run-lakehouse  # lakehouse E2E (needs sibling lakehouse checkout)
make test           # CHURN_DATA_SOURCE=synthetic pytest -q
make infer          # Santosh decision packet
make ui             # Streamlit
make docs-results   # copy artifact PNGs → results/plots/
```

Refresh model card / data dictionary after a train: `python -m src.docs_gen`.

See [CONTRIBUTING.md](CONTRIBUTING.md). Keep PRs on the synthetic path so CI metrics stay comparable.

## Related projects

- Articles are authored separately (**internal**); this repo is the public code home for Medium readers
- [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) — SILO gold / data SoR (foundation)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — SOLID train/serve boundary
- [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md)
- [docs/e2e-free-platforms.md](docs/e2e-free-platforms.md)

## License

MIT © Santosh Shinde — see [LICENSE](LICENSE).
