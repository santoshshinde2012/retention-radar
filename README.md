# Retention Radar

Human-in-the-loop churn ranking for a fictional AI platform.

It ranks quiet fade-out risk and returns a checklist for a human. It does **not** auto-cancel anyone.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](runtime.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live demo](https://img.shields.io/badge/live%20demo-TBD-lightgrey.svg)](docs/guides/e2e-free-platforms.md)

**Repo:** [santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

---

## Start here

Teaching trilogy hub (public FOSS only):

1. Clone **[retention-radar](https://github.com/santoshshinde2012/retention-radar)** + **[local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse)**
2. One synthetic command: `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh`
3. Optional lakehouse: `./scripts/run_lakehouse_e2e.sh ../local-data-lakehouse` (writes under `artifacts/lakehouse_run/` only — committed `models/` untouched)
4. **Live demo:** TBD — Streamlit Community Cloud / HF Space

Full map: [docs/guides/START_HERE.md](docs/guides/START_HERE.md). Do **not** clone any private article workspace.

## What this is

| This project | Not this project |
|--------------|------------------|
| FOSS teaching path: train → evaluate → serve | Production CRM or billing |
| Synthetic users (seed **42**), no real PII | ROI or fairness claims |
| HITL only (`auto_action: none`) | Auto account cancellation |
| Public **code + benchmarks + results** | Medium article home (internal) |

**One-record example (seed 42):** raw **0.043** → calibrated **0.017** → band **low** → HITL **monitor**.

**Honest ladder (test AUC):** LogReg / CatBoost **0.872** · Optuna XGB **0.870** · RF **0.868** · LightGBM **0.865**. Serving hero = **calibrated XGBoost**. Full tables: [`models/metrics.json`](models/metrics.json) · [results/BENCHMARKS.md](results/BENCHMARKS.md).

---

## Related repos

| Repo | Role |
|------|------|
| **This repo** | Public code, benchmarks, and results |
| [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) | Data foundation (SILO · N=5000 gold) |
| Articles | Written in a separate **internal** workspace (not a public clone target) |
| [medallion-write-back-loop](https://github.com/santoshshinde2012/medallion-write-back-loop) | See also — teaching write-back loop (not wired here) |
| [churn-vs-risk-poc](https://github.com/santoshshinde2012/churn-vs-risk-poc) | See also — churn vs risk POC (not wired here) |

---

## Requirements

- Python **3.11+**
- Linux, macOS, or Windows (WSL on Windows)
- CPU only
- **macOS:** `brew install libomp` if XGBoost or LightGBM fail to load

---

## Install

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Or: `make setup`

Optional env template: [`.env.example`](.env.example)

---

## Quick start

```bash
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Then:

| Step | Command / file |
|------|----------------|
| Cite metrics | [`models/metrics.json`](models/metrics.json) |
| Read benchmarks | [results/BENCHMARKS.md](results/BENCHMARKS.md) |
| Score Santosh | `make infer` |
| Open UI | `make ui` (committed models only — no fit on load) |
| Live demo | TBD — Streamlit Community Cloud / HF Space |
| Run tests | `make test` |

Faster smoke:

```bash
N_USERS=800 N_OPTUNA_TRIALS=5 CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

### Make targets

| Target | What it does |
|--------|----------------|
| `make setup` | Create venv and install |
| `make run` | Synthetic full pipeline |
| `make test` | Pytest (synthetic) |
| `make infer` | Santosh decision packet |
| `make ui` | Streamlit UI |
| `make run-lakehouse` | Lakehouse E2E (needs lakehouse checkout) |

CLI (after install):

```bash
python -m retention_radar.cli.train
python -m retention_radar.cli.infer --user santosh
```

---

## Data paths

| Path | How | Notes |
|------|-----|-------|
| **Synthetic** (CI / published numbers) | `CHURN_DATA_SOURCE=synthetic` | Seed-42 ladder; committed `models/` |
| **Lakehouse gold** | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` | Writes under `artifacts/lakehouse_run/` only (`RETENTION_RADAR_ARTIFACT_DIR`) |
| **Sync only** | `./scripts/sync_lakehouse_exports.sh …/data/export` | No retrain |

> **Isolation:** Lakehouse E2E sets `RETENTION_RADAR_ARTIFACT_DIR=artifacts/lakehouse_run` so train/eval/docs_gen never dirty committed `models/` or published `docs/MODEL_CARD.md`. Dual-world cite: [`results/lakehouse-e2e-summary.json`](results/lakehouse-e2e-summary.json).

Details: [docs/data/data-foundation-lakehouse.md](docs/data/data-foundation-lakehouse.md)

---

## Project structure

```text
retention-radar/
├── src/retention_radar/     # Package + CLI
├── app/                     # Streamlit UI
├── scripts/                 # run_all, lakehouse sync
├── configs/schemas/         # Serve payload schema
├── data/                    # raw · interim · processed · external
├── models/                  # Seed-42 serve bundle + metrics
├── results/                 # Benchmarks, Santosh analysis, plots
├── artifacts/               # Runtime output (gitignored)
├── notebooks/               # Exploration only
├── docs/                    # Guides, architecture, model card
└── tests/
```

Full map: [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md)

---

## Docs

| Doc | Purpose |
|-----|---------|
| [results/BENCHMARKS.md](results/BENCHMARKS.md) | Ladder, calibration, latency |
| [results/SANTOSH_ANALYSIS.md](results/SANTOSH_ANALYSIS.md) | Single-record outcome |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use + metrics |
| [docs/guides/START_HERE.md](docs/guides/START_HERE.md) | Trilogy hub: clone two repos → one command |
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Short walkthrough |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Train ≠ serve design |
| [docs/guides/ALGORITHM_LANDSCAPE.md](docs/guides/ALGORITHM_LANDSCAPE.md) | What we use vs defer |

---

## License

MIT © Santosh Shinde — see [LICENSE](LICENSE).
