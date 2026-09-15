# Retention Radar

FOSS **human-in-the-loop** churn ranking for a fictional AI platform. The model ranks quiet fade-out risk and hands a flight checklist to a human — it does **not** auto-cancel anyone.

Synthetic data only — no real PII. Seed **42**. CPU-fast. Scores go to a **human** (`auto_action: none`).

**GitHub:** [santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

> **Seed 42, one record:** validate → 22 features → raw **0.043** → calibrated **0.017** → band **low** → SHAP → HITL **monitor** (`auto_action: none`). Honest ladder: LogReg **0.872** / RF **0.868** / XGB **0.870** / LightGBM **0.865**; Brier **0.138 → 0.106**.

## Repo map (authoritative split)

| Repo | Role |
|------|------|
| **This repo (`retention-radar`)** | Final use-case **source code + benchmarks + results analysis** |
| [xgboost-ai-churn](https://github.com/santoshshinde2012/xgboost-ai-churn) | **Articles** (+ research supporting Medium) — not the code home |
| [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) | **Data foundation / SoR** (SILO · N=5000 gold) → sync into `data/external/` |

Dual-world notes: [docs/data-foundation-lakehouse.md](docs/data-foundation-lakehouse.md). Published ladder stays on the synthetic generator.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

Cite **`models/metrics.json`**. Narrative: [results/BENCHMARKS.md](results/BENCHMARKS.md) · Santosh: [results/SANTOSH_ANALYSIS.md](results/SANTOSH_ANALYSIS.md).  
Streamlit: `streamlit run app/streamlit_app.py`. Tests: `CHURN_DATA_SOURCE=synthetic pytest -q`.

### Dual path: synthetic vs lakehouse

| Path | How | Notes |
|------|-----|-------|
| **Synthetic (default / CI)** | `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh` | Seed-42 teaching ladder; committed `models/` bundle |
| **Lakehouse gold** | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` | Overwrites `models/`; restore with `git checkout -- models/ docs/MODEL_CARD.md docs/data-dictionary.md` |
| **Auto** | `CHURN_DATA_SOURCE=auto` (default in code) | Prefers `data/external/` when present |

Sync only:

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
| `results/` | **Benchmarks + Santosh analysis + committed plots** |
| `artifacts/` | Runtime plots + Santosh packet (gitignored) |
| `docs/` | Architecture, model card, dictionary, getting started |
| `docs/ARCHITECTURE.md` | SOLID package map |
| `docs/MODEL_CARD.md` | Metrics + intended use |

Full map: [docs/FOLDER_STRUCTURE.md](docs/FOLDER_STRUCTURE.md).

## Docs

- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) — 10-minute path
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — train/serve boundary + SOLID
- [docs/MODEL_CARD.md](docs/MODEL_CARD.md) — holdout metrics (from `models/metrics.json`)
- [results/BENCHMARKS.md](results/BENCHMARKS.md) — ladder, latency, Brier, τ
- [results/SANTOSH_ANALYSIS.md](results/SANTOSH_ANALYSIS.md) — single-record outcome
- [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md)
- [docs/e2e-free-platforms.md](docs/e2e-free-platforms.md)
- [docs/data-foundation-lakehouse.md](docs/data-foundation-lakehouse.md)
- [CONTRIBUTING.md](CONTRIBUTING.md)

## Makefile helpers

```bash
make setup          # venv + pip
make run            # synthetic run_all
make run-lakehouse  # lakehouse E2E (needs sibling lakehouse checkout)
make test           # CHURN_DATA_SOURCE=synthetic pytest -q
make infer          # Santosh decision packet
make ui             # Streamlit
make docs-results   # copy artifact PNGs → results/plots/
```

## License

MIT © Santosh Shinde — see [LICENSE](LICENSE).
