# Retention Radar

FOSS **human-in-the-loop** churn ranking for a fictional AI platform. The model ranks quiet fade-out risk and hands a flight checklist to a human — it does **not** auto-cancel anyone.

Synthetic data only — no real PII. Seed **42**. CPU-fast. Scores go to a **human** (`auto_action: none`).

**GitHub:** [santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

> **Seed 42, one record:** validate → 22 features → raw **0.043** → calibrated **0.017** → band **low** → SHAP → HITL **monitor** (`auto_action: none`). Honest ladder: LogReg **0.872** / RF **0.868** / XGB **0.870** / LightGBM **0.865**; Brier **0.138 → 0.106**.

**Data foundation / SoR:** [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) (SILO · N=5000 gold) → sync into `data/external/` · dual-world with the seed-42 synthetic ladder (see [docs/data-foundation-lakehouse.md](docs/data-foundation-lakehouse.md)).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

Cite **`models/metrics.json`**. Streamlit: `streamlit run app/streamlit_app.py`. Tests: `CHURN_DATA_SOURCE=synthetic pytest -q`.

### Dual path: synthetic vs lakehouse

| Path | How | Notes |
|------|-----|-------|
| **Synthetic (default / CI)** | `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh` | Seed-42 teaching ladder; committed `models/` bundle |
| **Lakehouse gold** | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` | Overwrites `models/`; restore with `git checkout -- models/ MODEL_CARD.md docs/data-dictionary.md` |
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
| `scripts/run_all.sh` | End-to-end pipeline |
| `scripts/run_lakehouse_e2e.sh` | Lakehouse gold → train/infer |
| `data/raw/` | Santosh JSON (+ generated `users.csv`, gitignored) |
| `data/external/` | Lakehouse gold sync (CSVs gitignored) |
| `models/` | Seed-42 serve bundle (`joblib` + `metrics.json`) |
| `artifacts/` | Runtime plots + Santosh packet (gitignored) |
| `docs/` | Dictionary, checklist, lakehouse notes, result charts |
| `ARCHITECTURE.md` | SOLID package map |
| `MODEL_CARD.md` | Metrics + intended use |

## Docs

- [ARCHITECTURE.md](ARCHITECTURE.md) — train/serve boundary + SOLID
- [MODEL_CARD.md](MODEL_CARD.md) — holdout metrics (from `models/metrics.json`)
- [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md)
- [docs/e2e-free-platforms.md](docs/e2e-free-platforms.md) — venv, Actions, Streamlit Cloud, optional HF Spaces
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
make docs-results   # copy artifact PNGs → docs/results/
```

## License

MIT © Santosh Shinde — see [LICENSE](LICENSE).
