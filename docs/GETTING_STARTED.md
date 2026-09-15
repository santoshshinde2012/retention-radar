# Getting started — 10-minute Retention Radar path

FOSS **human-in-the-loop** churn ranking for a fictional AI platform. Hero: **Santosh Shinde**. Synthetic data only (seed **42**).

**This repo** = source code + benchmarks + results analysis.  
**Articles:** [xgboost-ai-churn](https://github.com/santoshshinde2012/xgboost-ai-churn) · **Data SoR:** [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse)

## 1. Clone and venv (~2 min)

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)"
```

## 2. Run the full pipeline (~3–5 min CPU)

```bash
chmod +x scripts/run_all.sh
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

Faster smoke: `N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh`

Cite **`models/metrics.json`**. Analysis narrative: [`../results/BENCHMARKS.md`](../results/BENCHMARKS.md).

## 3. Score Santosh

```bash
python -m src.infer --user santosh
# or
make infer
```

Expect raw ≈ **0.043**, calibrated ≈ **0.017**, band **low**, HITL **monitor** (`auto_action: none`). See [`../results/SANTOSH_ANALYSIS.md`](../results/SANTOSH_ANALYSIS.md).

## 4. UI / tests

```bash
streamlit run app/streamlit_app.py
CHURN_DATA_SOURCE=synthetic pytest -q
```

## Dual path

| Path | Command |
|------|---------|
| Synthetic (default / CI) | `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh` |
| Lakehouse gold | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` |

After lakehouse E2E, restore committed models:  
`git checkout -- models/ docs/MODEL_CARD.md docs/data-dictionary.md`

More: [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [e2e-free-platforms.md](e2e-free-platforms.md)
