# Results — benchmarks & analysis

This folder is the **analysis home** for Retention Radar (seed **42** synthetic reference run).

> **Naming note:** In cookiecutter-data-science and many production templates this role is called `reports/` (with figures under `reports/figures/`). We keep the name **`results/`** (and `results/plots/`) so existing links into `results/` stay stable.

| Path | Role |
|------|------|
| [`benchmarks.md`](benchmarks.md) | Honest ladder (incl. CatBoost), latency, Brier, τ — narrative |
| [`example-account-analysis.md`](example-account-analysis.md) | Single-record outcome for Santosh Shinde |
| [`plots/`](plots/) | Committed ROC / PR / calibration / confusion / threshold charts (≡ `reports/figures`) |
| [`example_decision_packet.json`](example_decision_packet.json) | Sample HITL packet |
| [`lakehouse_e2e_summary.json`](lakehouse_e2e_summary.json) | Optional dual-world lakehouse summary (not the published ladder) |

**Source of truth for numbers:** [`../models/metrics.json`](../models/metrics.json).  
**Runtime dumps** (gitignored): `artifacts/` after `./scripts/run_all.sh`.  
Refresh committed plots: `make docs-results` (copies from `artifacts/` → `results/plots/`).

**Data foundation / SoR:** [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse)
