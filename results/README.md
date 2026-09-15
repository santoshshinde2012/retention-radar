# Results — benchmarks & analysis

This folder is the **analysis home** for Retention Radar (seed **42** synthetic reference run).

| Path | Role |
|------|------|
| [`BENCHMARKS.md`](BENCHMARKS.md) | Honest ladder table, latency, Brier, τ — narrative |
| [`SANTOSH_ANALYSIS.md`](SANTOSH_ANALYSIS.md) | Single-record outcome for Santosh Shinde |
| [`plots/`](plots/) | Committed ROC / PR / calibration / confusion / threshold charts |
| [`santosh_decision_packet.sample.json`](santosh_decision_packet.sample.json) | Sample HITL packet |
| [`lakehouse-e2e-summary.json`](lakehouse-e2e-summary.json) | Optional dual-world lakehouse summary (not the published ladder) |

**Source of truth for numbers:** [`../models/metrics.json`](../models/metrics.json).  
**Runtime dumps** (gitignored): `artifacts/` after `./scripts/run_all.sh`.  
Refresh committed plots: `make docs-results` (copies from `artifacts/` → `results/plots/`).

**Articles** (Medium series): authored separately (**internal**); this repo is the public code / results home.  
**Data foundation / SoR:** [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse)
