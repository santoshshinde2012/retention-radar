# Results

Analysis home for the seed-42 reference run.

| Path | Role |
|------|------|
| [`BENCHMARKS.md`](BENCHMARKS.md) | Ladder, calibration, operating point, policy on test, holdout size, latency |
| [`WORKED_EXAMPLES.md`](WORKED_EXAMPLES.md) | Maya and Arjun at T-7 |
| [`analysis.json`](analysis.json) | Bootstrap of the ladder, isotonic vs Platt, deciles/quintiles, holdout sizes (`python -m retention_radar.cli.analysis`) |
| [`plots/`](plots/) | Committed ROC / PR / calibration / confusion / threshold charts |
| [`maya_decision_packet.sample.json`](maya_decision_packet.sample.json) | Sample decision packet |
| [`lakehouse-e2e-summary.json`](lakehouse-e2e-summary.json) | Optional lakehouse-gold run summary (not the published ladder) |

**Source of truth for numbers:** [`../models/metrics.json`](../models/metrics.json).
Runtime dumps (gitignored) land in `artifacts/`; `make docs-results` copies plots here.
