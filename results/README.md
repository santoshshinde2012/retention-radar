# Results

Analysis home for the seed-42 reference run, plus the lakehouse consume run. File names follow
[CONTRIBUTING.md](../CONTRIBUTING.md#file-names).

| Path | Role |
|------|------|
| [`benchmarks.md`](benchmarks.md) | Ladder, calibration, operating point, policy on test, holdout size, latency |
| [`worked-examples.md`](worked-examples.md) | Santosh and Arjun at T-7 |
| [`analysis.json`](analysis.json) | Bootstrap of the ladder, isotonic vs Platt, deciles/quintiles, holdout sizes (`python -m retention_radar.cli.analysis`) |
| [`plots/`](plots/) | Committed ROC / PR / calibration / confusion / threshold charts |
| [`santosh_decision_packet.sample.json`](santosh_decision_packet.sample.json) | Sample decision packet |
| [`lakehouse-consume-e2e.md`](lakehouse-consume-e2e.md) | Radar scoring the lakehouse export in the 2026-10-03 run from empty volumes (no retrain): steps, rows, actions, hashes, tests |
| [`lakehouse_e2e_summary.json`](lakehouse_e2e_summary.json) | Optional lakehouse-gold run summary (not the published ladder) |

**Source of truth for numbers:** [`../models/metrics.json`](../models/metrics.json).
Runtime dumps (gitignored) land in `artifacts/`; `make docs-results` copies plots here.
