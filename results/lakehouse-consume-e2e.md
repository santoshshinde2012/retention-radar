# Lakehouse consume run (end to end)

Retention Radar scoring the v2 export of
[local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) with the committed
seed-42 bundle. No retraining: this is the consume path only (sync → ingest → batch score). For a retrain
on lakehouse gold, see [`lakehouse_e2e_summary.json`](lakehouse_e2e_summary.json) (a separate Linux run,
not the published ladder; CI `e2e-local` re-runs it on every pull request and compares).

**This run:** 2026-10-03 (IST), on a MacBook Pro (Apple M1 Pro, macOS 26.6.2, Docker Desktop 29.8.1), as part
of the lakehouse's full end-to-end run from empty volumes (`make purge` first). All numbers below are from
that run. The lakehouse write-up of the same run: its
[RESULTS.md](https://github.com/santoshshinde2012/local-data-lakehouse/blob/chore/sample-customer-santosh/RESULTS.md).

| Repo | Branch | Commit |
|---|---|---|
| local-data-lakehouse | `chore/sample-customer-santosh` ([PR #16](https://github.com/santoshshinde2012/local-data-lakehouse/pull/16)) | `2fcb92f` for the pipeline steps; `6c57111` for the second consume (README only in between) |
| retention-radar | `chore/sample-customer-santosh` ([PR #24](https://github.com/santoshshinde2012/retention-radar/pull/24)) | `98df572` for the first consume, `07d8205` for the second (README, `make setup` and docs only in between) |

`pipelines/radar_consume.sh` in the lakehouse clones radar at `RADAR_REF`, builds a Python 3.12 venv,
copies the export into `data/external/`, then runs `cli.ingest` with `CHURN_DATA_SOURCE=lakehouse` and
`cli.batch_score`.

## Steps and timings

| Step | Exit | Seconds |
|---|---|---|
| `make purge`, then `make up-light` and `make up-full` (empty volumes) | 0 | 1.1 + 8.0 + 11.6 |
| `make churn-sample` (8,001 subscriptions, seed 42) | 0 | 1.9 |
| `make churn-e2e` (Spark 4.1.3 + Iceberg 1.12.0 via Lakekeeper; writes `data/export/`) | 0 | 85.0 |
| `scripts/check_churn_export.py --strict` | 0 | 0.7 |
| `make churn-parity` (Spark SQL vs pandas, 8,001 × 27 cells) | 0 | 15.3 |
| `radar_consume.sh` on the Spark export, radar `98df572` (clone + venv + score) | 0 | 72.4 |
| radar pytest on that checkout | 0 | 38.2 (101 passed) |
| `make churn-gold-local` (pandas twin export) | 0 | 3.9 |
| `radar_consume.sh` on the pandas export, radar `07d8205` | 0 | 73.2 |
| radar pytest on that checkout | 0 | 42.3 (101 passed) |

Both consumes loaded 7,387 rows (churn rate 0.074) and produced the same action counts.

## The export it read (v2 contract)

| File | Content | sha256 |
|---|---|---|
| `churn_user_features.csv` | 7,387 renewals, 25 columns (22 T-7 features + ids + `churned`) | `742f9028e4216d32ad6029007ee3b57414d1772344005ea907a5e5cf80eba18d` |
| `hero_inference_record.json` | `sub_santosh`, 24 keys, no label | `0db4f2de0cdec5e15d31ef85ff2de814525f854de734b28414ac0bcaf7abcfe5` |
| `churn_renewals_audit.csv` | Lakehouse-side audit (radar does not read it; has a `built_at` time, so its hash changes every run) | `f081a52a725ad699e25b737e0297ceced57b3d7bfb471cb8171d980fd9593656` |

These are the hashes at the end of the run. The features and hero files have the same sha256 as in
the previous runs, so the export is deterministic. The hero is `holdout` in radar: `sub_santosh` hashes
into the 10% control group (the limit reset is what he would have got).

## Sync, ingest, batch score (second consume)

```text
$ env RADAR_REF=chore/sample-customer-santosh ./pipelines/radar_consume.sh data/export /tmp/radar-e2e
retention-radar ref: chore/sample-customer-santosh
retention-radar commit: 07d8205
XGBoost/LightGBM cannot load libomp; using scikit-learn's bundled copy (or: brew install libomp)
Synced -> /tmp/radar-e2e/data/external/churn_user_features.csv
Synced -> /tmp/radar-e2e/data/external/hero_inference_record.json
Loaded 7387 rows from /tmp/radar-e2e/data/external/churn_user_features.csv
CHURN_DATA_SOURCE=lakehouse
Churn rate: 0.074
... (head of the frame trimmed)
Scored 7387 rows → /tmp/radar-e2e/scores.csv (queue order: rank 1 = highest risk)
action
no_action               6499
cancel_flow_discount     410
limit_reset              276
pause_offer              111
holdout                   87
personal_email             4
auto_action unique: ['none']
==> radar scored 7387 rows -> /tmp/radar-e2e/scores.csv
```

801 of 7,387 renewals (10.8 %) get a playbook action, 87 (1.2 %) are held out, and 6,499 get none.
`auto_action` is `none` for every row: radar recommends, a person decides.

## Tests on that checkout

```text
$ cd /tmp/radar-e2e && git log -1 --oneline && CHURN_DATA_SOURCE=synthetic PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
07d8205 docs: professional README; make setup fixes OpenMP on macOS without Homebrew libomp
... (warnings summary trimmed: deprecation notices from fastapi, shap, sklearn)
101 passed, 7 warnings in 40.93s
```

## Committed model bundle (`models/metrics.json`, synthetic seed-42 test split)

These describe the bundle that scored the export. They are not measured on lakehouse data.

| Metric | Value |
|---|---|
| ROC AUC (calibrated XGBoost, sigmoid) | 0.757 |
| Average precision | 0.270 |
| Brier score (calibrated) | 0.079 |
| Operating threshold τ (validation, best F1) | 0.16 |
| Slice `pro` at τ (n 1,163) | precision 0.269, recall 0.488 |
| Policy on test (n 1,466) | contacted 15.8 %; expected value $837.48 (assumed playbook effects) |

## CI

The CI runs for this branch are listed in [PR #24](https://github.com/santoshshinde2012/retention-radar/pull/24)
(jobs `test` and `e2e-local`; `e2e-local` clones the lakehouse branch of the same name and re-runs the
lakehouse E2E on Linux).

## Known gaps

- **Drift falls back to SMD.** The committed `models/feature_stats.json` has no `psi_bins`, so
  `serving/drift.py` uses the standardised mean difference instead of PSI. Retrain the bundle to get
  PSI bins. The consume run above does not compute drift.
- **Synthetic metrics only.** The metrics table is the seed-42 synthetic bundle. Radar has no labelled
  outcome on the lakehouse export to measure it against.
- **Expected value is assumed.** The $837.48 uses assumed playbook effects; a holdout has to measure them.
