# Lakehouse consume run (end to end)

Radar `main` scoring the v2 export of
[local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) `main` with the
committed seed-42 bundle. No retraining: this is the consume path only (sync → ingest → batch score).
For a retrain on lakehouse gold, see [`lakehouse_e2e_summary.json`](lakehouse_e2e_summary.json)
(a separate run, 2026-09-30, not the published ladder).

Captured 2026-10-03 00:49–00:53 IST on a MacBook Pro (Apple M1 Pro, macOS 26.6.2), Docker Compose,
starting from the lakehouse's existing volumes.

| Repo | Ref | Commit |
|---|---|---|
| local-data-lakehouse | `main` | `08bb274` (squash merge of [#13](https://github.com/santoshshinde2012/local-data-lakehouse/pull/13)) |
| retention-radar | `main` | `4a947be` (code identical to `7e3bec8`, the merge of [#21](https://github.com/santoshshinde2012/retention-radar/pull/21); `4a947be` only adds docs) |

`pipelines/radar_consume.sh` in the lakehouse clones radar `main` (override with `RADAR_REF`),
builds a Python 3.12 venv, copies the export into `data/external/`, then runs `cli.ingest` with
`CHURN_DATA_SOURCE=lakehouse` and `cli.batch_score`.

## Steps and timings

| Step | Exit | Seconds |
|---|---|---|
| `make up-full` | 0 | 17.9 |
| `make churn-sample` | 0 | 2.2 |
| `make churn-e2e` (Spark 4.1.3 + Iceberg 1.12.0 via Lakekeeper; writes `data/export/`) | 0 | 109.3 |
| `scripts/check_churn_export.py --strict` | 0 | n/a |
| `pipelines/radar_consume.sh data/export /tmp/radar-main-e2e` (clone + venv + score) | 0 | 80.2 |
| radar pytest on that checkout | 0 | 38.3 |
| `make down` (volumes kept) | 0 | 4.1 |

## The export it read (v2 contract)

```text
Churn export contract OK (7387 renewals, 25 cols) → .../local-data-lakehouse/data/export
```

| File | Content | sha256 |
|---|---|---|
| `churn_user_features.csv` | 7,387 renewals, 25 columns (22 T-7 features + ids + `churned`) | `742f9028e4216d32ad6029007ee3b57414d1772344005ea907a5e5cf80eba18d` |
| `hero_inference_record.json` | `sub_maya`, 24 keys, no label | `49abdddc526827375b2f8d442fda6359cc2532353d528207545b204e3eb10878` |
| `churn_renewals_audit.csv` | Lakehouse-side audit (radar does not read it) | `e07dcfceb32c00a4b4ab34615ef7a9a40ea3ba165fa9468a6ab3db30cd5c12f2` |

The features and hero hashes match the earlier Spark export at lakehouse `7f5fc43`, so the export is
deterministic across these commits.

## Sync, ingest, batch score

```text
$ ./pipelines/radar_consume.sh data/export /tmp/radar-main-e2e
retention-radar ref: main
retention-radar commit: 4a947be
XGBoost/LightGBM cannot load libomp; using scikit-learn's bundled copy (or: brew install libomp)
Synced -> /private/tmp/radar-main-e2e/data/external/churn_user_features.csv
Synced -> /private/tmp/radar-main-e2e/data/external/hero_inference_record.json
Loaded 7387 rows from /private/tmp/radar-main-e2e/data/external/churn_user_features.csv
CHURN_DATA_SOURCE=lakehouse
Churn rate: 0.074
... (head of the frame trimmed)
Scored 7387 rows → /tmp/radar-main-e2e/scores.csv (queue order: rank 1 = highest risk)
action
no_action               6499
cancel_flow_discount     410
limit_reset              276
pause_offer              111
holdout                   87
personal_email             4
auto_action unique: ['none']
==> radar scored 7387 rows -> /tmp/radar-main-e2e/scores.csv
```

801 of 7,387 renewals (10.8 %) get a playbook action, 87 (1.2 %) are held out, and 6,499 get none.
`auto_action` is `none` for every row: radar recommends, a person decides.

## Tests on that checkout

```text
$ cd /tmp/radar-main-e2e && CHURN_DATA_SOURCE=synthetic PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
... (warnings summary trimmed: deprecation notices from fastapi, shap, sklearn)
96 passed, 7 warnings in 36.95s
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

| Where | Run | Result |
|---|---|---|
| radar `main` at `4a947be` | [37051673526](https://github.com/santoshshinde2012/retention-radar/actions/runs/37051673526) | `test` and `e2e-local` green |
| radar `main` at `7e3bec8` | [37029880425](https://github.com/santoshshinde2012/retention-radar/actions/runs/37029880425) | `test` and `e2e-local` green |
| radar PR #21 head `65cab25` | [37019854227](https://github.com/santoshshinde2012/retention-radar/actions/runs/37019854227) | `test` and `e2e-local` green |
| lakehouse `main` at `08bb274` | [37046795038](https://github.com/santoshshinde2012/local-data-lakehouse/actions/runs/37046795038), [`t0-unit` job](https://github.com/santoshshinde2012/local-data-lakehouse/actions/runs/37046795038/job/110970109891) | all green; step "Retention Radar consumes the export": radar `7e3bec8`, 7,387 rows scored |

## Known gaps

- **Drift falls back to SMD.** The committed `models/feature_stats.json` has no `psi_bins`, so
  `serving/drift.py` uses the standardised mean difference instead of PSI. Retrain the bundle to get
  PSI bins. The consume run above does not compute drift.
- **Synthetic metrics only.** The metrics table is the seed-42 synthetic bundle. Radar has no labelled
  outcome on the lakehouse export to measure it against.
- **Expected value is assumed.** The $837.48 uses assumed playbook effects; a holdout has to measure them.
