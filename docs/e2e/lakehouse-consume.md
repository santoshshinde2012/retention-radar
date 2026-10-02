# End to end with local-data-lakehouse

Radar `main` (the v2 reader, merged in
[#21](https://github.com/santoshshinde2012/retention-radar/pull/21)) consumes the v2 export of
[local-data-lakehouse PR #13](https://github.com/santoshshinde2012/local-data-lakehouse/pull/13)
(`feat/local-first-stack-2026`, Spark 4.1.3 + Iceberg 1.12.0 through the Lakekeeper REST catalog).
The lakehouse runs it with `pipelines/radar_consume.sh`, which clones radar, builds a Python 3.12 venv, syncs the export into `data/external/`, runs
`cli.ingest` with `CHURN_DATA_SOURCE=lakehouse` and `cli.batch_score` with the committed bundle.

The run below was captured before #21 merged, with `RADAR_REF=feat/local-first-stack-2026` at `65cab25`;
its tree is identical to `main` at `7e3bec8` (the squash merge of #21).

Captured 2026-10-02 (IST) on a MacBook Pro (Apple M1 Pro, macOS 26.6.2), from empty lakehouse volumes.
The lakehouse's full set of excerpts is in its
[docs/demo/README.md](https://github.com/santoshshinde2012/local-data-lakehouse/blob/feat/local-first-stack-2026/docs/demo/README.md).

## The export it read

| File | Rows / keys | sha256 (first 16) |
|---|---|---|
| `churn_user_features.csv` | 7,387 renewals routed to the model, 25 columns (22 T-7 features + ids + `churned`) | `742f9028e4216d32` |
| `hero_inference_record.json` | `sub_maya`, 24 keys (no label) | `49abdddc52682737` |

Written by `make churn-e2e` (Spark `04_export_features.py`) at lakehouse commit `7f5fc43`.

## Sync, ingest, batch score (exit 0, 66.6 s, including the clone and the venv)

```text
$ env RADAR_REF=feat/local-first-stack-2026 ./pipelines/radar_consume.sh data/export /tmp/radar-e2e
retention-radar ref: feat/local-first-stack-2026
retention-radar commit: 65cab25
XGBoost/LightGBM cannot load libomp; using scikit-learn's bundled copy (or: brew install libomp)
Synced -> /tmp/radar-e2e/data/external/churn_user_features.csv
Synced -> /tmp/radar-e2e/data/external/hero_inference_record.json
Loaded 7387 rows from /tmp/radar-e2e/data/external/churn_user_features.csv
CHURN_DATA_SOURCE=lakehouse
Churn rate: 0.074
  user_id    user_name plan_tier  renewals_completed  active_days_7d  active_days_28d  engagement_trend  last_active_days_ago  agent_requests_28d  allowance_used_pct  limit_hits_14d  cheap_model_share …
sub_00000 Ananya Singh       pro                   6               2                6            1.3333                     0                 146              0.2655               0                 0. …
sub_00001  Riley Singh       pro                  24               1                9            0.4444                     6                 306              0.5564               0                 0. …
sub_00002  Riley Patel       pro                   9               2               11            0.7273                     2                 491              0.8927               3                 0. …
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

## Tests on that checkout (exit 0, 40.8 s)

```text
$ zsh -c cd /tmp/radar-e2e && git log -1 --oneline && .venv/bin/python -m pip install -q pytest httpx && PYTHONPATH=src .venv/bin/python -m pytest -q -p no:cacheprovider
65cab25 models: slice metrics at tau in the committed metrics.json
........................................................................ [ 75%]
........................                                                 [100%]
... (pytest warnings summary trimmed: deprecation notices from fastapi, shap, sklearn)
96 passed, 7 warnings in 39.58s
```

## Committed model bundle (`models/metrics.json`)

| Metric (test split) | Value |
|---|---|
| ROC AUC (calibrated XGBoost, sigmoid) | 0.757 |
| Average precision | 0.270 |
| Brier score (calibrated) | 0.079 |
| Operating threshold τ (from validation, best F1) | 0.16 |
| Slice `pro` at τ | precision 0.269, recall 0.488 |
| Policy on test (n 1,466) | contacted 15.8 %; expected value $837.48 (assumed playbook effects) |

## CI

- This repo: `test` and `e2e-local` on the PR head `65cab25`
  ([PR #21](https://github.com/santoshshinde2012/retention-radar/pull/21)) and again on `main` after the merge.
- The lakehouse's `t0-unit` job runs the same consumer against its CI export ("Retention Radar consumes
  the export": 7,387 rows scored).
