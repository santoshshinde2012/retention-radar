# Data foundation: local-data-lakehouse

The synthetic generator in this repo produces the T-7 renewal table directly.
[local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) builds
the same table the way a real team would: from raw billing and usage events, through
bronze and silver, into a gold feature table as of each subscriber's T-7 date.

## Required export (v2 contract)

This repo reads two files from the lakehouse export directory:

| File | Contents |
|------|----------|
| `churn_user_features.csv` | One row per subscriber at T-7: the 24 fields in `configs/schemas/user_record.schema.json` (22 features + `user_id`, `user_name`) plus `churned` |
| `hero_inference_record.json` | One scoring-time record with the same 24 fields and no label |

Only voluntary lapses count as `churned = 1`. Renewals lost to a failed card and
subscribers who scheduled a cancel before T-7 must be left out of the table, as in the
synthetic generator.

The lakehouse writes this contract from raw
billing and usage events, computing every feature as of each renewal's T-7 and deriving the
label from billing events. The last run, recorded in
[`results/lakehouse_e2e_summary.json`](../../results/lakehouse_e2e_summary.json):
7,387 renewals routed to the model (7.4% voluntary lapse), calibrated test AUC 0.726, and
Maya's event-built record scored 0.774 raw → 0.210 calibrated, medium, `limit_reset`. Those
numbers are a different world from the synthetic run and are not the published ladder.

## Path

```text
bronze (subscriptions, invoices, usage events, tickets)
  → silver
  → gold churn_user_features (as of T-7)
  → data/export/churn_user_features.csv + hero_inference_record.json
  → retention-radar/data/external/        (scripts/sync_lakehouse_exports.sh)
  → train / calibrate / score
```

The lakehouse owns the features. Model choice lives in this repo
([algorithm-landscape.md](../guides/algorithm-landscape.md)).

## Choosing the source

`CHURN_DATA_SOURCE`:

- `synthetic`: the seed-42 generator. `run_all.sh` defaults to this so a leftover export
  cannot change the published numbers. CI always uses it.
- `lakehouse`: require `data/external/churn_user_features.csv`.
- `auto`: use `data/external/` if present, else synthetic.

With `lakehouse` (or `auto` and an export present), `--user maya` scores
`hero_inference_record.json` instead of `data/raw/subscribers/maya.json`.

## One command

With the lakehouse repo beside this one (or pass its path):

```bash
./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse
```

Steps: generate the lakehouse sample (`N_USERS`, default 8000; `CHURN_SEED=42`) → build
gold locally with pandas (no Docker) → sync the export → `CHURN_DATA_SOURCE=lakehouse
./scripts/run_all.sh` → write `results/lakehouse_e2e_summary.json`.

The run sets `RETENTION_RADAR_ARTIFACT_DIR=artifacts/lakehouse_run`, so models, metrics,
packets and the generated model card land there. Committed `models/` and
`docs/model-card.md` are not touched. It does rewrite the committed summary file;
`git checkout -- results/lakehouse_e2e_summary.json` if you do not mean to publish it.

Sync only, no retrain:

```bash
./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export
```

## Limits

- Lakehouse numbers come from events aggregated by the lakehouse jobs, not from the
  seed-42 generator. Do not put them in the same table as the published results.
- The published `models/metrics.json` is always the synthetic run.

Related: [data-dictionary.md](data-dictionary.md) ·
[e2e-free-platforms.md](../guides/e2e-free-platforms.md)
