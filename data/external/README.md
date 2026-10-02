# Lakehouse exports

Files synced from [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse).
They must follow the v2 renewal contract.

| File | Contents |
|------|----------|
| `churn_user_features.csv` | T-7 renewal table: 24 fields + `churned` |
| `hero_inference_record.json` | one scoring-time record, 24 fields, no label |

```bash
# In the lakehouse repo, build gold, then from this repo:
./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export

# Train and score on them
CHURN_DATA_SOURCE=lakehouse python -m retention_radar.cli.ingest
CHURN_DATA_SOURCE=lakehouse python -m retention_radar.cli.train
CHURN_DATA_SOURCE=lakehouse python -m retention_radar.cli.infer --user santosh   # scores hero_inference_record.json
```

`CHURN_DATA_SOURCE=auto` uses these files whenever they exist; `synthetic` ignores them.
To keep the committed `models/` untouched, use `./scripts/run_lakehouse_e2e.sh` instead,
which trains under `artifacts/lakehouse_run/`.
