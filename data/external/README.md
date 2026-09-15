# Lakehouse gold exports (external)

Drop Retention Radar’s training / serve inputs from
[local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse):

| File | Role |
|------|------|
| `churn_user_features.csv` | Train table (`churned` included) |
| `santosh_inference_record.json` | Single-record serve payload (no `churned`) |

```bash
# From lakehouse repo (Docker or local gold builder):
make churn-gold-local   # or: make up && make wait && make churn-e2e

# Sync into this folder:
./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export

# Train / infer prefer these automatically (CHURN_DATA_SOURCE=auto):
python -m src.ingest
python -m src.train
python -m src.infer --user santosh
```

Force synthetic: `CHURN_DATA_SOURCE=synthetic`.  
Force lakehouse: `CHURN_DATA_SOURCE=lakehouse`.
