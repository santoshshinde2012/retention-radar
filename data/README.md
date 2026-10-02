# data/

Layout follows the cookiecutter-data-science data layers.

| Layer | Path | Contents |
|-------|------|----------|
| raw | `raw/` | `renewals_t7.csv`: the model table, one row per subscriber at T-7 (generated, gitignored). `renewals_all.csv`: every generated renewal with `outcome` and `route` (dunning, cancel_flow, model), for auditing the label (generated, gitignored). `subscribers/maya.json`, `subscribers/arjun.json`: scoring-time records for the worked examples (committed) |
| interim | `interim/` | empty |
| processed | `processed/` | empty; features are built in memory |
| external | `external/` | lakehouse exports: `churn_user_features.csv`, `hero_inference_record.json` (gitignored) |
| use cases | [`use_cases/`](use_cases/README.md) | one renewal day from test-split subscribers: scenarios, invalid records, `daily_t7_batch.csv`, `actions_taken.csv`, `renewal_outcomes.csv` (committed) |

```bash
# Generate the synthetic tables (seed 42)
CHURN_DATA_SOURCE=synthetic python -m retention_radar.cli.generate_data

# Use lakehouse exports instead (needs the v2 export)
./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export
CHURN_DATA_SOURCE=auto python -m retention_radar.cli.ingest
```

Field definitions: [docs/data/data-dictionary.md](../docs/data/data-dictionary.md).
Lakehouse: [docs/data/data-foundation-lakehouse.md](../docs/data/data-foundation-lakehouse.md).
