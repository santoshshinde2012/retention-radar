# data/ — layer roles (CDS parity)

Aligned with [cookiecutter-data-science](https://cookiecutter-data-science.drivendata.org/) data layers. Generated CSVs are gitignored; placeholders (`.gitkeep`) and documented samples stay.

| Layer | Path | Role |
|-------|------|------|
| **raw** | `raw/` | Immutable inputs: Santosh hero JSON (`santosh_shinde.json`), generated `users.csv` (gitignored) |
| **interim** | `interim/` | Intermediate transforms (empty in teaching path; CDS parity) |
| **processed** | `processed/` | Model-ready tables if you materialize them locally (empty by default; features built in-memory) |
| **external** | `external/` | Third-party / lakehouse gold sync (`churn_user_features.csv`, `santosh_inference_record.json`) |
| **use cases** | [`use_cases/`](use_cases/README.md) | Committed serving scenarios from the seed-42 holdout: one record per HITL path, invalid records the service must hold, a weekly batch, reviewer decisions, day-30 labels (`make use-cases`) |

```bash
# Prefer synthetic (CI / published ladder):
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh

# Prefer lakehouse gold when exports are present:
./scripts/sync_lakehouse_exports.sh /path/to/local-data-lakehouse/data/export
CHURN_DATA_SOURCE=auto python -m retention_radar.cli.ingest
```

See [docs/data/data-foundation-lakehouse.md](../docs/data/data-foundation-lakehouse.md) and [external/README.md](external/README.md).
