# configs/

Project configuration and contracts (cookiecutter-data-science / production ML convention).

| Path | Role |
|------|------|
| [`schemas/user_record.schema.json`](schemas/user_record.schema.json) | JSON Schema for a single user / inference record (train + serve contract) |

**Not here (by design):** Airflow DAGs, DVC, MLflow experiment YAML, FastAPI OpenAPI — this is a FOSS teaching repo. Hyperparameters and paths live in `src/retention_radar/config.py` (env overrides: `N_USERS`, `N_OPTUNA_TRIALS`, `CHURN_DATA_SOURCE`).

Package code reads the schema via `retention_radar.config.USER_RECORD_SCHEMA_PATH`.
