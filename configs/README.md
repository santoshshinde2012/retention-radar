# configs/

Project configuration and contracts (cookiecutter-data-science / production ML convention).

| Path | Role |
|------|------|
| [`schemas/user_record.schema.json`](schemas/user_record.schema.json) | JSON Schema for a single user / inference record (train + serve contract) |
| [`hitl_review_log.schema.json`](hitl_review_log.schema.json) | Schema for one HITL review-log row (predict→act; outcome deferred) |
| [`templates/hitl_review_log.csv`](templates/hitl_review_log.csv) | CSV header template for the review log |

**Not here (by design):** Airflow DAGs, DVC, MLflow experiment YAML — this is a FOSS teaching repo. A thin local FastAPI OpenAPI surface lives in code (`serving/api.py`), not as committed OpenAPI YAML. Hyperparameters and paths live in `src/retention_radar/config.py` (env overrides: `N_USERS`, `N_OPTUNA_TRIALS`, `CHURN_DATA_SOURCE`).

Package code reads the schema via `retention_radar.config.USER_RECORD_SCHEMA_PATH`.
