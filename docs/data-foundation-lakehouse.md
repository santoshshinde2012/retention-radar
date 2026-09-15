# Data foundation — local-data-lakehouse

Retention Radar’s **feature system of record** is the FOSS laptop lakehouse:

https://github.com/santoshshinde2012/local-data-lakehouse

## Path

```text
bronze (users / daily usage / tickets / payments)
  → silver
  → gold.churn_user_features   (as-of CHURN_AS_OF, default 2024-03-02)
  → data/export/churn_user_features.csv
  → data/export/santosh_inference_record.json
  → retention-radar/data/external/   (via scripts/sync_lakehouse_exports.sh)
  → train / calibrate / Santosh infer / Streamlit
```

Feature column contract matches `schemas/user_record.schema.json` 1:1 (24 serve fields + `churned` on train).

## Two ways to build gold

| Mode | Command (lakehouse repo) | Needs Docker? |
|------|--------------------------|---------------|
| Spark E2E | `make up && make wait && make churn-e2e` | Yes |
| Local parity | `make churn-gold-local` | No (pandas mirrors Spark math) |

Scale bronze first: `make churn-sample` (`N_USERS=5000`, `CHURN_SEED=42`).

## Dual ingest in this repo

`CHURN_DATA_SOURCE`:

- `lakehouse` — require `data/external/` gold (`./scripts/run_lakehouse_e2e.sh`)
- `synthetic` — generator (`./scripts/run_all.sh` defaults here so leftover gold cannot hijack the published ladder)
- `auto` — use `data/external/*` if present, else synthetic (ingest-time helper only)

Synthetic remains the CI / offline fallback so `pytest` does not need **SILO** (S3-compatible object store in the lakehouse compose stack — not MinIO/Garage/RustFS) or Docker Spark.

## Honest limits

- Lakehouse Santosh is **event-aggregated as-of 2024-03-02**, not the seed-42 generator profile — scores differ by design.
- `models_used_count` and `seat_utilization` include documented proxies in the lakehouse gold job.
- Published seed-42 metrics.json in this repo were trained on the **synthetic** generator unless a retrain on lakehouse exports is explicitly committed.

## Related

- [data-dictionary.md](data-dictionary.md)
- [e2e-free-platforms.md](e2e-free-platforms.md)
- Articles: parts 01–08 Dig deeper / next-steps point here; part 02 is the SoR deep dive


## One-command E2E (best path)

From Retention Radar, with the lakehouse repo checked out beside it (or pass the path):

```bash
./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse
```

This runs: `churn-sample` → `churn-gold-local` → sync → `CHURN_DATA_SOURCE=lakehouse ./scripts/run_all.sh`
(train → evaluate → benchmark → Santosh infer → decision packet).

CI always sets `CHURN_DATA_SOURCE=synthetic` so GitHub Actions never requires Silo/Spark.

## Verified lakehouse E2E (box, 2026-09-14)

One-command path used: `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse`.

| Step | Result |
|------|--------|
| Sample | `N_USERS=5000`, `CHURN_SEED=42`, Santosh `u-0001` |
| Gold export | `data/export/churn_user_features.csv` + `santosh_inference_record.json` |
| Sync | → `data/external/` via `scripts/sync_lakehouse_exports.sh` |
| Train (lake gold) | `n_train=3000` · train churn ≈ **0.17** · Optuna XGB val AUC ≈ **0.721** · calibrated test AUC ≈ **0.694** |
| Santosh (lake as-of) | raw ≈ **0.399** · calibrated ≈ **0.170** · band **low** · HITL nurture |
| Drift lite | **severe** vs train feature_stats (expected when scoring one power-user against population means) |

### Dual-world honesty

| Track | Purpose | Published article ladder? |
|-------|---------|---------------------------|
| **Synthetic** (`CHURN_DATA_SOURCE=synthetic`, seed 42) | Reproducible Medium / model-card numbers (LogReg ~0.872 … Optuna XGB ~0.870) | **Yes** — keep `models/*` committed from this path |
| **Lakehouse** (`CHURN_DATA_SOURCE=lakehouse` or `auto` with exports present) | Real SoR path: bronze→silver→gold→Radar | **No** — retrain locally; do **not** overwrite committed `models/metrics.json` when publishing articles |

CI pins `CHURN_DATA_SOURCE=synthetic` so PRs stay deterministic.

Machine-readable summary and committed source of truth: [`results/lakehouse-e2e-summary.json`](../results/lakehouse-e2e-summary.json). A full lakehouse `metrics.json` may exist in the ignored runtime archive after a local E2E run, but it is not a published artifact here.

