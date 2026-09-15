# Best practices checklist (FOSS Retention Radar)

Short checklist for contributors and readers adapting this teaching repo.

## Reproducibility

- [ ] Keep `RANDOM_SEED = 42` (or document any change).
- [ ] Prefer `./scripts/run_all.sh` so generate → train → evaluate → packet stay in lockstep.
- [ ] Cite **`models/metrics.json`** over prose numbers after regenerating.
- [ ] CI uses smaller `N_USERS` / `N_OPTUNA_TRIALS`; local defaults are larger — both are OK if documented.
- [ ] `pytest -q` with `CHURN_DATA_SOURCE=synthetic` so a local lakehouse export cannot hijack CI-shaped tests.

## Dual-world data (synthetic vs lakehouse)

- [ ] **Published article / model-card ladder** is the **synthetic** seed-42 run. Do not overwrite committed `models/` with a lakehouse retrain.
- [ ] Feature SoR is [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse): `make churn-gold-local` (no Docker) or `make churn-e2e` (Spark). Sync with `./scripts/sync_lakehouse_exports.sh`.
- [ ] `CHURN_DATA_SOURCE=auto|synthetic|lakehouse`. After `./scripts/run_lakehouse_e2e.sh`, restore: `git checkout -- models/ docs/MODEL_CARD.md docs/data/data-dictionary.md`.
- [ ] Lakehouse Santosh is event-aggregated as-of **2024-03-02**. Scores differ from seed-42 Santosh **by design**. Do not mix the two in one caption.

## Leakage & evaluation honesty

- [ ] Features available at **score time only** (no post-churn or label-derived columns).
- [ ] Optuna / threshold search on **validation**; touch **test** once.
- [ ] Report Dummy + LogReg + RF + default XGB + Optuna XGB + LightGBM + CatBoost (honest ladder), not XGB alone.
- [ ] If AUC ≈ 1.0 on synthetic data, **stop and debug** before celebrating.
- [ ] Unknown `plan_tier` **raises**. Silent zeros are train/serve skew.

## Calibration & thresholds

- [ ] Treat raw `predict_proba` as ranking unless calibrated.
- [ ] Serve risk bands from **calibrated** `p` when the calibrator is enabled.
- [ ] Remember best-F1 τ often ≠ 0.5; document the dial you use.

## Human-in-the-loop (HITL)

- [ ] Decision packet `auto_action` stays **`none`** (`src/retention_radar/serving/policy.py`).
- [ ] UI copy never claims auto-cancel or destiny ("will churn").
- [ ] Explanations (SHAP) are **associative drivers**, not causal proof.

## Drift

- [ ] `python -m retention_radar.cli.drift_check` compares the **active** users table (`resolve_users_csv`) to `models/feature_stats.json`.
- [ ] Teaching default exits 0. Gate a pipeline / CI with `--strict`.
- [ ] Do not compare lakehouse gold to leftover synthetic `data/raw/users.csv`.

## Synthetic data disclaimer

- [ ] README, model card, and Streamlit mention **synthetic / no real PII**.
- [ ] Do not claim production ROI or protected-class fairness audits from this repo.
- [ ] Slice metrics by `plan_tier` are educational segments only.

## Train ≠ serve

- [ ] Serve path loads joblib + feature names; does not fit encoders or call Optuna.
- [ ] Shared contract: 22 features, bands/thresholds, model version metadata.
- [ ] Never retrain inside Streamlit on page load.

## Articles (Medium / Data Engineer Things)

- [ ] H1 + italic deck; no em-dash in title/subtitle; no `Part N of N`.
- [ ] One unique sketch under every `##`; italic credit under every embed.
- [ ] One lakehouse CTA per part. No private-repo clone pointers in article bodies.
- [ ] Seed-42 numbers in prose match `models/metrics.json`. Do not “improve” them.

## CI & docs

- [ ] `pytest -q` green before PR (`tests/test_articles_medium.py` is part of that).
- [ ] After metric-changing PRs, refresh guides via `python -m retention_radar.cli.docs_gen` — but not after a lakehouse smoke if you still publish the synthetic ladder.
- [ ] Keep engineering docs under `docs/` (`ARCHITECTURE.md`, `MODEL_CARD.md`); analysis under `results/`.
- [ ] Free E2E path stays documented in [e2e-free-platforms.md](e2e-free-platforms.md).

## Related

- [FOLDER_STRUCTURE.md](../FOLDER_STRUCTURE.md) · [ARCHITECTURE.md](../ARCHITECTURE.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) · [data-foundation-lakehouse.md](../data/data-foundation-lakehouse.md) · [../results/BENCHMARKS.md](../../results/BENCHMARKS.md)
