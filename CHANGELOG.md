## Unreleased

- feat: serving use-case pack `data/use_cases/` built from seed-42 **holdout** rows (`cli.build_use_cases`, `--check` against the committed bundle): 7 scenarios covering every band and HITL action (incl. nurture in both bands and a reviewer override), 4 invalid records, a 211-row weekly batch, 66 reviewer decisions and day-30 labels; `make use-cases` walks queue → packets → held records → reviews → outcomes; golden tests run every scenario through packet, CLI, batch, API and Streamlit.
- fix(serving): invalid records are never scored or queued — packets get `hold: fix input data` (single_record exits 1), batch rows go to `*_rejected.csv` instead of crashing the job, the API returns 422 with the reason. `validate_payload` no longer raises on text in numeric fields.
- feat(serving): batch output is a ranked review queue (calibrated risk, raw risk inside isotonic plateaus) scored in one vectorised pass; `--scored-at`, `--strict`; label/metadata columns are ignored so gold CSVs score as-is. API adds rationale, τ, validation warnings, `POST /v1/churn/batch`, `POST /v1/churn/reviews` (logged against the service's own score) and skips SHAP unless `?shap=true`. `hitl_log` bulk-imports reviewer decisions against a queue and accepts `--timestamp`.
- feat(ui): Streamlit **Use case** picker; widgets keyed per preset; slider bounds widen to fit a record instead of crashing; auto engagement_trend only when the record already follows the formula, so presets score exactly like the API.
- fix: τ (`best_f1_threshold`) is now chosen by a best-F1 sweep on calibrated **validation** probabilities and frozen; test is read once at τ. It was previously swept on test (holdout leakage into a serving decision). The validation optimum is also **0.34**, so τ, test F1 at τ (0.6015), Santosh's action and every published number are unchanged; metrics.json gains `best_f1_threshold_source` and `val_f1_at_threshold` (0.699), and the threshold plot shows the validation sweep.
- fix: Santosh's calibrated probability is 0.016461 → **0.016** at 3 dp (docs said 0.017 via double rounding); `risk_band` raises on NaN/inf instead of returning "high".
- fix: published seed-42 numbers now reproduce locally. The committed bundle was trained with XGBoost 3.4.1, which needs Python 3.12, but the repo documented/tested Python 3.11 with `xgboost>=2.0` (pip resolved 3.2.0 → τ 0.34 → 0.42, Santosh raw 0.043 → 0.052). Runtime is now Python 3.12+ (`runtime.txt`, CI, `requires-python`), model-affecting libs are pinned, and a clean `make run` re-creates all 182 non-latency metrics exactly.
- feat: `make reproduce` / `cli.check_reproduction` (retrain in isolation, diff vs committed metrics + Santosh packet) and `make e2e-local` / `scripts/run_local_e2e.sh` (reproduce, pytest, every CLI surface, live API + Streamlit, lakehouse E2E vs committed summary, isolation check); CI job `e2e-local` runs it. Headless Streamlit AppTest added to pytest.
- fix: Makefile targets use `.venv` without manual activation and `make setup` refuses Python < 3.12; lakehouse E2E uses the project venv and an absolute interpreter path; `LAKEHOUSE_SUMMARY_PATH` lets verification runs avoid rewriting the committed summary.
- fix: `run_all.sh` wrote the Santosh packet to `artifacts/` regardless of `RETENTION_RADAR_ARTIFACT_DIR`, so lakehouse E2E clobbered the synthetic packet and wrote `null` Santosh fields into `results/lakehouse-e2e-summary.json`; the summary step now refuses to run without the packet. Re-verified lakehouse E2E on Python 3.12 (calibrated test AUC ≈ 0.694; Santosh 0.399 → 0.170 · low · nurture — unchanged from the published summary).
- feat: outcome write-back — `cli.hitl_outcomes` joins the HITL review log to later labels (`user_id, churned[, observed_at]`, no backward leakage) and reports observed churn per band / action taken.
- feat: cohort percentiles fall back to `feature_stats.json` quantiles when `users.csv` is absent (Streamlit Cloud).
- fix: API rejects unknown `plan_tier` with 422 (case/whitespace normalised); clearer drift_check hint when the CSV is missing.
- chore: ruff `target-version = py311`; CI lints `src/ tests/ app/`; `load_payload` → `load_model_bundle` (alias kept); docs aligned with Python 3.11+ and the real CI steps.
- fix: pin scikit-learn==1.9.1 to match seed-42 calibrator pickle; docs honesty — lakehouse E2E isolation (no models/ overwrite / no restore ritual).
- feat: FOSS production-shaped path — `cli.batch_score`, HITL review log (template + schema), thin local FastAPI `POST /v1/churn/score` (no auth); docs + tests. Live Streamlit demo URL remains TBD.
- teaching: lakehouse E2E isolation via `RETENTION_RADAR_ARTIFACT_DIR` (no clobber of seed-42 `models/` / MODEL_CARD); Start-here trilogy hub; seed-42 canary + CI snapshot; exploration notebook; Streamlit deploy readiness (TBD live URL).
- chore: production layout — `configs/schemas/`, CDS `data/{interim,processed}/`, `notebooks/`, `.env.example`; rewrite README + polish `results/BENCHMARKS.md`; keep `results/` name for article dig-deeper URLs.
- chore: remove redundant `models/.gitkeep` (seed-42 bundle keeps `models/`); fix nested `docs/**` relative links after guides/data/case-study layout.
- Restructure to production folder layout: only `src/retention_radar/` under `src/`, CLI under `retention_radar.cli`, nested `docs/{guides,data,case-study}/`, runtime-only `artifacts/`.
- Add CatBoost (default) to the honest bake-off ladder; keep calibrated Optuna XGBoost as serving hero.
- Document algorithm landscape (IN vs DEFER: TabPFN, survival, conformal, uplift, …).

# Changelog

## 0.1.0 — 2026-09-15

- Initial public teaching repo: full E2E Retention Radar codebase (train → calibrate → serve → Santosh HITL).
- Package layout under `src/retention_radar/` with CLI under `src/retention_radar/cli/`.
- Committed seed-42 serve bundle in `models/`.
- Engineering docs: `ARCHITECTURE.md` (SOLID), `MODEL_CARD.md`, minimal `docs/`.
- Dual-path data: synthetic generator + [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) gold sync.
- CI: GitHub Actions synthetic smoke + pytest.
