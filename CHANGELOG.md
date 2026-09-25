## Unreleased

- fix: `run_all.sh` wrote the Santosh packet to `artifacts/` regardless of `RETENTION_RADAR_ARTIFACT_DIR`, so lakehouse E2E clobbered the synthetic packet and wrote `null` Santosh fields into `results/lakehouse-e2e-summary.json`; the summary step now refuses to run without the packet. Re-verified lakehouse E2E (calibrated test AUC ≈ 0.702; Santosh 0.396 → 0.178 · low · nurture).
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
