# Changelog

## 2026-09-30 — coding-assistant renewals (v2)

The use case changed. The repo now scores renewals of a monthly AI coding assistant plan
instead of generic AI-platform churn. Every model, number and file name below replaces
its earlier counterpart; nothing from the earlier setup is kept for compatibility except
where noted.

- Use case: a self-serve AI coding assistant (IDE extension + CLI agent) on monthly plans,
  Pro $20, Pro+ $60, Ultra $200. Teams plans are out of scope and held by validation.
  Sources and reasoning: `docs/USE_CASE.md`.
- Data: one row per paying subscriber seven days before renewal (T-7). New 24-field
  contract (22 features + `user_id`, `user_name`) covering tenure, habit, usage against the
  cap, quality and surface. `data/raw/users.csv` → `data/raw/renewals_t7.csv`; new
  `data/raw/renewals_all.csv` keeps every renewal with `outcome` and `route`.
- Label: `churned` = voluntary lapse. Failed cards go to dunning and cancels scheduled
  before T-7 go to the cancel flow; both are excluded from the model table.
- Cohort: `N_USERS` default 8000 → 7,329 model rows, 9.6% base rate, split 4,397 / 1,466 / 1,466.
- Ladder (test AUC): LogReg 0.781, CatBoost 0.770, Optuna XGBoost 0.763, RF 0.758,
  LightGBM 0.736, default XGBoost 0.728, Dummy 0.500.
- Calibration: Platt (sigmoid) replaces isotonic. Isotonic gave a staircase that tied most
  of the queue and returned P = 0.000 for some subscribers. Test Brier 0.162 raw → 0.079
  calibrated (0.087 for always predicting the base rate). τ = 0.14.
- Policy: `serving/policy.py` now picks, per subscriber above τ, the approved playbook in
  `config.PLAYBOOKS` with the highest expected value (`in_app_usage_tips`, `limit_reset`,
  `pause_offer`, `cancel_flow_discount`, `personal_email` for Ultra only), with a
  deterministic 10% holdout. Bands are low < 0.10 ≤ medium < 0.30 ≤ high. `auto_action` is
  always `none`; `hitl_required` is true only for `personal_email`.
- Worked examples: Maya (Pro, capped, first renewal since the cap cut; 0.554 → 0.153,
  medium, `limit_reset`) and Arjun (steady Pro+; 0.066 → 0.023, low, `no_action`) replace
  the earlier single example. Files: `data/raw/subscribers/maya.json`, `arjun.json`;
  `--user maya|arjun`; `make infer` writes `artifacts/maya_decision_packet.json`.
- Renames: packet key `hitl` → `decision`; queue column `hitl_action` → `action`, plus
  `holdout`, `would_have_sent`, `expected_value_usd`; `serving/hitl_log.py` →
  `serving/action_log.py` and `cli.hitl_log` → `cli.action_log` (columns `user_id, p_cal,
  band, action_suggested, holdout, would_have_sent, executed_by, action_taken, notes,
  timestamp`); `cli.hitl_outcomes` → `cli.outcomes`, which now reports lift per playbook
  against the holdout with a Newcombe interval and no verdict under 30 per group;
  `POST /v1/churn/reviews` → `POST /v1/churn/actions` (`user_id, executed_by,
  action_taken, notes`); `configs/action_log.schema.json`, `configs/templates/action_log.csv`.
  `HitlDecisionPolicy` and `hitl_action` remain as import aliases.
- Use-case pack: `data/use_cases/` rebuilt as one renewal day: eight scenarios, four
  invalid records, `daily_t7_batch.csv`, `actions_taken.csv` (send export) and
  `renewal_outcomes.csv` (simulated from the assumed playbook effects).
- Results and docs: `results/SANTOSH_ANALYSIS.md` → `results/WORKED_EXAMPLES.md`;
  `results/maya_decision_packet.sample.json`; `docs/case-study/renewal-worked-examples.md`;
  new `docs/USE_CASE.md`; guides rewritten for the new use case. Notebook renamed to
  `notebooks/01_explore_renewals.ipynb`.
- Lakehouse: the local-data-lakehouse export must now produce the v2 contract
  (`churn_user_features.csv` with the 24 fields + `churned`, and
  `hero_inference_record.json`), built from raw billing + usage events as of T-7.
  `results/lakehouse-e2e-summary.json` refreshed from that run: 7,387 renewals, calibrated
  test AUC 0.728, Maya 0.216 → `limit_reset`.
- Latency: warm single-row score + calibrate, p50 about 1.5 ms.
- New `python -m retention_radar.cli.analysis` (run by `run_all.sh`; committed copy
  `results/analysis.json`): paired bootstrap of logistic regression vs tuned XGBoost,
  isotonic vs Platt, deciles/quintiles, and holdout sizes for a 9:1 split.
- `scripts/run_local_e2e.sh`: works with macOS bash 3.2 (no negative array index).

## Between 0.1.0 and 2026-09-30

- fix(ci): the `test` job's N=800 smoke retrain wrote into `models/`, so the golden use-case / launcher tests that follow compared against a throwaway model (12 failures). The smoke run and its strict drift check now use `RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke`. Streamlit: `use_container_width` → `width="stretch"` (removed upstream), `streamlit>=1.50`.
- fix: `python -m retention_radar.cli.drift_check --strict` ignored the exit code (the wrapper called `main()` without `SystemExit`), so CI's strict drift gate could never fail; test pins exit codes of every CLI with a `--strict`/error contract.
- fix(serving, adversarial review): one input contract (`normalize_record` + `validate_payload`) for packet, CLI, CSV batch, API score/batch and UI. NaN/±inf, oversized ints and booleans in numeric fields are held (NaN rows were scored and queued by batch); `plan_tier` case/whitespace and extra fields are handled identically everywhere; every `/v1/churn/score` 422 carries the hold block; batch rejects non-object items and duplicate `user_id`s per row instead of failing or mis-attaching reviews.
- fix(hitl): log appends are lock-protected and header-once (concurrent `/reviews` could truncate the log); bulk import is all-or-nothing, idempotent on rerun and honours `--timestamp`; held / score-less records cannot be reviewed; outcomes parse mixed ISO dates and treat unparseable dates as not yet observed (the leak guard was bypassed); a truncated prediction-log line no longer blocks every review; `RETENTION_RADAR_LOG_DIR` keeps verification runs out of the operator's logs.
- fix(ui): one decision packet drives every tab, so invalid what-if input is held on Predict / Explain too; fixed dead doc links.
- fix(data): gone_dark is now a user with no activity at all in 30 days (the old record still had sessions and API calls); stories and reviewer notes are templated from each record; the Santosh hero is labelled as validation split; `--check` re-verifies every file of the pack. `single_record --json` names its output after the user instead of overwriting the Santosh packet; `--dir` holds unreadable files instead of aborting.
- fix: `models/calibrator.joblib` was pickled as `src.retention_radar.training.calibrate`, so it only loaded when the repo root happened to be on `sys.path` (`python -m ...`). `streamlit run` (and therefore Streamlit Community Cloud), the bare `uvicorn` command in the README and the notebook failed to load the bundle (API → HTTP 500). Re-pickled from the reproduced retrain (outputs identical on a 200k-point grid); `load_calibrator` aliases the legacy path for old bundles; subprocess tests launch the app/packet like Cloud does; `e2e-local` starts uvicorn/streamlit through their console scripts.
- fix: restore the published latency block in `models/metrics.json` / model card (p50 2.01 ms); a contended benchmark run had leaked into the τ commit.
- docs: isolated "faster smoke" (`RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke`, the old command overwrote the published bundle), batch/outcome commands that work in a fresh clone, lakehouse E2E wording (it refreshes the committed summary), latency definition, action-ladder order.
- feat: serving use-case pack `data/use_cases/` built from seed-42 rows (test split, plus the Santosh hero from the validation split; `cli.build_use_cases`, `--check` re-verifies every file against the committed bundle): 7 scenarios covering every band and HITL action (incl. nurture in both bands and a reviewer override), 4 invalid records, a 211-row weekly batch, 67 reviewer decisions and day-30 labels; `make use-cases` walks queue → packets → held records → reviews → outcomes; golden tests run every scenario through packet, CLI, batch, API and Streamlit.
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

## 0.1.0 — 2026-09-15

- Initial public teaching repo: full E2E Retention Radar codebase (train → calibrate → serve → Santosh HITL).
- Package layout under `src/retention_radar/` with CLI under `src/retention_radar/cli/`.
- Committed seed-42 serve bundle in `models/`.
- Engineering docs: `ARCHITECTURE.md` (SOLID), `MODEL_CARD.md`, minimal `docs/`.
- Dual-path data: synthetic generator + [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) gold sync.
- CI: GitHub Actions synthetic smoke + pytest.
