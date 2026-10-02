# Changelog

## 2026-10-02 — v2 on main (local-first lakehouse stack)

Radar `main` still read the v1 export (`santosh_inference_record.json`, 22 generic SaaS features),
which local-data-lakehouse no longer writes. The v2 reader (PR #18) was closed unmerged, so `main`
could not consume the lakehouse export at all. This release puts v2 on top of `main` (#19, #20):

- Reader and model: the v2 contract (`churn_user_features.csv` with the 22 T-7 features, and
  `hero_inference_record.json`) and the Linux-trained seed-42 v2 bundle, unchanged from `953af3a`.
- Kept from #19 / #20: real PSI drift (it falls back to SMD when a bundle's `feature_stats.json`
  has no `psi_bins`, which is the case for the committed v2 bundle), slice metrics at τ, the
  lowercase file names (every v2 reference is updated), pre-commit and `pyproject.toml` metadata.
- Superseded by v2: #19's τ-tied risk bands and `hitl_action`. v2's expected-value playbook policy
  with fixed band edges (0.10 / 0.30) and a 10% holdout replaces them, and `test_at_tau` is not in
  the v2 `metrics.json`.

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
- Ladder (test AUC): LogReg 0.781, CatBoost 0.765, RF 0.758, Optuna XGBoost 0.757,
  LightGBM 0.736, default XGBoost 0.729, Dummy 0.500. The committed bundle is trained on
  Linux x86-64, where `make reproduce` matches it exactly.
- Calibration: Platt (sigmoid) replaces isotonic. Isotonic gave a staircase that tied most
  of the queue and returned P = 0.000 for some subscribers. Test Brier 0.163 raw → 0.079
  calibrated (0.087 for always predicting the base rate). τ = 0.16.
- Policy: `serving/policy.py` now picks, per subscriber above τ, the approved playbook in
  `config.PLAYBOOKS` with the highest expected value (`in_app_usage_tips`, `limit_reset`,
  `pause_offer`, `cancel_flow_discount`, `personal_email` for Ultra only), with a
  deterministic 10% holdout. Bands are low < 0.10 ≤ medium < 0.30 ≤ high. `auto_action` is
  always `none`; `hitl_required` is true only for `personal_email`.
- Worked examples: Maya (Pro, capped, first renewal since the cap cut; 0.717 → 0.288,
  medium, `limit_reset`) and Arjun (steady Pro+; 0.124 → 0.025, low, `no_action`) replace
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
  test AUC 0.726, Maya 0.210 → `limit_reset`.
- Latency: warm single-row score + calibrate, p50 about 2.7 ms on a GitHub Actions runner.
- New `python -m retention_radar.cli.analysis` (run by `run_all.sh`; committed copy
  `results/analysis.json`): paired bootstrap of logistic regression vs tuned XGBoost,
  isotonic vs Platt, deciles/quintiles, and holdout sizes for a 9:1 split.
- `scripts/run_local_e2e.sh`: works with macOS bash 3.2 (no negative array index).

## Between 0.1.0 and 2026-09-30

- chore(naming): consistent file names. Docs are lowercase kebab-case, data and machine-readable outputs are snake_case, and generic files no longer carry a person's name: `results/santosh_decision_packet.sample.json` → `results/example_decision_packet.json`, `artifacts/santosh_decision_packet.json` → `artifacts/example_decision_packet.json`, `data/raw/santosh_shinde.json` → `data/raw/example_account.json`, `results/SANTOSH_ANALYSIS.md` → `results/example-account-analysis.md`, `docs/case-study/santosh-case-study.md` → `docs/case-study/example-account-case-study.md`, `notebooks/01_explore_santosh.ipynb` → `notebooks/01_explore_example_account.ipynb`, `results/lakehouse-e2e-summary.json` → `results/lakehouse_e2e_summary.json`, `results/BENCHMARKS.md` → `results/benchmarks.md`, and `docs/{ARCHITECTURE,FOLDER_STRUCTURE,GETTING_STARTED,MODEL_CARD}.md` and `docs/guides/{ALGORITHM_LANDSCAPE,BEST_PRACTICES,DEPLOY_LATER,START_HERE}.md` → lowercase kebab-case. Every reference in code, tests, scripts, CI and docs is updated. The lakehouse export name `santosh_inference_record.json` is unchanged because it is produced by local-data-lakehouse. The CLI shortcut `--user santosh` still works.
- docs: the repo describes the project only; references to external write-ups are removed. README gains Architecture and Results sections; CONTRIBUTING gains lint / pre-commit and naming conventions; `pyproject.toml` gains URLs, classifiers and a `dev` extra; optional `.pre-commit-config.yaml` (ruff, JSON / YAML checks). Unused `config.ARTICLES_DIR` removed.
- fix(serving/policy): risk bands are now aligned with τ, so a band always implies its action: low `p < τ/2` → monitor, medium `τ/2 ≤ p < τ` → nurture / check-in, high `p ≥ τ` → retention outreach, escalating at `p ≥ ESCALATE_PROB` (0.60). The old fixed edges (0.30 / 0.60) put nurture in both low and medium and outreach in medium. `risk_band(p, threshold)` now takes τ; `hitl_action` derives the band and raises on a stale band argument. Actions for every record are unchanged; only band labels move (Santosh stays low / monitor; payment_friction medium → high; the lakehouse Santosh at 0.1696 with τ 0.17 becomes medium; the 2026-09-25 run labelled it low). Use-case pack rebuilt (new_trial_friction is now a medium-band record, user_00548). Weekly batch escalate / outreach / nurture / monitor is 34 / 16 / 14 / 143 (was 33 / 14 / 18 / 142) and review decisions are 66 (was 67). `results/lakehouse_e2e_summary.json` is refreshed from the 2026-10-01 CI lakehouse E2E run: Santosh's band is medium, every other value matches the 2026-09-25 run.
- fix(drift): the "z" in `drift_check` divided the mean shift by the per-row standard deviation, which is a standardised mean difference, not a z-score, and the docs called it "PSI-lite". Drift is now a real PSI over training-decile bins saved in `feature_stats.json["<col>"]["psi_bins"]` (exact reference shares, ties handled); severity: any PSI ≥ 0.25 or five features ≥ 0.10 → severe, any ≥ 0.10 → mild. SMD and an SE-based mean z (`shift / (std / √n)`) are reported as context only. `--z-threshold` is kept as an alias of `--smd-threshold` (fallback when a bundle has no `psi_bins`).
- fix(evaluate): the confusion matrix and per-plan_tier slice precision / recall were computed at 0.5 while the service acts at τ = 0.34. Both now use τ; `metrics.json` gains `test_at_tau` (precision 0.577, recall 0.628, F1 0.602, 208 of 1,000 test accounts flagged). The ladder's `*_test` @0.5 values are unchanged and labelled as ladder-comparison only. `slice_metrics_by_plan_tier` requires the threshold. Plan-tier precision / recall at τ (was @0.5): enterprise 0.429 / 0.600 (0.75 / 0.30), free 0.570 / 0.640 (0.641 / 0.562), pro 0.548 / 0.447 (0.667 / 0.421), starter 0.635 / 0.741 (0.661 / 0.685).
- docs(metrics): `validation_reuse` in `metrics.json` and the model card states that the validation split picks the Optuna model, fits the isotonic calibrator and sets τ (no early stopping), so validation scores are optimistic. No modelling change: every published AUC / Brier / τ / Santosh number reproduces exactly. Regenerated: `models/metrics.json` (latency block kept at the published p50 2.01 ms), `models/feature_stats.json`, `docs/model-card.md`, `results/plots/confusion_matrix.png`, `results/example_decision_packet.json`, `data/use_cases/*`; model joblibs and other plots are byte-identical. Tests: 88 passed (was 82), new `tests/test_drift_psi_and_tau.py`.

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
- fix: `run_all.sh` wrote the Santosh packet to `artifacts/` regardless of `RETENTION_RADAR_ARTIFACT_DIR`, so lakehouse E2E clobbered the synthetic packet and wrote `null` Santosh fields into `results/lakehouse_e2e_summary.json`; the summary step now refuses to run without the packet. Re-verified lakehouse E2E on Python 3.12 (calibrated test AUC ≈ 0.694; Santosh 0.399 → 0.170 · low · nurture — unchanged from the published summary).
- feat: outcome write-back — `cli.hitl_outcomes` joins the HITL review log to later labels (`user_id, churned[, observed_at]`, no backward leakage) and reports observed churn per band / action taken.
- feat: cohort percentiles fall back to `feature_stats.json` quantiles when `users.csv` is absent (Streamlit Cloud).
- fix: API rejects unknown `plan_tier` with 422 (case/whitespace normalised); clearer drift_check hint when the CSV is missing.
- chore: ruff `target-version = py311`; CI lints `src/ tests/ app/`; `load_payload` → `load_model_bundle` (alias kept); docs aligned with Python 3.11+ and the real CI steps.
- fix: pin scikit-learn==1.9.1 to match seed-42 calibrator pickle; docs honesty — lakehouse E2E isolation (no models/ overwrite / no restore ritual).
- feat: FOSS production-shaped path — `cli.batch_score`, HITL review log (template + schema), thin local FastAPI `POST /v1/churn/score` (no auth); docs + tests. Live Streamlit demo URL remains TBD.
- teaching: lakehouse E2E isolation via `RETENTION_RADAR_ARTIFACT_DIR` (no clobber of seed-42 `models/` / MODEL_CARD); Start-here trilogy hub; seed-42 canary + CI snapshot; exploration notebook; Streamlit deploy readiness (TBD live URL).
- chore: production layout — `configs/schemas/`, CDS `data/{interim,processed}/`, `notebooks/`, `.env.example`; rewrite README + polish `results/benchmarks.md`; keep `results/` name for article dig-deeper URLs.
- chore: remove redundant `models/.gitkeep` (seed-42 bundle keeps `models/`); fix nested `docs/**` relative links after guides/data/case-study layout.
- Restructure to production folder layout: only `src/retention_radar/` under `src/`, CLI under `retention_radar.cli`, nested `docs/{guides,data,case-study}/`, runtime-only `artifacts/`.
- Add CatBoost (default) to the honest bake-off ladder; keep calibrated Optuna XGBoost as serving hero.
- Document algorithm landscape (IN vs DEFER: TabPFN, survival, conformal, uplift, …).

## 0.1.0 — 2026-09-15

- Initial public teaching repo: full E2E Retention Radar codebase (train → calibrate → serve → Santosh HITL).
- Package layout under `src/retention_radar/` with CLI under `src/retention_radar/cli/`.
- Committed seed-42 serve bundle in `models/`.
- Engineering docs: `architecture.md` (SOLID), `model-card.md`, minimal `docs/`.
- Dual-path data: synthetic generator + [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) gold sync.
- CI: GitHub Actions synthetic smoke + pytest.
