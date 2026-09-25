# End-to-end testing on free / open platforms

This guide is how to **run the full Retention Radar pipeline** without paid SaaS.
You do **not** need Databricks, SageMaker, Vertex, a hosted MLflow cloud, Snowflake, or any credit card.

Synthetic data only. Seed **42**. HITL packets always set `auto_action: none`.

Related: [GETTING_STARTED.md](../GETTING_STARTED.md) · [FOLDER_STRUCTURE.md](../FOLDER_STRUCTURE.md) · [ARCHITECTURE.md](../ARCHITECTURE.md)

## What “E2E” means here

| Step | Command | Proves |
|------|---------|--------|
| Generate | `python -m retention_radar.cli.generate_data` | Synthetic `users.csv` + Santosh JSON |
| Train | `python -m retention_radar.cli.train` | Dummy → LogReg → Optuna XGB + calibrator |
| Metrics | `python -m retention_radar.cli.evaluate` then `python -m retention_radar.cli.benchmark` | Plots + `models/metrics.json` |
| Santosh packet | `python -m retention_radar.cli.single_record --user santosh` | Validate → score → SHAP → HITL |
| Streamlit HITL | `streamlit run app/streamlit_app.py` | Serve-only Decision tab |
| Drift | `python -m retention_radar.cli.drift_check` | Lite z-score vs `feature_stats.json` |
| Slices | `python -m retention_radar.cli.slice_metrics` | Educational `plan_tier` segments |
| Explain | `python -m retention_radar.cli.explain` | XGB gain importance CSV |
| Lakehouse gold | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` | Same pipeline on bronze→gold export |

Or one shot (synthetic, published ladder): `CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh`.

**Serve never fits.** Streamlit / `infer` / `single_record` only **load** `models/*.joblib`. No Optuna on page load.

**Live demo URL:** TBD — Streamlit Community Cloud / HF Space (paste here after deploy). Deploy tip: point the app at `app/streamlit_app.py`; `runtime.txt` + `packages.txt` (libgomp1) are repo-root ready.

---

## 1. Local (laptop / codespace) — required path

Python **3.12+** (CI and Streamlit Cloud: **3.12**). The committed bundle was trained with XGBoost **3.4.1** (Python 3.12+ only) and scikit-learn **1.9.1**; on older Pythons pip resolves an older XGBoost and the published numbers drift. From the repo root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH="$(pwd)/src"
chmod +x scripts/run_all.sh
./scripts/run_all.sh
pytest -q
streamlit run app/streamlit_app.py
```

Faster smoke (same knobs as GitHub Actions):

```bash
N_USERS=800 N_OPTUNA_TRIALS=5 ./scripts/run_all.sh
```

You should see:

- `models/churn_xgb.joblib`, `models/calibrator.joblib`, `models/metrics.json`
- `artifacts/*.png` and `artifacts/santosh_decision_packet.json`
- pytest green
- Streamlit **Decision** tab: Santosh band + HITL action, `auto_action: none`

Cite **`models/metrics.json`** after a run. Narrative reference: [../results/BENCHMARKS.md](../../results/BENCHMARKS.md) (seed-42 ladder) — not a substitute for your file.

---

## 2. GitHub Actions (free for public repos)

Workflow: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml).

### What CI proves

On every push/PR to `main`, ubuntu-latest + Python 3.12:

1. `pip install -r requirements.txt && pip install -e .` (+ `ruff`)
2. Snapshot the committed seed-42 serve bundle (`models/*.joblib` + JSON) into `.ci_seed42_fixtures/`
3. Seed-42 canary: `pytest tests/test_seed42_canary.py tests/test_artifact_dir_isolation.py` against the **committed** bundle (Santosh raw ≈ 0.043 → calibrated ≈ 0.017 → low → monitor)
4. `ruff check src/ tests/ app/`
5. `./scripts/run_all.sh` with `N_USERS=800`, `N_OPTUNA_TRIALS=5` (generate → ingest → train → evaluate → slices → explain → benchmark → infer → drift → Santosh packet)
6. `pytest -q` (full suite on the freshly trained smoke bundle)
7. `python -m retention_radar.cli.drift_check --strict --z-threshold 3.0`

A second job, **`e2e-local`**, runs `make e2e-local` on Python 3.12 with local-data-lakehouse cloned beside it: exact reproduction of `models/metrics.json`, full pytest, live API + Streamlit, lakehouse gold E2E vs `results/lakehouse-e2e-summary.json`, and an isolation check.

That is canary → lint → generate → train → metrics → packet → tests → drift (+ the reproduction / full-surface job). It does **not** start Streamlit (no GUI on Actions). Local or Community Cloud covers the UI.

CI uses a **smaller** `N_USERS` / Optuna budget than a laptop default (`5000` / `20`). Do not treat CI `metrics.json` as the published seed-42 table.

### How to read a run

1. Open the repo on GitHub → **Actions**.
2. Click the workflow run for your commit.
3. Open the **test** job. Green checks mean the pipeline completed.
4. Failures: expand the step (Install / Generate / Train / Pytest / packet). Pytest output is the usual traceback.

Artifacts from CI are **not** uploaded (no paid storage). The job log is the record.

### Tightening for forks

Forks get Actions if enabled. Useful extras (still free):

| Change | Why |
|--------|-----|
| Keep `PYTHONPATH: ${{ github.workspace }}/src` | `python -m retention_radar.cli.*` needs `src` on PYTHONPATH (or `pip install -e .`) |
| Do not drop `pytest -q` | Catches serve/HITL regressions |
| Optional: add `python -m retention_radar.cli.infer --user santosh` | Extra canary (packet already scores Santosh) |
| Optional: `N_OPTUNA_TRIALS: "3"` | Faster forks; slightly noisier smoke metrics |
| Do **not** add paid runners or cloud training jobs | Out of scope for this teaching repo |

Enable Actions on the fork: **Settings → Actions → Allow all actions**.

---

## 3. Streamlit Community Cloud (free demo host)

[Streamlit Community Cloud](https://share.streamlit.io/) can host `app/streamlit_app.py` for a public GitHub repo. **No secrets are required** for this demo (no API keys, no DB).

### Why you must commit `models/` artifacts

Cloud **does not run** `./scripts/run_all.sh`. If joblibs are missing, the app shows “Model not found” and never trains (serve-only rule).

Commit these files (see `.gitignore` exceptions):

| File | Role |
|------|------|
| `models/churn_xgb.joblib` | Tuned XGB bundle |
| `models/calibrator.joblib` | Validation calibrator |
| `models/feature_names.json` | 22-column contract |
| `models/feature_stats.json` | Outlier / drift reference |
| `models/metrics.json` | Methodology / Benchmarks tabs |

After a local train:

```bash
git add models/churn_xgb.joblib models/calibrator.joblib \
        models/feature_names.json models/feature_stats.json models/metrics.json
```

`data/raw/santosh_shinde.json` is already committed (hero preset). `users.csv` stays gitignored; cohort percentiles may be empty on Cloud — scoring and HITL still work.

### Deploy steps

1. Push the repo (with model files) to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
3. **Create app** → this repo → branch `main` (or your PR branch).
4. **Main file path:** `app/streamlit_app.py`
5. **Python version:** `3.12` (matches CI; also listed in `runtime.txt`).
6. Leave **Secrets** empty.
7. Deploy. Wait for `pip install -r requirements.txt`.

The app adds `src/` to `sys.path`, so Cloud does not need a custom `PYTHONPATH` secret.

### Common failures

| Symptom | Fix |
|---------|-----|
| Model not found | Commit the five `models/` files above; redeploy |
| `ModuleNotFoundError: src` | Confirm main file is `app/streamlit_app.py` at repo root layout; Cloud working directory should be the repo |
| Deploy stuck on Optuna / OOM | Cloud must **not** train. You accidentally called `retention_radar.cli.train` from the app — do not |
| Python 3.13 / package wheels fail | Set Python **3.12** in Advanced settings / `runtime.txt` |
| SHAP slow first score | Expected on free CPU; fallback importance still works if SHAP errors |
| Empty Benchmarks images | Plots live in `artifacts/` (gitignored). Tabs still show `metrics.json`. Optional: copy plots into the repo if you want Cloud charts |
| Secrets / tokens prompted | Ignore; this app needs none |

Open **Decision** with the Santosh preset. Confirm `Auto action: none`.

---

## 4. Optional: Hugging Face Spaces (free)

Same idea as Community Cloud: host the **already trained** Streamlit app. Do not train on Space CPU.

1. Create a Space: SDK **Streamlit**, hardware **CPU basic** (free).
2. Point the Space at this GitHub repo **or** duplicate files.
3. Space README YAML (example):

```yaml
---
title: Retention Radar
sdk: streamlit
sdk_version: 1.28.0
app_file: app/streamlit_app.py
python_version: 3.12
---
```

4. Include `requirements.txt`, `app/streamlit_app.py`, `src/`, `models/` artifacts, `data/raw/santosh_shinde.json`, `docs/` (optional docs tabs).
5. No HF tokens required for a public Space serving local joblibs.

If the Space UI asks for a Gradio SDK, pick Streamlit instead — this repo’s demo is Streamlit, not Gradio.

---

## 5. What we explicitly do **not** require

- Databricks / Spark jobs
- Amazon SageMaker, Azure ML, Google Vertex
- Paid MLflow / W&B / Neptune hosting
- Feature stores, Kafka, Kubernetes
- Production FastAPI + auth (not needed to test E2E). A **thin local** FastAPI (`serving/api.py`, no auth) is an optional teaching serve path — not a Cloud deploy target.

Those are fine **after** this teaching repo. They are not part of the FOSS checklist.

---

---

## 5b. Optional: lakehouse SoR path (FOSS)

When [`local-data-lakehouse`](https://github.com/santoshshinde2012/local-data-lakehouse) is available locally:

```bash
./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse
```

That runs sample → gold CSV/JSON → sync into `data/external/` → train/eval/infer with `CHURN_DATA_SOURCE=lakehouse`.

- **CI / Medium numbers** stay on synthetic seed 42 (`CHURN_DATA_SOURCE=synthetic` in GitHub Actions).
- Lakehouse retrain sets `RETENTION_RADAR_ARTIFACT_DIR=artifacts/lakehouse_run` so committed `models/` and `docs/MODEL_CARD.md` stay untouched. The committed lakehouse source of truth is [the E2E summary](../../results/lakehouse-e2e-summary.json).
- Contract + dual-world notes: [data-foundation-lakehouse.md](../data/data-foundation-lakehouse.md).

## 6. Checklist (print / tick)

Use this once locally **or** via CI + Cloud.

- [ ] `python3.12 -m venv .venv` and `pip install -r requirements.txt`
- [ ] `export PYTHONPATH="$(pwd)/src"`
- [ ] **Generate** — `python -m retention_radar.cli.generate_data` (or `./scripts/run_all.sh`)
- [ ] **Train** — `python -m retention_radar.cli.train` (seed 42; Optuna on val only)
- [ ] **Metrics** — `python -m retention_radar.cli.evaluate` (ROC/PR/calibration/threshold plots)
- [ ] Optional latency — `python -m retention_radar.cli.benchmark`
- [ ] **Santosh packet** — `python -m retention_radar.cli.single_record --user santosh --out artifacts/santosh_decision_packet.json`
- [ ] Packet `hitl.auto_action` is **`none`**
- [ ] `pytest -q` green
- [ ] **Streamlit HITL** — Decision tab scores Santosh; no fit on page load
- [ ] **drift_check** — `python -m retention_radar.cli.drift_check` (exit 0 unless `--strict`)
- [ ] GitHub Actions run green (fork: enable Actions)
- [ ] (Demo host) Streamlit Cloud or HF Space using **committed** `models/` — no Optuna in the cloud

Notes: data is **synthetic**. Do not treat Santosh’s probability as real risk. Published [results/BENCHMARKS.md](../../results/BENCHMARKS.md) numbers are a reference full run; CI/Cloud may use a smaller bundle.

---

## 7. Code layout (where E2E calls land)

| You type | Implementation |
|----------|----------------|
| `python -m retention_radar.cli.generate_data` | `src/retention_radar/cli/generate_data.py` → `data/generate.py` |
| `python -m retention_radar.cli.train` | `src/retention_radar/training/train.py` |
| `python -m retention_radar.cli.evaluate` | `src/retention_radar/evaluation/evaluate.py` |
| `python -m retention_radar.cli.infer` | `src/retention_radar/serving/infer.py` |
| `python -m retention_radar.cli.single_record` | `src/retention_radar/serving/packet.py` |
| `python -m retention_radar.cli.drift_check` | `src/retention_radar/serving/drift.py` |
| Streamlit | `app/streamlit_app.py` → serving packet + policy |

Thin `src/retention_radar/cli/` entrypoints keep articles and CI stable. Full map: [FOLDER_STRUCTURE.md](../FOLDER_STRUCTURE.md).
