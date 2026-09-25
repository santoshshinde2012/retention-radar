# Deploy later — Streamlit Community Cloud

Public live demo URL is still **TBD**. When you are ready to host the HITL UI (no login from agents; you do this in the browser), use these exact steps. Do **not** invent or paste a placeholder URL into README until the app is live.

Related: longer free-platform map in [e2e-free-platforms.md](e2e-free-platforms.md). Local batch / HITL log / thin FastAPI stay on your machine — they are **not** Streamlit Cloud targets.

## Preconditions

- Repo public on GitHub: `santoshshinde2012/retention-radar`
- Branch to deploy: usually `main`
- Committed serve bundle under `models/` (Cloud never trains):
  - `models/churn_xgb.joblib`
  - `models/calibrator.joblib`
  - `models/feature_names.json`
  - `models/feature_stats.json`
  - `models/metrics.json`
- Repo root already has `runtime.txt` (`python-3.12`) and `packages.txt` (`libgomp1` for LightGBM/XGBoost)
- Python **3.12** is required: the committed bundle was trained with XGBoost **3.4.1** (Python 3.12+) and scikit-learn **1.9.1**.

## Steps (Streamlit Community Cloud)

1. Open [https://share.streamlit.io/](https://share.streamlit.io/) and sign in with GitHub.
2. **Create app** → select `santoshshinde2012/retention-radar`.
3. **Branch:** `main` (or a release branch you trust).
4. **Main file path:** `app/streamlit_app.py`
5. **Python version:** `3.12` (Advanced settings if the UI asks; matches `runtime.txt` + CI).
6. **Secrets:** leave empty — this demo needs no API keys or DB.
7. Click **Deploy**. Wait for `pip install -r requirements.txt` and the boot log to go healthy.
8. Open the **Decision** tab → Santosh preset → confirm `Auto action: none` (serve-only; no fit on load).
9. Copy the real `*.streamlit.app` URL and paste it into:
   - `README.md` (Live demo row)
   - `docs/guides/START_HERE.md` (section 4)
   - optionally `docs/guides/e2e-free-platforms.md` where it says TBD

## After deploy — sanity checks

| Check | Expect |
|-------|--------|
| App loads without “Model not found” | Five `models/` files are on the deployed commit |
| Decision / Predict tabs score Santosh | Uses committed joblibs only |
| No Optuna / train on boot | Serve-only rule; Cloud must not call `cli.train` |
| Sidebar disclaimer | Synthetic / no real PII |

## Common failures

| Symptom | Fix |
|---------|-----|
| Model not found | Ensure the five `models/` files are committed on the deployed branch; Redeploy |
| `ModuleNotFoundError` for package imports | Main file must be `app/streamlit_app.py` at the repo-root layout; app adds `src/` itself |
| Wheel / Python errors | Force Python **3.12**; do not use 3.13 on free Cloud |
| Deploy OOM / long “training” | Something is fitting on load — remove it; Cloud is serve-only |
| SHAP slow first click | Expected on free CPU; importance fallback still works |

## Out of scope for Cloud

- Thin teaching FastAPI (`make api` / `serving/api.py`) — local only, **no auth**
- Batch gold scores (`python -m retention_radar.cli.batch_score`) — local / cron
- HITL review log append (`python -m retention_radar.cli.hitl_log`) — local file
- Retraining, Optuna, or writing into `models/` from the Cloud app

Optional alternative host: Hugging Face Spaces (Streamlit SDK, CPU basic) — see [e2e-free-platforms.md](e2e-free-platforms.md) §4.
