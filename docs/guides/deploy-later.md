# Deploying the UI to Streamlit Community Cloud

There is no hosted demo yet. These are the steps when there is one. Do not put a
placeholder URL in the README before the app is live.

The API, batch scoring and the action log run locally; they are not Cloud targets.
Longer notes: [e2e-free-platforms.md](e2e-free-platforms.md).

## Before you start

- The repo is public on GitHub (`santoshshinde2012/retention-radar`).
- The branch you deploy has the committed bundle (Cloud never trains):
  `models/churn_xgb.joblib`, `models/calibrator.joblib`, `models/feature_names.json`,
  `models/feature_stats.json`, `models/metrics.json`.
- `data/raw/subscribers/santosh.json` and `arjun.json` and `data/use_cases/` are committed
  (sidebar presets).
- `runtime.txt` says `python-3.12`; `packages.txt` installs `libgomp1` for XGBoost and LightGBM.

## Steps

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Create app → `santoshshinde2012/retention-radar` → branch `main`.
3. Main file path: `app/streamlit_app.py`.
4. Python version: 3.12.
5. Leave Secrets empty.
6. Deploy and wait for `pip install -r requirements.txt` to finish.
7. Open the Decision tab with Santosh selected. Expect calibrated 0.288, `limit_reset`, `Auto action: none`.
8. Put the `*.streamlit.app` URL in `README.md` and `docs/guides/start-here.md`.

## Checks after deploy

| Check | Expect |
|-------|--------|
| App loads | No "Model not found" |
| Santosh and Arjun score | Same results as `make infer` locally |
| No training on boot | The log shows no Optuna or `cli.train` |
| Sidebar | Says the data is synthetic |

## Common failures

| Symptom | Fix |
|---------|-----|
| Model not found | Commit the five `models/` files on the deployed branch; redeploy |
| `ModuleNotFoundError` | Main file must be `app/streamlit_app.py`; the app adds `src/` itself |
| Wheel or Python errors | Use Python 3.12 |
| Out of memory, long boot | Something is training on load; remove it |
| First SHAP call slow | Normal on free CPU |

A Hugging Face Space (Streamlit SDK, CPU basic) works the same way; see
[e2e-free-platforms.md](e2e-free-platforms.md).
