# Serve bundle

These files are the **read-only** artifacts Streamlit Community Cloud and Hugging Face Spaces load. They exist so the demo does **not** run Optuna on page load.

Serve-bundle training env pin: **scikit-learn==1.9.1** (matches `calibrator.joblib` / `requirements.lock`). Keep `requirements.txt` pinned to the same major.minor so Mac/Linux loads do not drift (isotonic calibrator is sklearn-pickled).

This committed copy is the **seed-42 reference train**: `N_USERS=5000`, default Optuna trials (**20**), n_train/val/test = **3000 / 1000 / 1000**. Cite [MODEL_CARD.md](../docs/MODEL_CARD.md) and `models/metrics.json` together — one number set.

CI still smoke-trains (`N_USERS=800`, `N_OPTUNA_TRIALS=5`) and does **not** overwrite this bundle on GitHub Actions.

To refresh:

```bash
export PYTHONPATH="$(pwd)/src"
unset N_USERS N_OPTUNA_TRIALS
./scripts/run_all.sh
python -m retention_radar.cli.docs_gen
make docs-results  # copies plots → results/plots/
```

Then commit the files listed in [docs/guides/e2e-free-platforms.md](../docs/guides/e2e-free-platforms.md).
