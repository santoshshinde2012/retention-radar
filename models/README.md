# Serve bundle

The files the UI, API and CLI load. They are committed so a free host can serve the app
without training.

This is the seed-42 reference run: 8,000 generated renewals, 7,329 in the model table,
split 4,397 / 1,466 / 1,466, 20 Optuna trials, Platt calibration. Trained with
scikit-learn 1.9.1 and XGBoost 3.4.1 (pinned in `requirements.txt`; the calibrator is a
scikit-learn pickle). Cite this folder's `metrics.json` together with
[MODEL_CARD.md](../docs/MODEL_CARD.md).

CI trains a smaller smoke bundle (`N_USERS=800`, `N_OPTUNA_TRIALS=5`) in
`artifacts/smoke/` and never overwrites these files.

To refresh:

```bash
unset N_USERS N_OPTUNA_TRIALS
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
python -m retention_radar.cli.docs_gen
make docs-results      # copy plots to results/plots/
```

Then commit `churn_xgb.joblib`, `calibrator.joblib`, `feature_names.json`,
`feature_stats.json` and `metrics.json`.
