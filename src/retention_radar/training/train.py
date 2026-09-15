"""Train honest baselines (incl. RF + LightGBM) + Optuna XGBoost + calibrator.

Run:
    python -m src.train
"""

from __future__ import annotations

import json
import os
import warnings

import joblib
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier

from src.retention_radar import config
from src.retention_radar.data.ingest import load_users
from src.retention_radar.docs_gen import write_data_dictionary, write_model_card
from src.retention_radar.evaluation.metrics import classification_metrics
from src.retention_radar.features.transform import prepare_xy
from src.retention_radar.training.baselines import run_baselines
from src.retention_radar.training.calibrate import brier, fit_calibrator, save_calibrator
from src.retention_radar.training.split import stratified_train_val_test

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def compute_feature_stats(X) -> dict:
    """Percentiles / mean / std per training feature for outlier & cohort checks."""
    stats = {}
    for col in X.columns:
        series = X[col].astype(float)
        qs = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        stats[col] = {
            "count": int(series.shape[0]),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "p01": float(qs.loc[0.01]),
            "p05": float(qs.loc[0.05]),
            "p25": float(qs.loc[0.25]),
            "p50": float(qs.loc[0.5]),
            "p75": float(qs.loc[0.75]),
            "p95": float(qs.loc[0.95]),
            "p99": float(qs.loc[0.99]),
        }
    return stats


def _scale_pos_weight(y) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / max(pos, 1))


def train_xgb_baseline(X_train, y_train, X_val, y_val) -> XGBClassifier:
    """Simple default XGBoost for an internal tree-model comparison."""
    spw = _scale_pos_weight(y_train)
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        random_state=config.RANDOM_SEED,
        n_jobs=2,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def tune_optuna(X_train, y_train, X_val, y_val, n_trials: int):
    """Run Optuna TPE search; return best model refit on train."""
    spw = _scale_pos_weight(y_train)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "scale_pos_weight": spw,
            "random_state": config.RANDOM_SEED,
            "n_jobs": 2,
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        proba = model.predict_proba(X_val)[:, 1]
        return float(classification_metrics(y_val, proba)["roc_auc"])

    sampler = TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_params.update(
        {
            "scale_pos_weight": spw,
            "random_state": config.RANDOM_SEED,
            "n_jobs": 2,
            "eval_metric": "logloss",
        }
    )
    best = XGBClassifier(**best_params)
    best.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return best, study


def main(n_trials: int | None = None) -> None:
    if n_trials is None:
        n_trials = int(os.environ.get("N_OPTUNA_TRIALS", config.N_OPTUNA_TRIALS))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_users()
    X, y = prepare_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_train_val_test(X, y)
    print(
        f"Split sizes — train={len(X_train)} val={len(X_val)} test={len(X_test)} "
        f"churn_train={y_train.mean():.3f}"
    )

    print("Honest baselines (Dummy + LogReg + RF + LightGBM)...")
    base = run_baselines(X_train, y_train, X_val, y_val, X_test, y_test)
    print(
        f"  Dummy   val AUC={base['metrics']['dummy_val']['roc_auc']:.4f} "
        f"F1={base['metrics']['dummy_val']['f1']:.4f}"
    )
    print(
        f"  LogReg  val AUC={base['metrics']['logreg_val']['roc_auc']:.4f} "
        f"F1={base['metrics']['logreg_val']['f1']:.4f}"
    )
    print(
        f"  RF      val AUC={base['metrics']['rf_val']['roc_auc']:.4f} "
        f"F1={base['metrics']['rf_val']['f1']:.4f}"
    )
    print(
        f"  LightGBM val AUC={base['metrics']['lgbm_val']['roc_auc']:.4f} "
        f"F1={base['metrics']['lgbm_val']['f1']:.4f}"
    )

    print("Training default XGBoost (tree baseline)...")
    xgb_default = train_xgb_baseline(X_train, y_train, X_val, y_val)
    xgb_default_val = classification_metrics(
        y_val, xgb_default.predict_proba(X_val)[:, 1]
    )
    xgb_default_test = classification_metrics(
        y_test, xgb_default.predict_proba(X_test)[:, 1]
    )
    print(
        f"  XGB-default val AUC={xgb_default_val['roc_auc']:.4f} "
        f"test AUC={xgb_default_test['roc_auc']:.4f}"
    )

    print(f"Optuna tuning ({n_trials} trials)...")
    best_model, study = tune_optuna(X_train, y_train, X_val, y_val, n_trials)
    raw_val = best_model.predict_proba(X_val)[:, 1]
    raw_test = best_model.predict_proba(X_test)[:, 1]
    tuned_val = classification_metrics(y_val, raw_val)
    tuned_test = classification_metrics(y_test, raw_test)
    print(f"  Tuned val AUC={tuned_val['roc_auc']:.4f}")
    print(f"  Tuned test AUC={tuned_test['roc_auc']:.4f}")

    print(f"Calibrating probabilities on validation ({config.CALIBRATION_METHOD})...")
    calibrator = fit_calibrator(y_val, raw_val, method=config.CALIBRATION_METHOD)
    cal_val = calibrator.transform(raw_val)
    cal_test = calibrator.transform(raw_test)
    cal_val_metrics = classification_metrics(y_val, cal_val)
    cal_test_metrics = classification_metrics(y_test, cal_test)
    brier_raw_val = brier(y_val, raw_val)
    brier_raw_test = brier(y_test, raw_test)
    brier_cal_val = brier(y_val, cal_val)
    brier_cal_test = brier(y_test, cal_test)
    print(
        f"  Brier raw/cal test={brier_raw_test:.4f}/{brier_cal_test:.4f} "
        f"(lower is better)"
    )

    payload = {
        "model": best_model,
        "feature_names": list(config.MODEL_FEATURE_COLUMNS),
        "best_params": study.best_params,
        "random_seed": config.RANDOM_SEED,
        "calibrated": True,
        "calibration_method": config.CALIBRATION_METHOD,
    }
    joblib.dump(payload, config.MODEL_PATH)
    print(f"Saved model → {config.MODEL_PATH}")

    cal_path = save_calibrator(calibrator)
    print(f"Saved calibrator → {cal_path}")

    with open(config.FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(list(config.MODEL_FEATURE_COLUMNS), f, indent=2)

    feature_stats = compute_feature_stats(X_train)
    with open(config.FEATURE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"Saved feature stats → {config.FEATURE_STATS_PATH}")

    metrics = {
        "dummy_val": base["metrics"]["dummy_val"],
        "dummy_test": base["metrics"]["dummy_test"],
        "logreg_val": base["metrics"]["logreg_val"],
        "logreg_test": base["metrics"]["logreg_test"],
        "rf_val": base["metrics"]["rf_val"],
        "rf_test": base["metrics"]["rf_test"],
        "lgbm_val": base["metrics"]["lgbm_val"],
        "lgbm_test": base["metrics"]["lgbm_test"],
        "baseline_val": xgb_default_val,
        "xgb_default_val": xgb_default_val,
        "xgb_default_test": xgb_default_test,
        "tuned_val": tuned_val,
        "tuned_test": tuned_test,
        "calibrated_val": cal_val_metrics,
        "calibrated_test": cal_test_metrics,
        "brier_raw_val": brier_raw_val,
        "brier_raw_test": brier_raw_test,
        "brier_calibrated_val": brier_cal_val,
        "brier_calibrated_test": brier_cal_test,
        "calibration_method": config.CALIBRATION_METHOD,
        "calibrated": True,
        "best_optuna_auc": float(study.best_value),
        "n_trials": n_trials,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "churn_rate_train": float(y_train.mean()),
        "feature_names": list(config.MODEL_FEATURE_COLUMNS),
        "best_params": study.best_params,
        "random_seed": config.RANDOM_SEED,
    }
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics → {config.METRICS_PATH}")

    write_data_dictionary()
    write_model_card(metrics)
    print("Updated research/data-dictionary.md and research/model-card.md")


if __name__ == "__main__":
    main()
