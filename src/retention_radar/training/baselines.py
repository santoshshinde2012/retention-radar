"""Honest non-XGBoost baselines + peer tree models.

Ladder in metrics.json:
  Dummy(prior) → LogReg → Random Forest → default XGB → Optuna XGB → LightGBM → CatBoost
  (+ calibrated XGB for serving)

XGBoost remains the teaching / Santosh hero. RF, LightGBM, and CatBoost are bake-off
peers (GBDT trilogy). There is no separate “simple rule” baseline in this repo.
"""

from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.retention_radar import config
from src.retention_radar.evaluation.metrics import classification_metrics


def train_dummy(X_train, y_train) -> DummyClassifier:
    """Most-frequent class prior; predict_proba uses class priors."""
    clf = DummyClassifier(strategy="prior", random_state=config.RANDOM_SEED)
    clf.fit(X_train, y_train)
    return clf


def train_logreg(X_train, y_train) -> Pipeline:
    """Standardized L2 logistic regression (CPU-fast, FOSS)."""
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    random_state=config.RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Balanced Random Forest mid-tier (little tuning, FOSS sklearn)."""
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=config.RANDOM_SEED,
        n_jobs=2,
    )
    clf.fit(X_train, y_train)
    return clf


def _scale_pos_weight(y) -> float:
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / max(pos, 1))


def train_lightgbm(X_train, y_train) -> LGBMClassifier:
    """Sane-default LightGBM peer booster (same seed / imbalance handling)."""
    spw = _scale_pos_weight(y_train)
    clf = LGBMClassifier(
        n_estimators=120,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        random_state=config.RANDOM_SEED,
        n_jobs=2,
        verbosity=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def train_catboost(X_train, y_train) -> CatBoostClassifier:
    """Sane-default CatBoost peer booster (same seed / imbalance handling as LightGBM)."""
    spw = _scale_pos_weight(y_train)
    clf = CatBoostClassifier(
        iterations=120,
        depth=4,
        learning_rate=0.08,
        subsample=0.9,
        bootstrap_type="Bernoulli",
        rsm=0.9,
        scale_pos_weight=spw,
        random_seed=config.RANDOM_SEED,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    clf.fit(X_train, y_train)
    return clf


def run_baselines(X_train, y_train, X_val, y_val, X_test, y_test) -> dict:
    """Fit Dummy, LogReg, RF, LightGBM, CatBoost; return metrics on val and test."""
    dummy = train_dummy(X_train, y_train)
    logreg = train_logreg(X_train, y_train)
    rf = train_random_forest(X_train, y_train)
    lgbm = train_lightgbm(X_train, y_train)
    catboost = train_catboost(X_train, y_train)

    def _pair(model):
        return (
            classification_metrics(y_val, model.predict_proba(X_val)[:, 1]),
            classification_metrics(y_test, model.predict_proba(X_test)[:, 1]),
        )

    dummy_val, dummy_test = _pair(dummy)
    logreg_val, logreg_test = _pair(logreg)
    rf_val, rf_test = _pair(rf)
    lgbm_val, lgbm_test = _pair(lgbm)
    catboost_val, catboost_test = _pair(catboost)

    return {
        "dummy": dummy,
        "logreg": logreg,
        "random_forest": rf,
        "lightgbm": lgbm,
        "catboost": catboost,
        "metrics": {
            "dummy_val": dummy_val,
            "dummy_test": dummy_test,
            "logreg_val": logreg_val,
            "logreg_test": logreg_test,
            "rf_val": rf_val,
            "rf_test": rf_test,
            "lgbm_val": lgbm_val,
            "lgbm_test": lgbm_test,
            "catboost_val": catboost_val,
            "catboost_test": catboost_test,
        },
    }
