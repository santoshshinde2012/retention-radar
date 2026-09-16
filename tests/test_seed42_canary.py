"""Lock published seed-42 Santosh scores + ladder AUCs (teaching Medium contract).

Reads committed (or CI-snapshotted) serve artifacts — never the smoke-retrain
outputs. Set ``RETENTION_RADAR_CANARY_DIR`` to a directory containing the
frozen ``churn_xgb.joblib``, ``calibrator.joblib``, and ``metrics.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from retention_radar import config
from retention_radar.data.generate import santosh_profile
from retention_radar.serving.infer import load_payload, predict_user
from retention_radar.serving.policy import hitl_action, risk_band
from retention_radar.training.calibrate import load_calibrator


TOL_PROB = 0.005
TOL_AUC = 0.01


def _canary_dir() -> Path:
    override = os.environ.get("RETENTION_RADAR_CANARY_DIR", "").strip()
    if override:
        return Path(override)
    return config.SEED_MODELS_DIR


@pytest.fixture(scope="module")
def canary_paths():
    root = _canary_dir()
    model = root / "churn_xgb.joblib"
    cal = root / "calibrator.joblib"
    metrics = root / "metrics.json"
    if not model.exists() or not metrics.exists():
        pytest.skip(f"canary artifacts missing under {root}")
    return {"model": model, "calibrator": cal, "metrics": metrics, "root": root}


def test_seed42_santosh_scores(canary_paths):
    bundle = load_payload(canary_paths["model"])
    calibrator = None
    if canary_paths["calibrator"].exists():
        calibrator = load_calibrator(canary_paths["calibrator"])

    profile = {k: v for k, v in santosh_profile().items() if k != "churned"}
    result = predict_user(profile, bundle, calibrator=calibrator)

    raw = float(result["churn_probability_raw"])
    cal = result.get("churn_probability_calibrated")
    assert cal is not None, "expected calibrated probability from seed-42 bundle"
    cal_f = float(cal)

    assert abs(raw - 0.043) <= TOL_PROB, f"raw={raw}"
    assert abs(cal_f - 0.017) <= TOL_PROB, f"cal={cal_f}"

    band = risk_band(cal_f)
    assert band == "low", band
    hitl = hitl_action(cal_f, threshold=0.34, band=band)
    assert hitl["auto_action"] == "none"
    assert hitl["action"] == "monitor"


def test_seed42_ladder_metrics(canary_paths):
    metrics = json.loads(canary_paths["metrics"].read_text(encoding="utf-8"))

    logreg = float(metrics["logreg_test"]["roc_auc"])
    catboost = float(metrics["catboost_test"]["roc_auc"])
    tuned = float(metrics["tuned_test"]["roc_auc"])

    assert abs(logreg - 0.872) <= TOL_AUC, f"logreg_test={logreg}"
    assert abs(catboost - 0.872) <= TOL_AUC, f"catboost_test={catboost}"
    assert abs(tuned - 0.870) <= TOL_AUC, f"tuned_test={tuned}"
