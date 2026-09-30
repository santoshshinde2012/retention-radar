"""Lock the published seed-42 numbers the articles cite: worked examples + ladder.

Reads committed (or CI-snapshotted) serve artifacts, never the smoke-retrain
outputs. Set ``RETENTION_RADAR_CANARY_DIR`` to a directory containing the frozen
``churn_xgb.joblib``, ``calibrator.joblib`` and ``metrics.json``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from retention_radar import config
from retention_radar.data.generate import arjun_profile, maya_profile
from retention_radar.serving.infer import load_model_bundle, predict_user
from retention_radar.serving.policy import decide, risk_band
from retention_radar.training.calibrate import load_calibrator

TOL_PROB = 0.005
TOL_AUC = 0.01


def _canary_dir() -> Path:
    override = os.environ.get("RETENTION_RADAR_CANARY_DIR", "").strip()
    return Path(override) if override else config.SEED_MODELS_DIR


@pytest.fixture(scope="module")
def canary():
    root = _canary_dir()
    model, cal, metrics = root / "churn_xgb.joblib", root / "calibrator.joblib", root / "metrics.json"
    if not model.exists() or not metrics.exists():
        pytest.skip(f"canary artifacts missing under {root}")
    return {
        "bundle": load_model_bundle(model),
        "calibrator": load_calibrator(cal) if cal.exists() else None,
        "metrics": json.loads(metrics.read_text(encoding="utf-8")),
    }


def _score(canary, profile):
    res = predict_user(profile, canary["bundle"], calibrator=canary["calibrator"])
    raw = float(res["churn_probability_raw"])
    cal = float(res["churn_probability_calibrated"])
    band = risk_band(cal)
    return raw, cal, band, decide(cal, canary["metrics"]["best_f1_threshold"], band, profile)


def test_maya_gets_the_limit_reset(canary):
    raw, cal, band, d = _score(canary, maya_profile())
    assert abs(raw - 0.554) <= TOL_PROB, raw
    assert abs(cal - 0.153) <= TOL_PROB, cal
    assert band == "medium"
    assert d["action"] == "limit_reset" and d["auto_action"] == "none"
    assert [c["playbook"] for c in d["candidates"]][:2] == ["limit_reset", "cancel_flow_discount"]


def test_arjun_is_left_alone(canary):
    raw, cal, band, d = _score(canary, arjun_profile())
    assert abs(cal - 0.023) <= TOL_PROB, cal
    assert band == "low" and d["action"] == "no_action"


def test_ladder_and_operating_point(canary):
    m = canary["metrics"]
    assert abs(m["logreg_test"]["roc_auc"] - 0.781) <= TOL_AUC
    assert abs(m["catboost_test"]["roc_auc"] - 0.770) <= TOL_AUC
    assert abs(m["tuned_test"]["roc_auc"] - 0.763) <= TOL_AUC
    assert m["logreg_test"]["roc_auc"] > m["tuned_test"]["roc_auc"]  # the honest finding
    assert m["calibration_method"] == "sigmoid"
    assert abs(m["brier_calibrated_test"] - 0.079) <= 0.002
    assert abs(m["best_f1_threshold"] - 0.14) <= 0.011
    assert abs(m["base_rate_test"] - 0.096) <= 0.002
