"""Tiny smoke test: generate → train (few trials) → evaluate → packets for the worked examples."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retention_radar import config
from retention_radar.cli.train import main as train_main
from retention_radar.data.generate import (
    arjun_profile,
    generate_renewals,
    santosh_profile,
    model_table,
)
from retention_radar.data.ingest import validate_users
from retention_radar.features.transform import prepare_xy
from retention_radar.serving.infer import load_model_bundle, predict_user
from retention_radar.training.calibrate import load_calibrator

AI_TOOL_FEATURES = [
    "limit_hits_14d",
    "cheap_model_share_28d",
    "overage_toggled_off",
    "suggestion_accept_rate_28d",
    "agent_task_success_rate",
    "first_renewal_after_pricing_change",
]


@pytest.fixture()
def tiny_data(tmp_path, monkeypatch):
    """Point config paths at a temp dir and write a small renewal table + hero JSONs."""
    raw = tmp_path / "raw"
    models = tmp_path / "models"
    artifacts = tmp_path / "artifacts"
    guides = tmp_path / "research"
    schemas = tmp_path / "configs" / "schemas"
    for d in (raw, models, artifacts, guides, schemas):
        d.mkdir(parents=True)

    users_csv = raw / "renewals_t7.csv"
    hero_dir = raw / "subscribers"
    hero_dir.mkdir()
    schema_path = schemas / "user_record.schema.json"
    real_schema = Path(config.PROJECT_ROOT) / "configs" / "schemas" / "user_record.schema.json"
    schema_path.write_text(real_schema.read_text(encoding="utf-8"), encoding="utf-8")

    paths = {
        "RAW_DIR": raw,
        "USERS_CSV": users_csv,
        "HERO_DIR": hero_dir,
        "MODELS_DIR": models,
        "MODEL_PATH": models / "churn_xgb.joblib",
        "CALIBRATOR_PATH": models / "calibrator.joblib",
        "METRICS_PATH": models / "metrics.json",
        "FEATURE_NAMES_PATH": models / "feature_names.json",
        "FEATURE_STATS_PATH": models / "feature_stats.json",
        "ARTIFACTS_DIR": artifacts,
        "RESEARCH_DIR": guides,
        "GUIDES_DIR": guides,
        "MODEL_CARD_PATH": guides / "model-card.md",
        "DATA_DICTIONARY_PATH": guides / "data-dictionary.md",
        "USER_RECORD_SCHEMA_PATH": schema_path,
        "EXTERNAL_DIR": tmp_path / "external",
        "LAKEHOUSE_FEATURES_CSV": tmp_path / "external" / "churn_user_features.csv",
        "LAKEHOUSE_HERO_JSON": tmp_path / "external" / "hero_inference_record.json",
        "CHURN_DATA_SOURCE": "synthetic",
        "N_OPTUNA_TRIALS": 3,
        "LATENCY_WARMUP": 2,
        "LATENCY_RUNS": 5,
    }
    for k, v in paths.items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.setenv("CHURN_DATA_SOURCE", "synthetic")
    monkeypatch.setenv("N_OPTUNA_TRIALS", "3")

    model_table(generate_renewals(n=1500, seed=42)).to_csv(users_csv, index=False)
    (hero_dir / "santosh.json").write_text(json.dumps(santosh_profile()), encoding="utf-8")
    (hero_dir / "arjun.json").write_text(json.dumps(arjun_profile()), encoding="utf-8")
    return {k.lower(): v for k, v in paths.items() if isinstance(v, Path)} | {"santosh_json": hero_dir / "santosh.json"}


def test_ai_tool_features_in_config():
    for f in AI_TOOL_FEATURES:
        assert f in config.FEATURE_COLUMNS
        assert f in config.MODEL_FEATURE_COLUMNS
        assert f in config.FEATURE_DESCRIPTIONS
        assert f in config.INFERENCE_REQUIRED_KEYS
    assert len(config.MODEL_FEATURE_COLUMNS) == 22
    assert len(config.INFERENCE_REQUIRED_KEYS) == 24


def test_generator_routes_involuntary_and_scheduled_cancels_out():
    r = generate_renewals(n=3000, seed=42)
    assert set(r["route"]) == {config.ROUTE_MODEL, config.ROUTE_DUNNING, config.ROUTE_CANCEL_FLOW}
    dunning = r[r["route"] == config.ROUTE_DUNNING]
    assert (dunning["outcome"] == config.OUTCOME_INVOLUNTARY).all()
    assert (dunning["churned"] == 0).all()  # a failed card is not a voluntary lapse
    cancel_flow = r[r["route"] == config.ROUTE_CANCEL_FLOW]
    assert (cancel_flow["churned"] == 1).all()
    t = model_table(r)
    assert set(t.columns) == set(config.ID_COLUMNS + config.FEATURE_COLUMNS + [config.TARGET_COLUMN])
    assert 0.05 < t["churned"].mean() < 0.15
    # Features are as of T-7: first renewals lapse more than year-old subscriptions.
    assert r.loc[r.renewals_completed == 0, "churned"].mean() > r.loc[r.renewals_completed >= 12, "churned"].mean()


def test_validate_and_features(tiny_data):
    from retention_radar.data.ingest import load_users

    df = load_users()
    validate_users(df)
    X, y = prepare_xy(df)
    assert len(X) == len(y) == len(df)
    assert list(X.columns) == config.MODEL_FEATURE_COLUMNS


def test_train_evaluate_and_packet(tiny_data):
    train_main(n_trials=3)
    for key in ("model_path", "calibrator_path", "metrics_path", "feature_stats_path", "model_card_path", "data_dictionary_path"):
        assert tiny_data[key].exists(), key

    metrics = json.loads(tiny_data["metrics_path"].read_text(encoding="utf-8"))
    for key in (
        "dummy_test", "logreg_test", "rf_test", "lgbm_test", "catboost_test",
        "xgb_default_test", "tuned_test", "brier_calibrated_test", "calibration_method",
    ):
        assert key in metrics, key

    from retention_radar.cli.benchmark import main as bench_main
    from retention_radar.cli.evaluate import main as eval_main
    from retention_radar.serving.packet import build_decision_packet

    eval_main()
    for name in ("roc_curve.png", "pr_curve.png", "calibration_curve.png", "threshold_f1.png", "confusion_matrix.png"):
        assert (tiny_data["artifacts_dir"] / name).exists(), name
    metrics = json.loads(tiny_data["metrics_path"].read_text(encoding="utf-8"))
    for key in ("best_f1_threshold", "capture_at_top_10pct", "precision_at_top_10pct", "policy_on_test"):
        assert key in metrics, key
    assert sum(metrics["policy_on_test"]["actions"].values()) == metrics["n_test"]

    bench_main()
    assert "latency_ms_p50" in json.loads(tiny_data["metrics_path"].read_text())["latency"]

    from retention_radar.cli.analysis import main as analysis_main

    analysis = analysis_main()
    assert (tiny_data["artifacts_dir"] / "analysis.json").exists()
    assert analysis["split"]["lapses"]["test"] > 0
    assert set(analysis["holdout"]["power"]) == {"limit_reset", "cancel_flow_discount"}
    assert analysis["calibration"]["distinct_values_platt"] >= analysis["calibration"]["distinct_values_isotonic"]

    profile = json.loads(tiny_data["santosh_json"].read_text(encoding="utf-8"))
    bundle = load_model_bundle(tiny_data["model_path"])
    calibrator = load_calibrator(tiny_data["calibrator_path"])
    result = predict_user(profile, bundle, calibrator=calibrator)
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_probability_calibrated"] is not None

    packet = build_decision_packet(profile, model_bundle=bundle, calibrator=calibrator)
    assert {"validation", "scoring", "explanation", "cohort_compare", "decision", "meta"} <= set(packet)
    assert packet["validation"]["ok"] is True
    assert packet["decision"]["auto_action"] == "none"
    assert packet["decision"]["action"] in {"no_action", "holdout", *config.PLAYBOOKS}
    assert packet["meta"]["users_csv"] == str(tiny_data["users_csv"])
    assert "limit_hits_14d" in packet["cohort_compare"]


def test_worked_example_profiles_match_contract():
    for fn in (santosh_profile, arjun_profile):
        p = fn()
        assert set(p) == set(config.INFERENCE_REQUIRED_KEYS)
        assert "churned" not in p
        assert p["active_days_7d"] <= p["active_days_28d"]
        assert p["engagement_trend"] == pytest.approx(
            p["active_days_7d"] / max(1.0, p["active_days_28d"] / 4.0), abs=1e-4
        )
