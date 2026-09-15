"""Tiny smoke test: generate → train (few trials) → infer / single_record Santosh."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config
from src.generate_data import generate_users, inject_santosh, santosh_profile
from src.features import prepare_xy, row_to_feature_frame
from src.ingest import validate_users
from src.train import main as train_main
from src.infer import predict_user, load_payload
from src.calibrate import load_calibrator


NEW_FEATURES = [
    "engagement_trend",
    "spend_usd_last_30d",
    "days_until_renewal",
    "agent_runs_last_30d",
    "ide_plugin_sessions_last_30d",
    "seat_utilization",
]


@pytest.fixture()
def tiny_data(tmp_path, monkeypatch):
    """Point config paths at a temp dir and write a small CSV + Santosh JSON."""
    raw = tmp_path / "raw"
    models = tmp_path / "models"
    artifacts = tmp_path / "artifacts"
    research = tmp_path / "research"
    guides = research
    schemas = tmp_path / "schemas"
    raw.mkdir()
    models.mkdir()
    artifacts.mkdir()
    guides.mkdir()
    schemas.mkdir()

    users_csv = raw / "users.csv"
    santosh_json = raw / "santosh_shinde.json"
    model_path = models / "churn_xgb.joblib"
    calibrator_path = models / "calibrator.joblib"
    metrics_path = models / "metrics.json"
    feature_names_path = models / "feature_names.json"
    feature_stats_path = models / "feature_stats.json"
    model_card = guides / "model-card.md"
    data_dict = guides / "data-dictionary.md"
    schema_path = schemas / "user_record.schema.json"

    real_schema = Path(config.PROJECT_ROOT) / "schemas" / "user_record.schema.json"
    if real_schema.exists():
        schema_path.write_text(real_schema.read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(config, "RAW_DIR", raw)
    monkeypatch.setattr(config, "USERS_CSV", users_csv)
    monkeypatch.setattr(config, "SANTOSH_JSON", santosh_json)
    monkeypatch.setattr(config, "MODELS_DIR", models)
    monkeypatch.setattr(config, "MODEL_PATH", model_path)
    monkeypatch.setattr(config, "CALIBRATOR_PATH", calibrator_path)
    monkeypatch.setattr(config, "METRICS_PATH", metrics_path)
    monkeypatch.setattr(config, "FEATURE_NAMES_PATH", feature_names_path)
    monkeypatch.setattr(config, "FEATURE_STATS_PATH", feature_stats_path)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(config, "RESEARCH_DIR", research)
    monkeypatch.setattr(config, "GUIDES_DIR", guides)
    monkeypatch.setattr(config, "MODEL_CARD_PATH", model_card)
    monkeypatch.setattr(config, "DATA_DICTIONARY_PATH", data_dict)
    monkeypatch.setattr(config, "USER_RECORD_SCHEMA_PATH", schema_path)
    monkeypatch.setattr(config, "EXTERNAL_DIR", tmp_path / "external")
    monkeypatch.setattr(
        config, "LAKEHOUSE_FEATURES_CSV", tmp_path / "external" / "churn_user_features.csv"
    )
    monkeypatch.setattr(
        config,
        "LAKEHOUSE_SANTOSH_JSON",
        tmp_path / "external" / "santosh_inference_record.json",
    )
    monkeypatch.setattr(config, "CHURN_DATA_SOURCE", "synthetic")
    monkeypatch.setenv("CHURN_DATA_SOURCE", "synthetic")
    monkeypatch.setattr(config, "N_OPTUNA_TRIALS", 3)
    monkeypatch.setenv("N_OPTUNA_TRIALS", "3")
    monkeypatch.setattr(config, "LATENCY_WARMUP", 2)
    monkeypatch.setattr(config, "LATENCY_RUNS", 5)

    df = generate_users(n=400, seed=42)
    df = inject_santosh(df)
    df.to_csv(users_csv, index=False)
    profile = {k: v for k, v in santosh_profile().items() if k != "churned"}
    santosh_json.write_text(json.dumps(profile), encoding="utf-8")

    return {
        "users_csv": users_csv,
        "santosh_json": santosh_json,
        "model_path": model_path,
        "calibrator_path": calibrator_path,
        "metrics_path": metrics_path,
        "feature_stats_path": feature_stats_path,
        "artifacts": artifacts,
        "model_card": model_card,
        "data_dict": data_dict,
    }


def test_new_features_in_config():
    for f in NEW_FEATURES:
        assert f in config.FEATURE_COLUMNS
        assert f in config.MODEL_FEATURE_COLUMNS or f == "plan_tier"
        assert f in config.FEATURE_DESCRIPTIONS
        assert f in config.INFERENCE_REQUIRED_KEYS


def test_validate_and_features(tiny_data):
    from src.ingest import load_users

    df = load_users()
    validate_users(df)
    X, y = prepare_xy(df)
    assert len(X) == len(y) == len(df)
    assert "plan_tier_code" in X.columns
    for f in NEW_FEATURES:
        assert f in X.columns
        assert f in df.columns


def test_train_and_infer_smoke(tiny_data):
    train_main(n_trials=3)
    assert tiny_data["model_path"].exists()
    assert tiny_data["calibrator_path"].exists()
    assert tiny_data["metrics_path"].exists()
    assert tiny_data["feature_stats_path"].exists()
    assert tiny_data["model_card"].exists()
    assert tiny_data["data_dict"].exists()

    metrics = json.loads(tiny_data["metrics_path"].read_text(encoding="utf-8"))
    for key in (
        "dummy_val",
        "dummy_test",
        "logreg_val",
        "logreg_test",
        "rf_val",
        "rf_test",
        "lgbm_val",
        "lgbm_test",
        "xgb_default_val",
        "xgb_default_test",
        "tuned_test",
        "brier_calibrated_test",
        "calibration_method",
        "calibrated",
    ):
        assert key in metrics, f"missing metrics key: {key}"

    stats = json.loads(tiny_data["feature_stats_path"].read_text(encoding="utf-8"))
    for f in NEW_FEATURES:
        assert f in stats
        assert "p50" in stats[f]

    bundle = load_payload(tiny_data["model_path"])
    calibrator = load_calibrator(tiny_data["calibrator_path"])
    assert calibrator is not None
    profile = json.loads(tiny_data["santosh_json"].read_text(encoding="utf-8"))
    result = predict_user(profile, bundle, calibrator=calibrator)

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert 0.0 <= result["churn_probability_raw"] <= 1.0
    assert result["churn_probability_calibrated"] is not None
    assert result["risk_band"] in {"low", "medium", "high"}
    assert len(result["top_features"]) >= 1


def test_single_record_packet(tiny_data):
    train_main(n_trials=3)
    from src.evaluate import main as eval_main
    from src.single_record import build_decision_packet

    eval_main()  # populates best_f1_threshold
    profile = json.loads(tiny_data["santosh_json"].read_text(encoding="utf-8"))
    bundle = load_payload(tiny_data["model_path"])
    calibrator = load_calibrator(tiny_data["calibrator_path"])
    packet = build_decision_packet(profile, model_bundle=bundle, calibrator=calibrator)

    required_keys = {
        "user_id",
        "user_name",
        "validation",
        "payload",
        "scoring",
        "explanation",
        "outliers",
        "cohort_compare",
        "hitl",
        "meta",
    }
    assert required_keys.issubset(packet.keys())
    assert packet["validation"]["ok"] is True
    scoring = packet["scoring"]
    for k in (
        "churn_probability_raw",
        "churn_probability_calibrated",
        "churn_probability",
        "risk_band",
        "best_f1_threshold",
        "half_threshold",
    ):
        assert k in scoring
    assert packet["hitl"]["action"] in {
        "monitor",
        "nurture / check-in",
        "retention outreach (human review)",
        "escalate",
    }
    assert packet["hitl"]["auto_action"] == "none"
    assert packet["meta"]["data_source"] == "synthetic"
    assert packet["meta"]["users_csv"] == str(tiny_data["users_csv"])
    assert len(packet["explanation"]["top_features"]) >= 1
    assert "engagement_trend" in packet["cohort_compare"]
    assert "seat_utilization" in packet["payload"]


def test_evaluate_artifacts(tiny_data):
    train_main(n_trials=3)
    from src.evaluate import main as eval_main
    from src.benchmark import main as bench_main

    eval_main()
    for name in (
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "threshold_f1.png",
        "confusion_matrix.png",
    ):
        assert (tiny_data["artifacts"] / name).exists(), name

    metrics = json.loads(tiny_data["metrics_path"].read_text(encoding="utf-8"))
    assert "evaluate_test" in metrics
    assert "best_f1_threshold" in metrics

    bench_main()
    metrics = json.loads(tiny_data["metrics_path"].read_text(encoding="utf-8"))
    assert "latency" in metrics
    assert "latency_ms_p50" in metrics["latency"]


def test_santosh_profile_keys():
    p = santosh_profile()
    required = {
        "user_id",
        "user_name",
        "days_since_signup",
        "sessions_last_7d",
        "sessions_last_30d",
        "avg_session_minutes",
        "models_used_count",
        "api_calls_last_30d",
        "tokens_consumed_last_30d",
        "tools_used_count",
        "failed_requests_rate",
        "support_tickets_last_90d",
        "plan_tier",
        "payment_failures_last_90d",
        "feature_adoption_score",
        "nps_score",
        "last_active_days_ago",
        "weekend_usage_ratio",
        *NEW_FEATURES,
    }
    assert required.issubset(p.keys())
    assert p["user_name"] == "Santosh Shinde"
    assert 0.0 <= p["seat_utilization"] <= 1.0
    assert p["engagement_trend"] > 0
