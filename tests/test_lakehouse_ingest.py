"""Lakehouse export ingest path (no Docker)."""
from __future__ import annotations

import pandas as pd
import pytest

from retention_radar import config
from retention_radar.data.ingest import (
    load_users,
    resolve_santosh_json,
    resolve_users_csv,
    sync_lakehouse_exports,
)


def test_resolve_prefers_external_when_present(tmp_path, monkeypatch):
    ext = tmp_path / "external"
    ext.mkdir()
    raw = tmp_path / "raw"
    raw.mkdir()
    # minimal valid frame
    cols = config.ID_COLUMNS + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    row = {c: 0 for c in cols}
    row.update(
        {
            "user_id": "u-1",
            "user_name": "T",
            "plan_tier": "pro",
            "churned": 0,
            "avg_session_minutes": 1.0,
            "nps_score": 5.0,
            "feature_adoption_score": 0.5,
            "seat_utilization": 0.5,
            "failed_requests_rate": 0.1,
            "weekend_usage_ratio": 0.2,
            "engagement_trend": 1.0,
            "days_since_signup": 10,
            "days_until_renewal": 5,
        }
    )
    # need enough rows + churn band — skip full validate by testing resolve only
    monkeypatch.setattr(config, "EXTERNAL_DIR", ext)
    monkeypatch.setattr(config, "LAKEHOUSE_FEATURES_CSV", ext / "churn_user_features.csv")
    monkeypatch.setattr(config, "LAKEHOUSE_SANTOSH_JSON", ext / "santosh_inference_record.json")
    monkeypatch.setattr(config, "USERS_CSV", raw / "users.csv")
    monkeypatch.setattr(config, "SANTOSH_JSON", raw / "santosh_shinde.json")
    monkeypatch.setattr(config, "CHURN_DATA_SOURCE", "auto")

    (ext / "churn_user_features.csv").write_text("user_id\n1\n")
    (ext / "santosh_inference_record.json").write_text("{}")
    assert resolve_users_csv() == ext / "churn_user_features.csv"
    assert resolve_santosh_json() == ext / "santosh_inference_record.json"

    monkeypatch.setattr(config, "CHURN_DATA_SOURCE", "synthetic")
    (raw / "users.csv").write_text("user_id\n1\n")
    assert resolve_users_csv() == raw / "users.csv"


def test_sync_lakehouse_exports(tmp_path, monkeypatch):
    src = tmp_path / "export"
    src.mkdir()
    dest = tmp_path / "external"
    (src / "churn_user_features.csv").write_text("user_id,churned\nu-1,0\n")
    (src / "santosh_inference_record.json").write_text('{"user_id":"u-1"}\n')
    monkeypatch.setattr(config, "EXTERNAL_DIR", dest)
    csv_p, json_p = sync_lakehouse_exports(src, dest)
    assert csv_p.exists() and json_p.exists()
    assert "u-1" in csv_p.read_text()


def test_drift_cli_defaults_to_active_users_table():
    import inspect

    from retention_radar.serving import drift as drift_mod
    from retention_radar.serving import packet as packet_mod

    assert "resolve_users_csv()" in inspect.getsource(drift_mod.main)
    assert "resolve_users_csv()" in inspect.getsource(packet_mod.cohort_percentiles)


def _gold_frame(n: int = 20) -> pd.DataFrame:
    cols = config.ID_COLUMNS + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    rows = []
    for i in range(n):
        row = {c: 0 for c in cols}
        row.update(
            {
                "user_id": f"u-{i:04d}",
                "user_name": "Santosh Shinde" if i == 0 else f"User {i}",
                "plan_tier": "pro" if i % 2 == 0 else "free",
                "churned": 1 if i % 8 == 0 else 0,
                "avg_session_minutes": 12.0,
                "nps_score": 7.0,
                "feature_adoption_score": 0.5,
                "seat_utilization": 0.4,
                "failed_requests_rate": 0.05,
                "weekend_usage_ratio": 0.2,
                "engagement_trend": 1.0,
                "days_since_signup": 100,
                "days_until_renewal": 20,
                "sessions_last_7d": 5,
                "sessions_last_30d": 20,
                "models_used_count": 4,
                "api_calls_last_30d": 200,
                "tokens_consumed_last_30d": 50000,
                "tools_used_count": 3,
                "support_tickets_last_90d": 1,
                "payment_failures_last_90d": 0,
                "last_active_days_ago": 2,
                "spend_usd_last_30d": 12.0,
                "agent_runs_last_30d": 10,
                "ide_plugin_sessions_last_30d": 8,
                "city": "Pune",
                "feature_as_of": "2024-03-02",
                "built_at": "2024-03-02T00:00:00Z",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_users_from_lakehouse_shaped_export(tmp_path, monkeypatch):
    ext = tmp_path / "external"
    ext.mkdir()
    csv_path = ext / "churn_user_features.csv"
    _gold_frame().to_csv(csv_path, index=False)
    monkeypatch.setattr(config, "EXTERNAL_DIR", ext)
    monkeypatch.setattr(config, "LAKEHOUSE_FEATURES_CSV", csv_path)
    monkeypatch.setattr(config, "CHURN_DATA_SOURCE", "lakehouse")

    df = load_users()
    assert len(df) == 20
    assert "city" not in df.columns
    assert "feature_as_of" not in df.columns
    assert set(df["churned"].unique()).issubset({0, 1})

    from retention_radar.features.transform import prepare_xy

    X, y = prepare_xy(df)
    assert list(X.columns) == list(config.MODEL_FEATURE_COLUMNS)
    assert len(X) == len(y) == 20


def test_unknown_plan_tier_fails_loud():
    from retention_radar.features.transform import encode_plan_tier

    df = _gold_frame(n=2)
    df.loc[0, "plan_tier"] = "gold"
    with pytest.raises(ValueError, match="Unknown plan_tier"):
        encode_plan_tier(df)
