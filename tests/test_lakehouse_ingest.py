"""Lakehouse export ingest path (no Docker)."""
from __future__ import annotations

import pandas as pd
import pytest

from retention_radar import config
from retention_radar.data.generate import generate_renewals, maya_profile
from retention_radar.data.ingest import (
    load_users,
    resolve_hero_json,
    resolve_users_csv,
    sync_lakehouse_exports,
)


def test_resolve_prefers_external_when_present(tmp_path, monkeypatch):
    ext, raw = tmp_path / "external", tmp_path / "raw"
    ext.mkdir()
    (raw / "subscribers").mkdir(parents=True)
    monkeypatch.setattr(config, "EXTERNAL_DIR", ext)
    monkeypatch.setattr(config, "LAKEHOUSE_FEATURES_CSV", ext / "churn_user_features.csv")
    monkeypatch.setattr(config, "LAKEHOUSE_HERO_JSON", ext / "hero_inference_record.json")
    monkeypatch.setattr(config, "USERS_CSV", raw / "renewals_t7.csv")
    monkeypatch.setattr(config, "HERO_DIR", raw / "subscribers")
    monkeypatch.setattr(config, "CHURN_DATA_SOURCE", "auto")

    (ext / "churn_user_features.csv").write_text("user_id\n1\n")
    (ext / "hero_inference_record.json").write_text("{}")
    assert resolve_users_csv() == ext / "churn_user_features.csv"
    assert resolve_hero_json() == ext / "hero_inference_record.json"
    assert resolve_hero_json("arjun") == raw / "subscribers" / "arjun.json"  # only the default hero comes from gold

    monkeypatch.setattr(config, "CHURN_DATA_SOURCE", "synthetic")
    assert resolve_users_csv() == raw / "renewals_t7.csv"
    assert resolve_hero_json() == raw / "subscribers" / "maya.json"
    with pytest.raises(ValueError, match="Unknown worked example"):
        resolve_hero_json("santosh")


def test_sync_lakehouse_exports(tmp_path, monkeypatch):
    src, dest = tmp_path / "export", tmp_path / "external"
    src.mkdir()
    (src / "churn_user_features.csv").write_text("user_id,churned\nu-1,0\n")
    (src / "hero_inference_record.json").write_text('{"user_id":"u-1"}\n')
    monkeypatch.setattr(config, "EXTERNAL_DIR", dest)
    csv_p, json_p = sync_lakehouse_exports(src, dest)
    assert csv_p.exists() and json_p.exists()
    assert "u-1" in csv_p.read_text()


def test_drift_cli_defaults_to_active_table():
    import inspect

    from retention_radar.serving import drift as drift_mod
    from retention_radar.serving import packet as packet_mod

    assert "resolve_users_csv()" in inspect.getsource(drift_mod.main)
    assert "resolve_users_csv()" in inspect.getsource(packet_mod.cohort_percentiles)


def _gold_frame(n: int = 400) -> pd.DataFrame:
    """Gold export shape: the model table plus lake metadata and routing columns."""
    df = generate_renewals(n=n, seed=7)
    df["city"] = "Pune"
    df["feature_as_of"] = df["as_of_date"]
    df["built_at"] = "2026-09-28T00:00:00Z"
    return df[df["route"] == config.ROUTE_MODEL].reset_index(drop=True)


def test_load_users_from_lakehouse_shaped_export(tmp_path, monkeypatch):
    ext = tmp_path / "external"
    ext.mkdir()
    csv_path = ext / "churn_user_features.csv"
    _gold_frame().to_csv(csv_path, index=False)
    monkeypatch.setattr(config, "EXTERNAL_DIR", ext)
    monkeypatch.setattr(config, "LAKEHOUSE_FEATURES_CSV", csv_path)
    monkeypatch.setattr(config, "CHURN_DATA_SOURCE", "lakehouse")

    df = load_users()
    for dropped in ("city", "feature_as_of", "built_at", "as_of_date", "renewal_date", "outcome", "route"):
        assert dropped not in df.columns
    assert set(df["churned"].unique()).issubset({0, 1})

    from retention_radar.features.transform import prepare_xy

    X, y = prepare_xy(df)
    assert list(X.columns) == list(config.MODEL_FEATURE_COLUMNS)
    assert len(X) == len(y) == len(df)


def test_unknown_plan_tier_fails_loud():
    from retention_radar.features.transform import encode_plan_tier

    df = pd.DataFrame([maya_profile(), {**maya_profile(), "plan_tier": "pro_max"}])
    with pytest.raises(ValueError, match="Unknown plan_tier"):
        encode_plan_tier(df)


def test_validate_rejects_seven_day_count_above_28_day_count():
    from retention_radar.data.ingest import validate_users

    df = _gold_frame()[config.ID_COLUMNS + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]]
    df.loc[0, ["active_days_7d", "active_days_28d"]] = [7, 2]
    with pytest.raises(ValueError, match="active_days_7d"):
        validate_users(df)
