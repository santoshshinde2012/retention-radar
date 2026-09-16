"""FOSS production-shaped path: batch score, HITL log, thin FastAPI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from retention_radar import config
from retention_radar.data.generate import santosh_profile
from retention_radar.serving.batch_score import score_csv
from retention_radar.serving.hitl_log import (
    HITL_LOG_COLUMNS,
    append_hitl_row,
    row_from_packet,
    row_from_score_record,
)

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "batch" / "tiny_features.csv"
SANTOSH_JSON = config.PROJECT_ROOT / "data" / "raw" / "santosh_shinde.json"


@pytest.fixture()
def committed_models_ok():
    assert config.MODEL_PATH.exists(), "committed serve model required"
    assert config.CALIBRATOR_PATH.exists(), "committed calibrator required"


def test_batch_score_tiny_fixture(tmp_path, committed_models_ok):
    assert FIXTURE_CSV.exists()
    out_csv = tmp_path / "scores.csv"
    out_jsonl = tmp_path / "scores.jsonl"
    scores = score_csv(
        FIXTURE_CSV,
        out_csv=out_csv,
        out_jsonl=out_jsonl,
        model_path=config.MODEL_PATH,
        calibrator_path=config.CALIBRATOR_PATH,
    )
    assert len(scores) == 3
    assert out_csv.exists() and out_jsonl.exists()
    assert list(scores.columns) == [
        "user_id",
        "p_raw",
        "p_cal",
        "band",
        "hitl_action",
        "model_version",
        "scored_at",
        "auto_action",
    ]
    assert set(scores["auto_action"].unique()) == {"none"}
    assert scores["band"].isin(["low", "medium", "high"]).all()
    assert scores["user_id"].tolist() == ["u-0001", "u-0002", "u-0003"]
    line = out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assert rec["auto_action"] == "none"
    assert "p_cal" in rec


def test_hitl_log_append(tmp_path):
    log_path = tmp_path / "hitl_review_log.csv"
    packet = {
        "user_id": "santosh_shinde",
        "scoring": {
            "churn_probability_calibrated": 0.017,
            "risk_band": "low",
        },
        "hitl": {"action": "monitor", "auto_action": "none"},
    }
    row = row_from_packet(
        packet,
        reviewer="santosh",
        action_taken="monitor",
        notes="teaching append",
        timestamp="2026-09-16T06:00:00Z",
    )
    append_hitl_row(log_path, row)

    score_row = row_from_score_record(
        {
            "user_id": "u-0001",
            "p_cal": 0.12,
            "band": "low",
            "hitl_action": "monitor",
        },
        reviewer="santosh",
        action_taken="nurture / check-in",
        notes="overrode lightly",
        timestamp="2026-09-16T06:01:00Z",
    )
    append_hitl_row(log_path, score_row)

    with open(log_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == HITL_LOG_COLUMNS
    assert len(rows) == 2
    assert rows[0]["user_id"] == "santosh_shinde"
    assert rows[0]["action_suggested"] == "monitor"
    assert rows[1]["action_taken"] == "nurture / check-in"

    template = config.PROJECT_ROOT / "configs" / "templates" / "hitl_review_log.csv"
    schema = config.PROJECT_ROOT / "configs" / "hitl_review_log.schema.json"
    assert template.exists()
    assert schema.exists()
    header = template.read_text(encoding="utf-8").strip().split(",")
    assert header == HITL_LOG_COLUMNS


def test_fastapi_score_santosh(committed_models_ok, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retention_radar.serving import api as api_mod

    # Isolate prediction log under tmp
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    api_mod._load_serve_bundle.cache_clear()

    if SANTOSH_JSON.exists():
        payload = json.loads(SANTOSH_JSON.read_text(encoding="utf-8"))
    else:
        payload = {k: v for k, v in santosh_profile().items() if k != "churned"}

    client = TestClient(api_mod.app)
    resp = client.post("/v1/churn/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["auto_action"] == "none"
    assert body["band"] in {"low", "medium", "high"}
    assert "p_raw" in body and "p_cal" in body
    assert body["hitl_action"]
    assert body["model_version"]
    assert body.get("shap_top") is None

    resp_shap = client.post("/v1/churn/score?shap=true&log=false", json=payload)
    assert resp_shap.status_code == 200
    shap_body = resp_shap.json()
    assert shap_body["shap_top"] is not None
    assert len(shap_body["shap_top"]) >= 1

    # Default log=true should have written JSONL
    log_dir = tmp_path / "prediction_log"
    logs = list(log_dir.glob("scores_*.jsonl")) if log_dir.exists() else []
    assert logs, "expected prediction_log JSONL when log=true"
