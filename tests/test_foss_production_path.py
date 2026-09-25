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
        "rank",
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
    assert sorted(scores["user_id"]) == ["u-0001", "u-0002", "u-0003"]
    assert scores["rank"].tolist() == [1, 2, 3] and scores["p_cal"].is_monotonic_decreasing
    assert not out_csv.with_name("scores_rejected.csv").exists()
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


def test_fastapi_rejects_unknown_plan_tier(committed_models_ok, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retention_radar.serving import api as api_mod

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    api_mod._load_serve_bundle.cache_clear()
    payload = {k: v for k, v in santosh_profile().items() if k != "churned"}
    client = TestClient(api_mod.app)

    bad = client.post("/v1/churn/score?log=false", json={**payload, "plan_tier": "platinum"})
    assert bad.status_code == 422
    assert "plan_tier" in bad.text

    # Case / whitespace are normalised like lakehouse gold.
    ok = client.post("/v1/churn/score?log=false", json={**payload, "plan_tier": " Pro "})
    assert ok.status_code == 200, ok.text


def test_cohort_percentiles_fall_back_to_feature_stats(tmp_path):
    """Streamlit Cloud ships no users.csv — cohort compare uses training quantiles."""
    from retention_radar.serving.packet import cohort_percentiles

    stats = {
        "sessions_last_30d": {
            "mean": 20.0, "min": 0.0, "p01": 1.0, "p05": 4.0, "p25": 10.0,
            "p50": 20.0, "p75": 30.0, "p95": 45.0, "p99": 55.0, "max": 60.0,
        },
        "last_active_days_ago": {
            "mean": 3.0, "min": 0.0, "p01": 0.0, "p05": 0.0, "p25": 0.0,
            "p50": 1.0, "p75": 4.0, "p95": 12.0, "p99": 20.0, "max": 30.0,
        },
    }
    payload = {"sessions_last_30d": 25.0, "last_active_days_ago": 0.0, "spend_usd_last_30d": 5}
    out = cohort_percentiles(
        payload, users_csv=tmp_path / "missing.csv", feature_stats=stats
    )
    assert set(out) == {"sessions_last_30d", "last_active_days_ago"}
    assert out["sessions_last_30d"]["percentile"] == pytest.approx(62.5)
    assert out["sessions_last_30d"]["population_median"] == 20.0
    # Ties at zero take the top of the tie (≤ semantics, like the users.csv path).
    assert out["last_active_days_ago"]["percentile"] == pytest.approx(25.0)
    assert all(v["source"] == "feature_stats" for v in out.values())

    assert cohort_percentiles({"sessions_last_30d": 99.0}, users_csv=tmp_path / "x.csv",
                              feature_stats=stats)["sessions_last_30d"]["percentile"] == 100.0


def test_hitl_outcomes_join_and_summary(tmp_path):
    """Outcome write-back: review rows ⋈ later labels → per-band / per-action report."""
    from retention_radar.serving.outcomes import write_outcomes

    log_path = tmp_path / "hitl_review_log.csv"
    for uid, p, band, sugg, taken, ts in [
        ("u-1", 0.05, "low", "monitor", "monitor", "2026-09-16T06:00:00Z"),
        ("u-2", 0.45, "medium", "retention outreach (human review)",
         "retention outreach (human review)", "2026-09-16T06:00:00Z"),
        ("u-3", 0.70, "high", "escalate", "", "2026-09-16T06:00:00Z"),
        ("u-4", 0.40, "medium", "nurture / check-in", "monitor", "2026-09-20T00:00:00Z"),
        ("u-5", 0.10, "low", "monitor", "monitor", "2026-09-16T06:00:00Z"),
    ]:
        append_hitl_row(log_path, row_from_score_record(
            {"user_id": uid, "p_cal": p, "band": band, "hitl_action": sugg},
            reviewer="r", action_taken=taken, timestamp=ts,
        ))

    labels = tmp_path / "labels.csv"
    labels.write_text(
        "user_id,churned,observed_at\n"
        "u-1,0,2026-10-16\n"
        "u-2,1,2026-10-16\n"
        "u-3,1,2026-10-16\n"
        "u-4,1,2026-09-01\n"  # observed before the review → must not count
        "u-9,1,2026-10-16\n",  # never reviewed → ignored
        encoding="utf-8",
    )
    summary = write_outcomes(log_path, labels, out_csv=tmp_path / "outcomes.csv")

    assert summary["n_reviewed"] == 5
    assert summary["n_labelled"] == 3  # u-4 too early, u-5 unlabelled
    assert summary["observed_churn_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["reviewer_agreement_rate"] == pytest.approx(3 / 5)
    assert summary["auto_action"] == "none"
    by_band = {r["band"]: r for r in summary["by_band"]}
    assert by_band["low"]["n_reviewed"] == 2 and by_band["low"]["observed_churn_rate"] == 0.0
    assert by_band["high"]["observed_churn_rate"] == 1.0
    by_action = {r["action_taken"]: r for r in summary["by_action_taken"]}
    assert "(not recorded)" in by_action

    with open(tmp_path / "outcomes.csv", encoding="utf-8", newline="") as f:
        rows = {r["user_id"]: r for r in csv.DictReader(f)}
    assert rows["u-4"]["churned"] == "" and rows["u-2"]["churned"] in {"1", "1.0"}
    assert json.loads((tmp_path / "outcomes.json").read_text())["n_labelled"] == 3
