"""FOSS production-shaped path: daily batch score, action log, thin FastAPI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from retention_radar import config
from retention_radar.data.generate import santosh_profile
from retention_radar.serving.action_log import (
    ACTION_LOG_COLUMNS,
    append_action_row,
    row_from_packet,
    row_from_score_record,
)
from retention_radar.serving.batch_score import SCORE_COLUMNS, score_csv

FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures" / "batch" / "tiny_features.csv"
SANTOSH_JSON = config.PROJECT_ROOT / "data" / "raw" / "subscribers" / "santosh.json"


@pytest.fixture()
def committed_models_ok():
    assert config.MODEL_PATH.exists(), "committed serve model required"
    assert config.CALIBRATOR_PATH.exists(), "committed calibrator required"


def test_batch_score_tiny_fixture(tmp_path, committed_models_ok):
    out_csv, out_jsonl = tmp_path / "scores.csv", tmp_path / "scores.jsonl"
    scores = score_csv(
        FIXTURE_CSV,
        out_csv=out_csv,
        out_jsonl=out_jsonl,
        model_path=config.MODEL_PATH,
        calibrator_path=config.CALIBRATOR_PATH,
    )
    assert len(scores) == 3
    assert out_csv.exists() and out_jsonl.exists()
    assert list(scores.columns) == SCORE_COLUMNS
    assert SCORE_COLUMNS[:6] == ["rank", "user_id", "p_raw", "p_cal", "band", "action"]
    assert set(scores["auto_action"].unique()) == {"none"}
    assert scores["band"].isin(["low", "medium", "high"]).all()
    assert sorted(scores["user_id"]) == ["u-0001", "u-0002", "u-0003"]
    assert scores["rank"].tolist() == [1, 2, 3] and scores["p_cal"].is_monotonic_decreasing
    assert not out_csv.with_name("scores_rejected.csv").exists()
    rec = json.loads(out_jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert rec["auto_action"] == "none" and "p_cal" in rec and "holdout" in rec


def test_action_log_append(tmp_path):
    log_path = tmp_path / "action_log.csv"
    packet = {
        "user_id": "sub_santosh",
        "scoring": {"churn_probability_calibrated": 0.153, "risk_band": "medium"},
        "decision": {"action": "limit_reset", "auto_action": "none", "holdout": False},
    }
    append_action_row(log_path, row_from_packet(
        packet, executed_by="lifecycle-tool", action_taken="limit_reset",
        notes="sent", timestamp="2026-09-28T09:00:00Z",
    ))
    append_action_row(log_path, row_from_score_record(
        {"user_id": "u-0001", "p_cal": 0.25, "band": "medium", "action": "holdout",
         "holdout": True, "would_have_sent": "limit_reset"},
        executed_by="none", action_taken="holdout", timestamp="2026-09-28T09:01:00Z",
    ))

    with open(log_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ACTION_LOG_COLUMNS
    assert len(rows) == 2
    assert rows[0]["user_id"] == "sub_santosh" and rows[0]["action_suggested"] == "limit_reset"
    assert rows[1]["holdout"] == "true" and rows[1]["would_have_sent"] == "limit_reset"

    template = config.PROJECT_ROOT / "configs" / "templates" / "action_log.csv"
    schema = json.loads((config.PROJECT_ROOT / "configs" / "action_log.schema.json").read_text())
    assert template.read_text(encoding="utf-8").strip().split(",") == ACTION_LOG_COLUMNS
    assert schema["required"] == ACTION_LOG_COLUMNS


def test_fastapi_score_santosh(committed_models_ok, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retention_radar.serving import api as api_mod

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    api_mod._load_serve_bundle.cache_clear()
    payload = json.loads(SANTOSH_JSON.read_text()) if SANTOSH_JSON.exists() else santosh_profile()

    client = TestClient(api_mod.app)
    resp = client.post("/v1/churn/score", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["auto_action"] == "none"
    assert body["band"] == "medium" and body["action"] == "limit_reset"
    assert body["expected_value_usd"] > 0 and body["holdout"] is False
    assert body["model_version"] and body.get("shap_top") is None

    shap_body = client.post("/v1/churn/score?shap=true&log=false", json=payload).json()
    assert shap_body["shap_top"] and len(shap_body["shap_top"]) >= 1

    log_dir = tmp_path / "prediction_log"
    assert list(log_dir.glob("scores_*.jsonl")), "expected prediction_log JSONL when log=true"


def test_fastapi_rejects_teams_plan_and_normalises_case(committed_models_ok, tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retention_radar.serving import api as api_mod

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    api_mod._load_serve_bundle.cache_clear()
    client = TestClient(api_mod.app)

    bad = client.post("/v1/churn/score?log=false", json={**santosh_profile(), "plan_tier": "teams"})
    assert bad.status_code == 422 and "plan_tier" in bad.text
    ok = client.post("/v1/churn/score?log=false", json={**santosh_profile(), "plan_tier": " Pro "})
    assert ok.status_code == 200, ok.text


def test_cohort_percentiles_fall_back_to_feature_stats(tmp_path):
    """Streamlit Cloud ships no renewals_t7.csv: cohort compare uses training quantiles."""
    from retention_radar.serving.packet import cohort_percentiles

    stats = {
        "active_days_28d": {
            "mean": 10.0, "min": 0.0, "p01": 0.0, "p05": 1.0, "p25": 5.0,
            "p50": 10.0, "p75": 15.0, "p95": 22.0, "p99": 25.0, "max": 28.0,
        },
        "limit_hits_14d": {
            "mean": 1.0, "min": 0.0, "p01": 0.0, "p05": 0.0, "p25": 0.0,
            "p50": 0.0, "p75": 2.0, "p95": 5.0, "p99": 9.0, "max": 20.0,
        },
    }
    payload = {"active_days_28d": 12.5, "limit_hits_14d": 0.0, "overage_usd_28d": 5}
    out = cohort_percentiles(payload, users_csv=tmp_path / "missing.csv", feature_stats=stats)
    assert set(out) == {"active_days_28d", "limit_hits_14d"}
    assert out["active_days_28d"]["percentile"] == pytest.approx(62.5)
    assert out["active_days_28d"]["population_median"] == 10.0
    # Ties at zero take the top of the tie (≤ semantics, like the CSV path).
    assert out["limit_hits_14d"]["percentile"] == pytest.approx(50.0)
    assert all(v["source"] == "feature_stats" for v in out.values())


def test_outcomes_join_summary_and_lift(tmp_path):
    """Outcome write-back: action rows ⋈ later outcomes → per-band report + lift vs holdout."""
    from retention_radar.serving.outcomes import write_outcomes

    log_path = tmp_path / "action_log.csv"
    rows = []
    # 100 treated with limit_reset (20 lapse) vs 100 holdout that would have got it (40 lapse).
    # (At 40 per group the same 20-point gap is not significant: the interval crosses zero.)
    for i in range(100):
        rows.append((f"t{i}", 0.25, "medium", "limit_reset", False, "", "lifecycle-tool", "limit_reset", int(i < 20)))
        rows.append((f"h{i}", 0.25, "medium", "holdout", True, "limit_reset", "none", "holdout", int(i < 40)))
    rows.append(("late", 0.25, "medium", "limit_reset", False, "", "lifecycle-tool", "limit_reset", 1))
    labels = ["user_id,churned,observed_at"]
    for uid, p, band, sugg, hold, would, by, taken, lapsed in rows:
        append_action_row(log_path, row_from_score_record(
            {"user_id": uid, "p_cal": p, "band": band, "action": sugg, "holdout": hold, "would_have_sent": would},
            executed_by=by, action_taken=taken, timestamp="2026-09-28T09:00:00Z",
        ))
        labels.append(f"{uid},{lapsed},{'2026-09-01' if uid == 'late' else '2026-10-05'}")
    (tmp_path / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")

    summary = write_outcomes(log_path, tmp_path / "labels.csv", out_csv=tmp_path / "outcomes.csv")
    assert summary["n_logged"] == 201 and summary["n_labelled"] == 200  # "late" observed before the action
    assert summary["auto_action"] == "none"
    lift = {r["playbook"]: r for r in summary["lift_vs_holdout"]}["limit_reset"]
    assert (lift["n_treated"], lift["n_holdout"]) == (100, 100)
    assert lift["lapse_rate_treated"] == 0.2 and lift["lapse_rate_holdout"] == 0.4
    assert lift["lift_pp"] == pytest.approx(20.0)
    assert lift["ci95_pp"][0] > 0 and lift["verdict"] == "lift"


def test_small_holdout_never_gets_a_verdict(tmp_path):
    """Two holdout rows with no lapses must not produce a zero-width 'conclusive' interval."""
    from retention_radar.serving.outcomes import write_outcomes

    log_path = tmp_path / "action_log.csv"
    labels = ["user_id,churned"]
    for i in range(36):
        append_action_row(log_path, row_from_score_record(
            {"user_id": f"t{i}", "p_cal": 0.2, "band": "medium", "action": "cancel_flow_discount"},
            executed_by="lifecycle-tool", action_taken="cancel_flow_discount", timestamp="2026-09-28T09:00:00Z",
        ))
        labels.append(f"t{i},{int(i < 5)}")
    for i in range(2):
        append_action_row(log_path, row_from_score_record(
            {"user_id": f"h{i}", "p_cal": 0.2, "band": "medium", "action": "holdout", "holdout": True,
             "would_have_sent": "cancel_flow_discount"},
            executed_by="none", action_taken="holdout", timestamp="2026-09-28T09:00:00Z",
        ))
        labels.append(f"h{i},0")
    (tmp_path / "labels.csv").write_text("\n".join(labels) + "\n", encoding="utf-8")
    summary = write_outcomes(log_path, tmp_path / "labels.csv", out_csv=tmp_path / "o.csv")
    row = {r["playbook"]: r for r in summary["lift_vs_holdout"]}["cancel_flow_discount"]
    assert row["verdict"].startswith("too small")
    lo, hi = row["ci95_pp"]
    assert hi - lo > 40  # honest width, not zero
