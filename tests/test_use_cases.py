"""Golden use cases: every scenario and every invalid record through every serve surface.

Data: data/use_cases/ (seed-42 holdout rows; see its README). Expected results are
recorded in personas.json from the committed bundle.
"""

from __future__ import annotations

import csv
import json

import pandas as pd
import pytest

from retention_radar import config
from retention_radar.data.use_cases import USE_CASE_DIR, check_use_cases, load_manifest
from retention_radar.serving.batch_score import score_csv
from retention_radar.serving.hitl_log import (
    append_hitl_row,
    row_from_packet,
    rows_from_decisions,
)
from retention_radar.serving.outcomes import write_outcomes
from retention_radar.serving.packet import build_decision_packet
from retention_radar.serving.policy import HOLD_ACTION

MANIFEST = USE_CASE_DIR / "personas.json"
pytestmark = pytest.mark.skipif(
    not MANIFEST.exists() or config.MODEL_PATH.parent != config.SEED_MODELS_DIR,
    reason="needs data/use_cases/ and the committed seed-42 bundle",
)


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(MANIFEST)


def _ids(items):
    return [i["id"] for i in items]


def test_pack_matches_committed_bundle():
    assert check_use_cases() == []


def test_pack_covers_every_band_and_action(manifest):
    got = {(p["expected"]["band"], p["expected"]["hitl_action"]) for p in manifest["personas"]}
    assert {b for b, _ in got} == {"low", "medium", "high"}
    assert {a for _, a in got} == {
        "monitor",
        "nurture / check-in",
        "retention outreach (human review)",
        "escalate",
    }
    assert ("low", "nurture / check-in") in got and ("medium", "nurture / check-in") in got
    for p in manifest["personas"]:
        assert "churned" not in p["record"]
        assert set(p["record"]) == set(config.INFERENCE_REQUIRED_KEYS)


@pytest.mark.parametrize("pid", _ids(load_manifest(MANIFEST)["personas"]) if MANIFEST.exists() else [])
def test_packet_per_scenario(manifest, pid):
    p = next(x for x in manifest["personas"] if x["id"] == pid)
    packet = build_decision_packet(p["record"])
    assert packet["validation"]["ok"], packet["validation"]
    assert packet["scoring"]["risk_band"] == p["expected"]["band"]
    assert packet["hitl"]["action"] == p["expected"]["hitl_action"]
    assert packet["scoring"]["churn_probability"] == pytest.approx(p["expected"]["p_cal"], abs=1e-6)
    assert packet["hitl"]["auto_action"] == "none"


@pytest.mark.parametrize("iid", _ids(load_manifest(MANIFEST)["invalid_records"]) if MANIFEST.exists() else [])
def test_invalid_record_is_held_not_scored(manifest, iid, tmp_path):
    case = next(x for x in manifest["invalid_records"] if x["id"] == iid)
    packet = build_decision_packet(case["record"])
    assert packet["validation"]["ok"] is False
    assert any(case["error_contains"] in e for e in packet["validation"]["errors"])
    assert packet["scoring"] is None
    assert packet["hitl"]["action"] == HOLD_ACTION
    assert packet["hitl"]["blocked_by_validation"] is True
    with pytest.raises(ValueError, match="held by validation"):
        row_from_packet(packet, reviewer="x", action_taken="monitor")


def test_single_record_cli_fails_loud_on_invalid(tmp_path):
    from retention_radar.serving.packet import main as packet_main

    out = tmp_path / "p.json"
    with pytest.raises(SystemExit) as exc:
        packet_main(["--json", str(USE_CASE_DIR / "invalid" / "rate_out_of_range.json"), "--out", str(out)])
    assert exc.value.code == 1
    assert json.loads(out.read_text())["hitl"]["action"] == HOLD_ACTION


def test_weekly_batch_queue_and_rejects(manifest, tmp_path):
    out = tmp_path / "queue.csv"
    scores = score_csv(USE_CASE_DIR / "weekly_batch.csv", out_csv=out, scored_at=manifest["calendar"]["scored_at"])
    wb = manifest["weekly_batch"]
    assert len(scores) == wb["valid_rows"]
    assert scores["hitl_action"].value_counts().sort_index().to_dict() == wb["expected_actions"]
    assert scores["rank"].tolist() == list(range(1, len(scores) + 1))
    assert scores["p_cal"].is_monotonic_decreasing
    for _, plateau in scores.groupby("p_cal"):  # raw risk orders ties inside a plateau
        assert plateau["p_raw"].is_monotonic_decreasing
    assert (scores["auto_action"] == "none").all()
    by_uid = scores.set_index("user_id")
    for p in manifest["personas"]:
        row = by_uid.loc[p["record"]["user_id"]]
        assert (row["band"], row["hitl_action"]) == (p["expected"]["band"], p["expected"]["hitl_action"])
    rejects = pd.read_csv(out.with_name("queue_rejected.csv"))
    assert len(rejects) == wb["invalid_rows"]
    for case in manifest["invalid_records"]:
        errs = rejects.set_index("user_id").loc[case["record"]["user_id"], "errors"]
        assert case["error_contains"] in errs


def test_reviews_then_outcomes_close_the_loop(manifest, tmp_path):
    queue = tmp_path / "queue.csv"
    score_csv(USE_CASE_DIR / "weekly_batch.csv", out_csv=queue, scored_at=manifest["calendar"]["scored_at"])
    with open(queue, encoding="utf-8", newline="") as f:
        scores = list(csv.DictReader(f))
    with open(USE_CASE_DIR / "review_decisions.csv", encoding="utf-8", newline="") as f:
        decisions = list(csv.DictReader(f))
    rows, problems = rows_from_decisions(scores, decisions)
    assert problems == [] and len(rows) == len(decisions)
    log = tmp_path / "hitl_review_log.csv"
    for r in rows:
        append_hitl_row(log, r)

    override = next(p for p in manifest["personas"] if p["id"] == "borderline_friction")
    logged = next(r for r in rows if r["user_id"] == override["record"]["user_id"])
    assert logged["action_suggested"] == "nurture / check-in"  # from the queue, not the file
    assert logged["action_taken"] == "retention outreach (human review)"

    summary = write_outcomes(log, USE_CASE_DIR / "labels_day30.csv", out_csv=tmp_path / "outcomes.csv")
    assert summary["n_reviewed"] == len(decisions)
    assert summary["n_labelled"] == len(decisions)  # labels observed after the review date
    assert 0 < summary["reviewer_agreement_rate"] < 1
    churn = {b["band"]: b["observed_churn_rate"] for b in summary["by_band"]}
    assert churn["high"] > churn["low"]


def test_decisions_cannot_reference_unscored_users():
    rows, problems = rows_from_decisions(
        [{"user_id": "u1", "p_cal": 0.5, "band": "medium", "hitl_action": "escalate"}],
        [{"user_id": "ghost", "reviewer": "r", "action_taken": "monitor"}, {"user_id": "u1", "reviewer": "", "action_taken": "x"}],
    )
    assert rows == [] and len(problems) == 2


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retention_radar.serving import api as api_mod

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    api_mod._load_serve_bundle.cache_clear()
    return TestClient(api_mod.app)


def test_api_scores_every_scenario(client, manifest):
    for p in manifest["personas"]:
        r = client.post("/v1/churn/score?log=false", json=p["record"])
        assert r.status_code == 200, r.text
        body = r.json()
        assert (body["band"], body["hitl_action"]) == (p["expected"]["band"], p["expected"]["hitl_action"])
        assert body["p_cal"] == pytest.approx(p["expected"]["p_cal"], abs=1e-6)
        assert body["rationale"] and body["best_f1_threshold"] == manifest["best_f1_threshold"]
        assert body["auto_action"] == "none" and body["shap_top"] is None


def test_api_rejects_invalid_records(client, manifest):
    for case in manifest["invalid_records"]:
        r = client.post("/v1/churn/score?log=false", json=case["record"])
        assert r.status_code == 422, (case["id"], r.text)
        assert case["error_contains"] in r.text


def test_api_batch_and_review_flow(client, manifest, tmp_path):
    batch = pd.read_csv(USE_CASE_DIR / "weekly_batch.csv", dtype={"user_id": str, "user_name": str, "plan_tier": str})
    records = [{k: v for k, v in r.items() if pd.notna(v)} for r in batch.to_dict(orient="records")]
    r = client.post("/v1/churn/batch", json={"records": records})
    assert r.status_code == 200, r.text
    body = r.json()
    wb = manifest["weekly_batch"]
    assert len(body["queue"]) == wb["valid_rows"] and len(body["rejected"]) == wb["invalid_rows"]
    assert [q["rank"] for q in body["queue"][:3]] == [1, 2, 3]
    assert body["queue"][0]["hitl_action"] == "escalate"

    gone = next(p for p in manifest["personas"] if p["id"] == "gone_dark")["record"]["user_id"]
    rv = client.post("/v1/churn/reviews", json={"user_id": gone, "reviewer": "cs", "action_taken": "escalate", "notes": "called"})
    assert rv.status_code == 200, rv.text
    logged = rv.json()["logged"]
    assert logged["action_suggested"] == "escalate" and logged["band"] == "high"
    assert (tmp_path / "hitl_review_log.csv").exists()

    missing = client.post("/v1/churn/reviews", json={"user_id": "never_scored", "reviewer": "cs", "action_taken": "monitor"})
    assert missing.status_code == 404


def test_streamlit_presets_reproduce_each_scenario(manifest):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(config.PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception
    options = at.sidebar.selectbox(key="use_case").options
    for p in manifest["personas"]:
        title = p["title"] if p["title"] in options else options[0]  # Santosh is the default preset
        at.sidebar.selectbox(key="use_case").set_value(title).run()
        assert not at.exception, [e.value for e in at.exception]
        text = " ".join(m.value for m in at.markdown)
        assert f"**Risk band:** `{p['expected']['band']}`" in text, p["id"]
        assert f"**HITL action:** `{p['expected']['hitl_action']}`" in text, p["id"]
