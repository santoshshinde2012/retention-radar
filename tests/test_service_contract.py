"""Regression tests for the serving contract (adversarial review, 2026-09-25).

One contract for every surface: packet, CLI, CSV batch, API score/batch/reviews,
HITL log and outcomes. Invalid input is never scored or queued.
"""

from __future__ import annotations

import csv
import json
import threading

import pandas as pd
import pytest

from retention_radar import config
from retention_radar.data.use_cases import USE_CASE_DIR, load_manifest
from retention_radar.serving.batch_score import score_csv, split_valid_payloads
from retention_radar.serving.hitl_log import (
    HITL_LOG_COLUMNS,
    append_hitl_row,
    row_from_score_record,
    rows_from_decisions,
)
from retention_radar.serving.hitl_log import main as hitl_main
from retention_radar.serving.outcomes import join_outcomes
from retention_radar.serving.packet import build_decision_packet, normalize_record, validate_payload
from retention_radar.serving.policy import HOLD_ACTION

MANIFEST = USE_CASE_DIR / "personas.json"
pytestmark = pytest.mark.skipif(
    not MANIFEST.exists() or config.MODEL_PATH.parent != config.SEED_MODELS_DIR,
    reason="needs data/use_cases/ and the committed seed-42 bundle",
)


@pytest.fixture(scope="module")
def gone_dark():
    m = load_manifest(MANIFEST)
    return next(p for p in m["personas"] if p["id"] == "gone_dark")["record"]


BAD_VALUES = [
    ("nps_score", float("nan"), "finite"),
    ("nps_score", float("inf"), "finite"),
    ("sessions_last_30d", "nan", "finite"),
    ("sessions_last_30d", 10**400, "numeric"),
    ("payment_failures_last_90d", True, "boolean"),
    ("sessions_last_30d", [3], "numeric"),
]


@pytest.mark.parametrize("field,value,needle", BAD_VALUES)
def test_non_finite_bool_and_overflow_are_held_everywhere(gone_dark, field, value, needle):
    rec = {**gone_dark, field: value}
    v = validate_payload(normalize_record(rec)[0])
    assert not v["ok"] and any(needle in e for e in v["errors"]), v
    packet = build_decision_packet(rec)
    assert packet["scoring"] is None and packet["hitl"]["action"] == HOLD_ACTION
    valid, rejects = split_valid_payloads([rec, gone_dark | {"user_id": "ok_user"}])
    assert [r["user_id"] for r in valid] == ["ok_user"] and len(rejects) == 1


def test_one_contract_for_identities_enums_and_extras(gone_dark):
    rec = {**gone_dark, "plan_tier": " Free ", "user_id": f" {gone_dark['user_id']} ", "churned": 1, "crm_id": 7}
    payload, notes = normalize_record(rec)
    assert payload["plan_tier"] == "free" and payload["user_id"] == gone_dark["user_id"]
    assert set(payload) == set(config.INFERENCE_REQUIRED_KEYS)
    assert notes and "churned" in notes[0]
    packet = build_decision_packet(rec)  # extras ignored, not a hold
    assert packet["scoring"] is not None and "churned" not in packet["payload"]
    assert any("ignored fields" in w for w in packet["validation"]["warnings"])
    valid, rejects = split_valid_payloads([rec, {**gone_dark, "user_id": {"a": 1}}, None, "x"])
    assert len(valid) == 1 and [r["row_number"] for r in rejects] == [2, 3, 4]


def test_duplicate_user_ids_are_rejected_and_block_review_attachment(gone_dark):
    low = {**gone_dark, "last_active_days_ago": 0, "sessions_last_30d": 40, "sessions_last_7d": 10}
    valid, rejects = split_valid_payloads([gone_dark, low])
    assert len(valid) == 1 and "duplicate user_id" in rejects[0]["errors"]
    queue = [{"user_id": "u1", "p_cal": 0.9, "band": "high", "hitl_action": "escalate"},
             {"user_id": "u1", "p_cal": 0.1, "band": "low", "hitl_action": "monitor"}]
    rows, problems = rows_from_decisions(queue, [{"user_id": "u1", "reviewer": "r", "action_taken": "escalate"}])
    assert rows == [] and "more than once" in problems[0]


def test_held_or_scoreless_records_cannot_be_reviewed(gone_dark):
    held = build_decision_packet({**gone_dark, "nps_score": None})
    for rec in (held, {"user_id": "u", "hitl_action": "escalate"}, {"user_id": "u", "p_cal": 0.4, "band": ""}):
        with pytest.raises(ValueError, match="no score"):
            row_from_score_record(rec, reviewer="r", action_taken="monitor")


def test_bulk_import_is_all_or_nothing_idempotent_and_timestamped(tmp_path):
    queue = tmp_path / "queue.csv"
    pd.DataFrame([{"rank": 1, "user_id": "u1", "p_raw": 0.9, "p_cal": 0.9, "band": "high", "hitl_action": "escalate"},
                  {"rank": 2, "user_id": "u2", "p_raw": 0.1, "p_cal": 0.1, "band": "low", "hitl_action": "monitor"}]).to_csv(queue, index=False)
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"user_id": "u1", "reviewer": "r", "action_taken": "escalate"},
                  {"user_id": "ghost", "reviewer": "r", "action_taken": "monitor"}]).to_csv(bad, index=False)
    log = tmp_path / "log.csv"
    with pytest.raises(SystemExit):
        hitl_main(["--from-scores", str(queue), "--decisions", str(bad), "--log", str(log)])
    assert not log.exists()  # nothing half-imported

    good = tmp_path / "good.csv"
    pd.DataFrame([{"user_id": "u1", "reviewer": "r", "action_taken": "escalate", "notes": ""}]).to_csv(good, index=False)
    args = ["--from-scores", str(queue), "--decisions", str(good), "--log", str(log), "--timestamp", "2026-09-29T15:00:00Z"]
    hitl_main(args)
    hitl_main(args)  # rerun → no duplicate
    with open(log, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["timestamp"] == "2026-09-29T15:00:00Z"


def test_log_appends_are_safe_under_concurrency(tmp_path):
    log = tmp_path / "log.csv"
    row = {c: "x" for c in HITL_LOG_COLUMNS}

    def worker(i):
        append_hitl_row(log, {**row, "user_id": f"u{i}"})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(log, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 40 and {r["user_id"] for r in rows} == {f"u{i}" for i in range(40)}
    assert log.read_text().count("user_id,p_cal") == 1  # header once


def test_outcomes_leak_guard_handles_mixed_and_bad_dates():
    log = pd.DataFrame([{c: "" for c in HITL_LOG_COLUMNS} | {"user_id": u, "p_cal": "0.5", "band": "high",
                        "action_suggested": "escalate", "timestamp": ts}
                        for u, ts in [("a", "2026-09-29T15:00:00Z"), ("b", "2026-09-29"), ("c", "not a date")]])
    labels = pd.DataFrame({"user_id": ["a", "b", "c", "a"], "churned": [1, 1, 1, 0],
                           "observed_at": ["2026-10-28", "2026-09-01T00:00:00Z", "2026-10-28", ""]})
    j = join_outcomes(log, labels).set_index("user_id")
    assert j.loc["a", "churned"] == 1  # dated label beats the undated one; after review
    assert pd.isna(j.loc["b", "churned"])  # observed before review → not counted
    assert pd.isna(j.loc["c", "churned"])  # unparseable review time → not counted


def test_empty_csv_fails_cleanly_and_clears_stale_queue(tmp_path):
    out = tmp_path / "q.csv"
    out.write_text("stale")
    (tmp_path / "q_rejected.csv").write_text("stale")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        score_csv(empty, out_csv=out)
    assert not out.exists() and not (tmp_path / "q_rejected.csv").exists()


def test_packet_dir_holds_unreadable_files_and_default_out_follows_user(tmp_path, gone_dark, monkeypatch):
    from retention_radar.serving.packet import main as packet_main

    d = tmp_path / "records"
    d.mkdir()
    (d / "a_good.json").write_text(json.dumps(gone_dark))
    (d / "b_broken.json").write_text("{not json")
    (d / "c_list.json").write_text("[1, 2]")
    packet_main(["--dir", str(d), "--out", str(tmp_path / "p.jsonl")])
    lines = [json.loads(x) for x in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert [p["hitl"]["action"] for p in lines] == ["escalate", HOLD_ACTION, HOLD_ACTION]

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    packet_main(["--json", str(d / "a_good.json")])
    assert (tmp_path / f"{gone_dark['user_id']}_decision_packet.json").exists()
    assert not (tmp_path / "santosh_decision_packet.json").exists()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retention_radar.serving import api as api_mod

    monkeypatch.setenv("RETENTION_RADAR_LOG_DIR", str(tmp_path / "logs"))
    api_mod._load_serve_bundle.cache_clear()
    return TestClient(api_mod.app)


def test_api_every_invalid_record_gets_the_hold_envelope(client, gone_dark):
    m = load_manifest(MANIFEST)
    bodies = [c["record"] for c in m["invalid_records"]] + [
        {**gone_dark, "payment_failures_last_90d": True},
        {k: v for k, v in gone_dark.items() if k != "user_id"},
        {**gone_dark, "user_name": ""},
    ]
    for body in bodies:
        r = client.post("/v1/churn/score?log=false", json=body)
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["hitl"]["action"] == HOLD_ACTION
    for raw in ('{"nps_score": NaN}', "[1, 2]", "not json"):
        r = client.post("/v1/churn/score", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422 and r.json()["detail"]["hitl"]["action"] == HOLD_ACTION, raw


def test_api_same_contract_as_batch(client, gone_dark):
    rec = {**gone_dark, "plan_tier": " FREE ", "extra": 1}
    one = client.post("/v1/churn/score?log=false", json=rec)
    body = json.dumps({"records": [rec, None, 5, {**gone_dark, "nps_score": float("inf")}]})  # → Infinity
    many = client.post("/v1/churn/batch?log=false", content=body, headers={"content-type": "application/json"})
    assert one.status_code == 200 and many.status_code == 200, (one.text, many.text)
    assert many.json()["queue"][0]["p_cal"] == one.json()["p_cal"]
    assert [r["row_number"] for r in many.json()["rejected"]] == [2, 3, 4]


def test_api_reviews_strip_whitespace_survive_bad_log_lines_and_use_log_dir(client, gone_dark, tmp_path):
    assert client.post("/v1/churn/score", json=gone_dark).status_code == 200
    log_dir = tmp_path / "logs" / "prediction_log"
    with open(next(log_dir.glob("scores_*.jsonl")), "a", encoding="utf-8") as f:
        f.write("\n{truncated\n")
    r = client.post("/v1/churn/reviews", json={"user_id": f" {gone_dark['user_id']} ", "reviewer": " cs ", "action_taken": "escalate"})
    assert r.status_code == 200, r.text
    assert r.json()["logged"]["reviewer"] == "cs"
    assert (tmp_path / "logs" / "hitl_review_log.csv").exists()
    blank = client.post("/v1/churn/reviews", json={"user_id": gone_dark["user_id"], "reviewer": "   ", "action_taken": "escalate"})
    assert blank.status_code == 422


def test_streamlit_invalid_what_if_is_held_on_every_tab():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(config.PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    # Auto engagement_trend is on for Santosh; 30d sessions → 0 makes the trend 9.0 (> 5): invalid.
    at.slider(key="santosh_default:sessions_last_30d").set_value(0).run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "Churn probability" not in text
    assert f"**HITL action:** `{HOLD_ACTION}`" in text
    assert any("engagement_trend" in e.value for e in at.markdown) or any("engagement_trend" in e.value for e in at.error)
