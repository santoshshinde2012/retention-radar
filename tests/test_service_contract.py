"""Regression tests for the serving contract.

One contract for every surface: packet, CLI, CSV batch, API score/batch/actions,
action log and outcomes. Invalid input is never scored or queued.
"""

from __future__ import annotations

import csv
import json
import threading

import pandas as pd
import pytest

from retention_radar import config
from retention_radar.data.use_cases import USE_CASE_DIR, load_manifest
from retention_radar.serving.action_log import (
    ACTION_LOG_COLUMNS,
    append_action_row,
    row_from_score_record,
    rows_from_decisions,
)
from retention_radar.serving.action_log import main as action_main
from retention_radar.serving.batch_score import score_csv, split_valid_payloads
from retention_radar.serving.outcomes import join_outcomes
from retention_radar.serving.packet import build_decision_packet, normalize_record, validate_payload
from retention_radar.serving.policy import HOLD_ACTION

MANIFEST = USE_CASE_DIR / "personas.json"
pytestmark = pytest.mark.skipif(
    not MANIFEST.exists() or config.MODEL_PATH.parent != config.SEED_MODELS_DIR,
    reason="needs data/use_cases/ and the committed seed-42 bundle",
)


@pytest.fixture(scope="module")
def santosh():
    m = load_manifest(MANIFEST)
    return next(p for p in m["personas"] if p["id"] == "santosh_capped_pro")["record"]


BAD_VALUES = [
    ("suggestion_accept_rate_28d", float("nan"), "finite"),
    ("suggestion_accept_rate_28d", float("inf"), "finite"),
    ("active_days_28d", "nan", "finite"),
    ("agent_requests_28d", 10**400, "numeric"),
    ("limit_hits_14d", True, "boolean"),
    ("active_days_28d", [3], "numeric"),
    ("overage_toggled_off", 0.5, "0 or 1"),
]


@pytest.mark.parametrize("field,value,needle", BAD_VALUES)
def test_non_finite_bool_overflow_and_non_binary_are_held_everywhere(santosh, field, value, needle):
    rec = {**santosh, field: value}
    v = validate_payload(normalize_record(rec)[0])
    assert not v["ok"] and any(needle in e for e in v["errors"]), v
    packet = build_decision_packet(rec)
    assert packet["scoring"] is None and packet["decision"]["action"] == HOLD_ACTION
    valid, rejects = split_valid_payloads([rec, santosh | {"user_id": "ok_user"}])
    assert [r["user_id"] for r in valid] == ["ok_user"] and len(rejects) == 1


def test_one_contract_for_identities_enums_and_extras(santosh):
    rec = {**santosh, "plan_tier": " Pro ", "user_id": f" {santosh['user_id']} ", "churned": 1, "crm_id": 7}
    payload, notes = normalize_record(rec)
    assert payload["plan_tier"] == "pro" and payload["user_id"] == santosh["user_id"]
    assert set(payload) == set(config.INFERENCE_REQUIRED_KEYS)
    assert notes and "churned" in notes[0]
    packet = build_decision_packet(rec)  # extras ignored, not a hold
    assert packet["scoring"] is not None and "churned" not in packet["payload"]
    assert any("ignored fields" in w for w in packet["validation"]["warnings"])
    valid, rejects = split_valid_payloads([rec, {**santosh, "user_id": {"a": 1}}, None, "x"])
    assert len(valid) == 1 and [r["row_number"] for r in rejects] == [2, 3, 4]


def test_duplicate_user_ids_are_rejected_and_block_action_attachment(santosh):
    other = {**santosh, "limit_hits_14d": 0, "cheap_model_share_28d": 0.1}
    valid, rejects = split_valid_payloads([santosh, other])
    assert len(valid) == 1 and "duplicate user_id" in rejects[0]["errors"]
    queue = [{"user_id": "u1", "p_cal": 0.3, "band": "high", "action": "limit_reset"},
             {"user_id": "u1", "p_cal": 0.02, "band": "low", "action": "no_action"}]
    rows, problems = rows_from_decisions(
        queue, [{"user_id": "u1", "executed_by": "lifecycle-tool", "action_taken": "limit_reset"}]
    )
    assert rows == [] and "more than once" in problems[0]


def test_held_or_scoreless_records_cannot_be_logged(santosh):
    held = build_decision_packet({**santosh, "suggestion_accept_rate_28d": None})
    for rec in (held, {"user_id": "u", "action": "limit_reset"}, {"user_id": "u", "p_cal": 0.4, "band": ""}):
        with pytest.raises(ValueError, match="no score"):
            row_from_score_record(rec, executed_by="lifecycle-tool", action_taken="limit_reset")


def test_bulk_import_is_all_or_nothing_idempotent_and_keeps_queue_holdout(tmp_path):
    queue = tmp_path / "queue.csv"
    pd.DataFrame([
        {"rank": 1, "user_id": "u1", "p_raw": 0.8, "p_cal": 0.3, "band": "high", "action": "holdout",
         "holdout": True, "would_have_sent": "limit_reset"},
        {"rank": 2, "user_id": "u2", "p_raw": 0.1, "p_cal": 0.02, "band": "low", "action": "no_action",
         "holdout": False, "would_have_sent": ""},
    ]).to_csv(queue, index=False)
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"user_id": "u1", "executed_by": "none", "action_taken": "holdout"},
                  {"user_id": "ghost", "executed_by": "x", "action_taken": "limit_reset"}]).to_csv(bad, index=False)
    log = tmp_path / "log.csv"
    with pytest.raises(SystemExit):
        action_main(["--from-scores", str(queue), "--decisions", str(bad), "--log", str(log)])
    assert not log.exists()  # nothing half-imported

    good = tmp_path / "good.csv"
    # A send export cannot flip holdout: the queue's value wins.
    pd.DataFrame([{"user_id": "u1", "executed_by": "none", "action_taken": "holdout", "notes": "",
                   "holdout": "false"}]).to_csv(good, index=False)
    args = ["--from-scores", str(queue), "--decisions", str(good), "--log", str(log), "--timestamp", "2026-09-28T09:00:00Z"]
    action_main(args)
    action_main(args)  # rerun → no duplicate
    with open(log, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["timestamp"] == "2026-09-28T09:00:00Z"
    assert rows[0]["holdout"] == "true" and rows[0]["would_have_sent"] == "limit_reset"


def test_log_appends_are_safe_under_concurrency(tmp_path):
    log = tmp_path / "log.csv"
    row = {c: "x" for c in ACTION_LOG_COLUMNS}

    def worker(i):
        append_action_row(log, {**row, "user_id": f"u{i}"})

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
    log = pd.DataFrame([{c: "" for c in ACTION_LOG_COLUMNS} | {"user_id": u, "p_cal": "0.3", "band": "high",
                        "action_suggested": "limit_reset", "timestamp": ts}
                        for u, ts in [("a", "2026-09-28T09:00:00Z"), ("b", "2026-09-28"), ("c", "not a date")]])
    labels = pd.DataFrame({"user_id": ["a", "b", "c", "a"], "churned": [1, 1, 1, 0],
                           "observed_at": ["2026-10-05", "2026-09-01T00:00:00Z", "2026-10-05", ""]})
    j = join_outcomes(log, labels).set_index("user_id")
    assert j.loc["a", "churned"] == 1  # dated label beats the undated one; after the action
    assert pd.isna(j.loc["b", "churned"])  # observed before the action → not counted
    assert pd.isna(j.loc["c", "churned"])  # unparseable action time → not counted


def test_empty_csv_fails_cleanly_and_clears_stale_queue(tmp_path):
    out = tmp_path / "q.csv"
    out.write_text("stale")
    (tmp_path / "q_rejected.csv").write_text("stale")
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        score_csv(empty, out_csv=out)
    assert not out.exists() and not (tmp_path / "q_rejected.csv").exists()


def test_packet_dir_holds_unreadable_files_and_default_out_follows_user(tmp_path, santosh, monkeypatch):
    from retention_radar.serving.packet import main as packet_main

    d = tmp_path / "records"
    d.mkdir()
    (d / "a_good.json").write_text(json.dumps(santosh))
    (d / "b_broken.json").write_text("{not json")
    (d / "c_list.json").write_text("[1, 2]")
    packet_main(["--dir", str(d), "--out", str(tmp_path / "p.jsonl")])
    lines = [json.loads(x) for x in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert [p["decision"]["action"] for p in lines] == ["limit_reset", HOLD_ACTION, HOLD_ACTION]

    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    packet_main(["--json", str(d / "a_good.json")])
    assert (tmp_path / f"{santosh['user_id']}_decision_packet.json").exists()
    assert not (tmp_path / "santosh_decision_packet.json").exists()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from retention_radar.serving import api as api_mod

    monkeypatch.setenv("RETENTION_RADAR_LOG_DIR", str(tmp_path / "logs"))
    api_mod._load_serve_bundle.cache_clear()
    return TestClient(api_mod.app)


def test_api_every_invalid_record_gets_the_hold_envelope(client, santosh):
    m = load_manifest(MANIFEST)
    bodies = [c["record"] for c in m["invalid_records"]] + [
        {**santosh, "limit_hits_14d": True},
        {k: v for k, v in santosh.items() if k != "user_id"},
        {**santosh, "user_name": ""},
    ]
    for body in bodies:
        r = client.post("/v1/churn/score?log=false", json=body)
        assert r.status_code == 422, r.text
        assert r.json()["detail"]["decision"]["action"] == HOLD_ACTION
    for raw in ('{"limit_hits_14d": NaN}', "[1, 2]", "not json"):
        r = client.post("/v1/churn/score", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422 and r.json()["detail"]["decision"]["action"] == HOLD_ACTION, raw


def test_api_same_contract_as_batch(client, santosh):
    rec = {**santosh, "plan_tier": " PRO ", "extra": 1}
    one = client.post("/v1/churn/score?log=false", json=rec)
    body = json.dumps({"records": [rec, None, 5, {**santosh, "failed_requests_rate": float("inf")}]})
    many = client.post("/v1/churn/batch?log=false", content=body, headers={"content-type": "application/json"})
    assert one.status_code == 200 and many.status_code == 200, (one.text, many.text)
    assert many.json()["queue"][0]["p_cal"] == one.json()["p_cal"]
    assert many.json()["queue"][0]["action"] == one.json()["action"] == "limit_reset"
    assert [r["row_number"] for r in many.json()["rejected"]] == [2, 3, 4]


def test_api_actions_strip_whitespace_survive_bad_log_lines_and_use_log_dir(client, santosh, tmp_path):
    assert client.post("/v1/churn/score", json=santosh).status_code == 200
    log_dir = tmp_path / "logs" / "prediction_log"
    with open(next(log_dir.glob("scores_*.jsonl")), "a", encoding="utf-8") as f:
        f.write("\n{truncated\n")
    r = client.post("/v1/churn/actions", json={"user_id": f" {santosh['user_id']} ", "executed_by": " lifecycle-tool ",
                                               "action_taken": "limit_reset"})
    assert r.status_code == 200, r.text
    assert r.json()["logged"]["executed_by"] == "lifecycle-tool"
    assert (tmp_path / "logs" / "action_log.csv").exists()
    blank = client.post("/v1/churn/actions", json={"user_id": santosh["user_id"], "executed_by": "   ",
                                                   "action_taken": "limit_reset"})
    assert blank.status_code == 422


def test_streamlit_what_if_sliders_stay_inside_the_contract():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(config.PROJECT_ROOT / "app" / "streamlit_app.py"), default_timeout=120).run()
    # Santosh is the default preset. Push sliders to their extremes: the record must stay valid
    # (active days are clamped, engagement_trend is derived), so every tab still scores.
    at.slider(key="hero_santosh:active_days_28d").set_value(0).run()
    at.slider(key="hero_santosh:limit_hits_14d").set_value(20).run()
    at.slider(key="hero_santosh:allowance_used_pct").set_value(3.0).run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "**Risk band:**" in text and HOLD_ACTION not in text


def test_cli_wrappers_propagate_exit_codes(tmp_path, santosh):
    """`python -m retention_radar.cli.*` must return the module's exit code (CI relies on it)."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONPATH": str(config.PROJECT_ROOT / "src"), "CHURN_DATA_SOURCE": "synthetic"}
    base = {k: santosh[k] for k in config.INFERENCE_REQUIRED_KEYS}
    rows = [{**base, "user_id": f"u{i}"} for i in range(30)]
    drifted = pd.DataFrame(rows).assign(limit_hits_14d=40, active_days_28d=0, active_days_7d=0)
    drifted.to_csv(tmp_path / "drifted.csv", index=False)
    pd.DataFrame(rows + [{**base, "user_id": "bad", "failed_requests_rate": 9}]).to_csv(tmp_path / "batch.csv", index=False)

    def run(*args):
        return subprocess.run([sys.executable, "-m", *args], cwd=tmp_path, env=env, capture_output=True, text=True).returncode

    assert run("retention_radar.cli.drift_check", "--csv", str(tmp_path / "drifted.csv"), "--strict",
               "--out", str(tmp_path / "d.json")) == 1
    assert run("retention_radar.cli.batch_score", "--csv", str(tmp_path / "batch.csv"), "--out",
               str(tmp_path / "q.csv"), "--strict") == 1
    assert run("retention_radar.cli.check_reproduction", "--artifact-dir", str(tmp_path / "none")) == 2
