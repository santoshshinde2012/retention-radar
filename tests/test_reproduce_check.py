"""check_reproduction: exact diff of a retrain vs the committed bundle (latency ignored)."""

from __future__ import annotations

import json

from retention_radar.evaluation.reproduce import compare_metrics, compare_packets, main


def test_compare_metrics_ignores_latency_and_flags_drift():
    ref = {"logreg_test": {"roc_auc": 0.8719}, "best_f1_threshold": 0.34,
           "latency": {"latency_ms_p50": 2.0}, "feature_names": ["a", "b"]}
    same = json.loads(json.dumps(ref))
    same["latency"]["latency_ms_p50"] = 9.9
    assert compare_metrics(ref, same) == []

    drift = json.loads(json.dumps(ref))
    drift["best_f1_threshold"] = 0.42
    drift["feature_names"] = ["a"]
    keys = [k for k, _, _ in compare_metrics(ref, drift)]
    assert keys == ["best_f1_threshold", "feature_names"]


def test_compare_packets_checks_scores_band_and_action():
    ref = {"scoring": {"churn_probability_raw": 0.0434, "churn_probability_calibrated": 0.0165,
                       "risk_band": "low", "best_f1_threshold": 0.34}, "hitl": {"action": "monitor"}}
    assert compare_packets(ref, json.loads(json.dumps(ref))) == []
    moved = json.loads(json.dumps(ref))
    moved["scoring"]["churn_probability_raw"] = 0.0515
    moved["hitl"]["action"] = "nurture / check-in"
    assert [k for k, _, _ in compare_packets(ref, moved)] == [
        "scoring.churn_probability_raw", "hitl.action",
    ]


def test_cli_exit_codes(tmp_path):
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"x": 1.0}))
    art = tmp_path / "art"
    art.mkdir()
    assert main(["--artifact-dir", str(art), "--reference", str(ref)]) == 2
    (art / "metrics.json").write_text(json.dumps({"x": 1.0}))
    assert main(["--artifact-dir", str(art), "--reference", str(ref),
                 "--reference-packet", str(tmp_path / "none.json")]) == 0
    (art / "metrics.json").write_text(json.dumps({"x": 2.0}))
    assert main(["--artifact-dir", str(art), "--reference", str(ref),
                 "--reference-packet", str(tmp_path / "none.json")]) == 1
