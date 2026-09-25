"""Import smoke + HITL policy unit tests (no model fit)."""

from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyClassifier

from retention_radar.features.transform import DefaultFeatureTransformer
from retention_radar.protocols import (
    Calibrator,
    DecisionPolicy,
    FeatureTransformer,
    ProbabilisticClassifier,
)
from retention_radar.serving.policy import HitlDecisionPolicy, hitl_action, risk_band
from retention_radar.serving.scoring import CalibratedScorer
from retention_radar.training.calibrate import ProbabilityCalibrator


def test_compat_entrypoints_importable():
    import retention_radar.cli.batch_score  # noqa: F401
    import retention_radar.cli.benchmark  # noqa: F401
    import retention_radar.cli.docs_gen  # noqa: F401
    import retention_radar.cli.drift_check  # noqa: F401
    import retention_radar.cli.evaluate  # noqa: F401
    import retention_radar.cli.explain  # noqa: F401
    import retention_radar.cli.generate_data  # noqa: F401
    import retention_radar.cli.hitl_log  # noqa: F401
    import retention_radar.cli.hitl_outcomes  # noqa: F401
    import retention_radar.cli.infer  # noqa: F401
    import retention_radar.cli.ingest  # noqa: F401
    import retention_radar.cli.single_record  # noqa: F401
    import retention_radar.cli.slice_metrics  # noqa: F401
    import retention_radar.cli.train  # noqa: F401
    from retention_radar import config as rr_config
    from retention_radar.evaluation.slices import main as slice_main
    from retention_radar.serving.explain import main as explain_main

    assert rr_config.PROJECT_ROOT.exists()
    assert callable(slice_main) and callable(explain_main)


def test_protocols_satisfied_by_defaults():
    assert isinstance(DefaultFeatureTransformer(), FeatureTransformer)
    assert isinstance(HitlDecisionPolicy(), DecisionPolicy)
    dummy = DummyClassifier(strategy="prior")
    dummy.fit([[0.0], [1.0]], [0, 1])
    assert isinstance(dummy, ProbabilisticClassifier)
    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit([0, 1], [0.1, 0.9])
    assert isinstance(cal, Calibrator)
    scorer = CalibratedScorer(dummy, cal)
    raw, calibrated, display = scorer.score([[0.0]])
    assert raw.shape[0] == 1
    assert calibrated is not None
    assert display.shape[0] == 1


def test_hitl_policy_never_auto_acts():
    policy = HitlDecisionPolicy()
    cases = [
        (0.01, 0.34, "low"),
        (0.20, 0.34, "medium"),
        (0.50, 0.34, "medium"),
        (0.80, 0.34, "high"),
    ]
    for prob, thr, band in cases:
        out = policy.decide(prob, thr, band)
        assert out["auto_action"] == "none"
        assert out["hitl_required"] is True
        assert hitl_action(prob, thr, band)["action"] == out["action"]


def test_risk_band_cutoffs():
    assert risk_band(0.10) == "low"
    assert risk_band(0.45) == "medium"
    assert risk_band(0.90) == "high"


def test_streamlit_scores_active_santosh():
    text = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text(
        encoding="utf-8"
    )
    assert "resolve_santosh_json" in text
    assert "SANTOSH_JSON" not in text
