"""Import smoke + decision policy unit tests (no model fit)."""

from __future__ import annotations

from pathlib import Path

from sklearn.dummy import DummyClassifier

from retention_radar import config
from retention_radar.features.transform import DefaultFeatureTransformer
from retention_radar.protocols import (
    Calibrator,
    DecisionPolicy,
    FeatureTransformer,
    ProbabilisticClassifier,
)
from retention_radar.serving.policy import (
    RenewalDecisionPolicy,
    decide,
    expected_value,
    in_holdout,
    risk_band,
)
from retention_radar.serving.scoring import CalibratedScorer
from retention_radar.training.calibrate import ProbabilityCalibrator


def test_compat_entrypoints_importable():
    import retention_radar.cli.action_log  # noqa: F401
    import retention_radar.cli.batch_score  # noqa: F401
    import retention_radar.cli.benchmark  # noqa: F401
    import retention_radar.cli.check_reproduction  # noqa: F401
    import retention_radar.cli.docs_gen  # noqa: F401
    import retention_radar.cli.drift_check  # noqa: F401
    import retention_radar.cli.evaluate  # noqa: F401
    import retention_radar.cli.explain  # noqa: F401
    import retention_radar.cli.generate_data  # noqa: F401
    import retention_radar.cli.infer  # noqa: F401
    import retention_radar.cli.ingest  # noqa: F401
    import retention_radar.cli.outcomes  # noqa: F401
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
    assert isinstance(RenewalDecisionPolicy(), DecisionPolicy)
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


def test_policy_never_auto_acts_and_only_ultra_gets_a_person():
    policy = RenewalDecisionPolicy()
    heavy = {"user_id": "u", "plan_tier": "pro", "limit_hits_14d": 4, "weekend_usage_ratio": 0.1, "active_days_7d": 3}
    for prob in (0.01, 0.10, 0.20, 0.45):
        for tier in config.PLAN_TIER_ORDER:
            rec = {**heavy, "plan_tier": tier, "user_id": f"u-{tier}-{prob}"}
            out = policy.decide(prob, 0.14, risk_band(prob), rec)
            assert out["auto_action"] == "none"
            assert out["hitl_required"] == (out["action"] == "personal_email")
            if out["action"] == "personal_email":
                assert tier == "ultra"
            assert decide(prob, 0.14, risk_band(prob), rec)["action"] == out["action"]


def test_below_tau_is_no_action_even_with_cap_hits():
    out = decide(0.10, 0.14, "medium", {"user_id": "x", "plan_tier": "pro", "limit_hits_14d": 9})
    assert out["action"] == "no_action" and out["candidates"] == []


def test_expected_value_counts_discounts_paid_to_sure_things():
    rec = {"plan_tier": "pro"}
    ev_low = expected_value("cancel_flow_discount", 0.05, rec)
    ev_high = expected_value("cancel_flow_discount", 0.40, rec)
    assert ev_low < ev_high
    price = config.PLAN_PRICE_USD["pro"]
    pb = config.PLAYBOOKS["cancel_flow_discount"]
    saved = 0.40 * pb["effect"]
    discount = pb["discount_usd_pct_of_price"] * price
    want = saved * price * config.MONTHS_RETAINED_AFTER_SAVE - discount * (saved + 0.60 * pb["sure_thing_accept"])
    assert ev_high == __import__("pytest").approx(want)


def test_holdout_is_deterministic_and_near_its_share():
    ids = [f"sub_{i:05d}" for i in range(20000)]
    share = sum(in_holdout(u) for u in ids) / len(ids)
    assert abs(share - config.HOLDOUT_PCT / 100) < 0.01
    assert [in_holdout(u) for u in ids[:50]] == [in_holdout(u) for u in ids[:50]]
    assert in_holdout(None) is False


def test_risk_band_cutoffs():
    low, high = config.RISK_BAND_EDGES
    assert risk_band(low - 1e-6) == "low"
    assert risk_band(low) == "medium"
    assert risk_band(high) == "high"


def test_streamlit_uses_active_hero_resolver():
    text = (Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    assert "resolve_hero_json" in text
    assert "santosh" not in text.lower()


def test_risk_band_rejects_non_finite():
    import math

    import pytest

    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            risk_band(bad)


def test_holdout_sizes_account_for_the_unequal_split():
    from retention_radar.evaluation.analysis import holdout_sizes

    equal = holdout_sizes(0.25, 0.25, 50)
    nine_to_one = holdout_sizes(0.25, 0.25, 10)
    assert equal["n_holdout"] == equal["n_treated"]
    # A 9:1 split needs a smaller holdout than an equal split, but more subscribers overall.
    assert nine_to_one["n_holdout"] < equal["n_holdout"]
    assert nine_to_one["n_eligible"] > 2 * equal["n_holdout"]
    assert nine_to_one["n_treated"] == 9 * nine_to_one["n_holdout"]
