"""Headless Streamlit smoke: the HITL app boots on the committed bundle and never auto-acts."""

from __future__ import annotations

import pytest

from retention_radar import config

APP = config.PROJECT_ROOT / "app" / "streamlit_app.py"


@pytest.fixture()
def committed_bundle():
    if not (config.SEED_MODELS_DIR / "churn_xgb.joblib").exists():
        pytest.skip("committed seed-42 bundle missing")
    if config.MODEL_PATH.parent != config.SEED_MODELS_DIR:
        pytest.skip("RETENTION_RADAR_ARTIFACT_DIR is set; app would load another bundle")


def test_streamlit_app_scores_santosh_without_auto_action(committed_bundle):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=120).run()

    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert [t.label for t in at.tabs] == [
        "Predict", "Explain", "Decision", "Methodology", "Benchmarks",
    ]
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "Auto action: `none`" in text
    assert "**Risk band:** `low`" in text  # Santosh preset (seed 42) → low / monitor
