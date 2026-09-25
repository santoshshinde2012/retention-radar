"""The committed bundle must load however the service is launched.

`python -m ...` puts the repo root on sys.path, but the `streamlit` / `uvicorn`
console scripts and Streamlit Community Cloud do not. A pickle that references
`src.retention_radar...` only loads in the first case, so these tests run in a
subprocess from outside the repo with only `src/` importable.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from retention_radar import config

ROOT = config.PROJECT_ROOT


def _run(code: str, tmp_path) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "RETENTION_RADAR_ARTIFACT_DIR")}
    env.update(PYTHONPATH=str(ROOT / "src"), CHURN_DATA_SOURCE="synthetic")
    return subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=300
    )


def test_committed_pickles_do_not_reference_src_package():
    for name in ("churn_xgb.joblib", "calibrator.joblib"):
        assert b"src.retention_radar" not in (config.SEED_MODELS_DIR / name).read_bytes(), name


def test_bundle_scores_santosh_outside_repo_root(tmp_path):
    code = f"""
import json, sys
assert {str(ROOT)!r} not in sys.path and '' not in sys.path[1:]
from retention_radar.serving.packet import build_decision_packet
p = build_decision_packet(json.load(open({str(ROOT / 'data/raw/santosh_shinde.json')!r})))
print(round(p['scoring']['churn_probability_calibrated'], 6), p['hitl']['action'])
"""
    res = _run(code, tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.split()[-2:] == ["0.016461", "monitor"]


def test_streamlit_app_boots_like_streamlit_cloud(tmp_path):
    pytest.importorskip("streamlit.testing.v1")
    code = f"""
import sys
sys.path = [p for p in sys.path if p not in ('', {str(ROOT)!r})]
from streamlit.testing.v1 import AppTest
at = AppTest.from_file({str(ROOT / 'app/streamlit_app.py')!r}, default_timeout=120).run()
assert not at.exception, [e.value for e in at.exception]
assert not at.error, [e.value for e in at.error]
print('OK', any('Risk band:** `low`' in m.value for m in at.markdown))
"""
    res = _run(code, tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip().endswith("OK True")


def test_legacy_src_pickle_still_loads(tmp_path):
    """Old bundles pickled as src.retention_radar.* load via the alias shim."""
    code = """
import joblib, sys, types
import retention_radar.training.calibrate as cal
legacy = types.ModuleType("src.retention_radar.training.calibrate")
legacy.ProbabilityCalibrator = type("ProbabilityCalibrator", (cal.ProbabilityCalibrator,), {"__module__": legacy.__name__})
c = cal.ProbabilityCalibrator("isotonic").fit([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9])
c.__class__ = legacy.ProbabilityCalibrator
for n in ("src", "src.retention_radar", "src.retention_radar.training"):
    sys.modules[n] = types.ModuleType(n)
sys.modules[legacy.__name__] = legacy
joblib.dump(c, "legacy.joblib")
for n in list(sys.modules):
    if n == "src" or n.startswith("src."):
        del sys.modules[n]
from pathlib import Path
loaded = cal.load_calibrator(Path("legacy.joblib"))
print(float(loaded.transform([0.9])[0]))
"""
    res = _run(code, tmp_path)
    assert res.returncode == 0, res.stderr[-2000:]
    assert float(res.stdout.split()[-1]) == pytest.approx(1.0)
