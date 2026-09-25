"""Lakehouse / alternate artifact dir must not touch committed seed-42 models/."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from retention_radar import config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def seed_fingerprints():
    """Snapshot committed seed-42 serve files (must remain bit-identical)."""
    seed = config.SEED_MODELS_DIR
    files = [
        seed / "churn_xgb.joblib",
        seed / "calibrator.joblib",
        seed / "metrics.json",
        seed / "feature_names.json",
        seed / "feature_stats.json",
    ]
    docs = [
        config.DOCS_DIR / "MODEL_CARD.md",
        config.DOCS_DIR / "data" / "data-dictionary.md",
    ]
    present = [p for p in files + docs if p.exists()]
    assert present, "expected committed seed-42 artifacts"
    return {p: _sha256(p) for p in present}


def test_apply_artifact_dir_redirects_model_and_docs(tmp_path, seed_fingerprints):
    lake = tmp_path / "lakehouse_run"
    try:
        root = config.apply_artifact_dir(lake)
        assert root == lake.resolve() or root == lake
        assert config.MODELS_DIR == lake
        assert config.ARTIFACTS_DIR == lake
        assert config.MODEL_PATH == lake / "churn_xgb.joblib"
        assert config.METRICS_PATH == lake / "metrics.json"
        assert config.MODEL_CARD_PATH == lake / "MODEL_CARD.md"
        assert config.DATA_DICTIONARY_PATH == lake / "data-dictionary.md"
        assert config.SEED_MODELS_DIR == config.PROJECT_ROOT / "models"

        # Simulate train / docs_gen writes into the alternate dir only.
        config.MODEL_PATH.write_bytes(b"lakehouse-model-bytes")
        config.METRICS_PATH.write_text('{"source":"lakehouse"}\n', encoding="utf-8")
        config.MODEL_CARD_PATH.write_text("# lakehouse card\n", encoding="utf-8")
        config.DATA_DICTIONARY_PATH.write_text("# lakehouse dict\n", encoding="utf-8")
        (lake / "santosh_decision_packet.json").write_text("{}\n", encoding="utf-8")

        assert config.MODEL_PATH.exists()
        assert not (config.SEED_MODELS_DIR / "churn_xgb.joblib").samefile(config.MODEL_PATH)

        for path, digest in seed_fingerprints.items():
            assert path.exists()
            assert _sha256(path) == digest, f"seed artifact mutated: {path}"
    finally:
        config.apply_artifact_dir("")


def test_env_artifact_dir_via_apply(tmp_path, monkeypatch, seed_fingerprints):
    lake = tmp_path / "from_env"
    monkeypatch.setenv("RETENTION_RADAR_ARTIFACT_DIR", str(lake))
    try:
        config.apply_artifact_dir()
        assert config.MODELS_DIR == lake
        config.METRICS_PATH.write_text('{"ok":true}\n', encoding="utf-8")
        for path, digest in seed_fingerprints.items():
            assert _sha256(path) == digest
    finally:
        monkeypatch.delenv("RETENTION_RADAR_ARTIFACT_DIR", raising=False)
        config.apply_artifact_dir("")


def test_restore_seed_paths_after_override(tmp_path):
    lake = tmp_path / "tmp_art"
    config.apply_artifact_dir(lake)
    assert config.MODELS_DIR == lake
    config.apply_artifact_dir("")
    assert config.MODELS_DIR == config.SEED_MODELS_DIR
    assert config.MODEL_PATH == config.SEED_MODELS_DIR / "churn_xgb.joblib"
    assert config.MODEL_CARD_PATH == config.DOCS_DIR / "MODEL_CARD.md"
    assert config.ARTIFACTS_DIR == config.PROJECT_ROOT / "artifacts"


def test_run_all_packet_follows_artifact_dir():
    """run_all.sh must not pin the Santosh packet to artifacts/ — lakehouse E2E
    reads it from RETENTION_RADAR_ARTIFACT_DIR for results/lakehouse-e2e-summary.json."""
    script = (config.PROJECT_ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    packet_lines = [
        ln for ln in script.splitlines()
        if "cli.single_record" in ln and not ln.lstrip().startswith("#")
    ]
    assert packet_lines, "run_all.sh should build the Santosh packet"
    assert all("--out" not in ln for ln in packet_lines)
