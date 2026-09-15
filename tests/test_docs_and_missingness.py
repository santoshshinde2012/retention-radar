"""Docs generator + fail-loud missingness (no full Optuna run)."""

from __future__ import annotations

import pytest

from src.retention_radar.docs_gen import (
    load_user_record_schema,
    schema_field_meta,
    write_data_dictionary,
    write_model_card,
)
from src.retention_radar.features.transform import row_to_feature_frame
from src.generate_data import santosh_profile


def test_schema_meta_covers_required_fields():
    schema = load_user_record_schema()
    assert schema.get("properties")
    dtype, rng, nullable = schema_field_meta("nps_score", schema)
    assert "number" in dtype
    assert "0" in rng and "10" in rng
    assert "required" in nullable
    dtype, rng, nullable = schema_field_meta("plan_tier_code", schema)
    assert dtype == "integer"
    assert "0" in rng


def test_write_dictionary_and_model_card(tmp_path, monkeypatch):
    from src import config

    dict_path = tmp_path / "data-dictionary.md"
    card_path = tmp_path / "model-card.md"
    monkeypatch.setattr(config, "DATA_DICTIONARY_PATH", dict_path)
    monkeypatch.setattr(config, "MODEL_CARD_PATH", card_path)
    monkeypatch.setattr(config, "GUIDES_DIR", tmp_path)

    write_data_dictionary(dict_path)
    text = dict_path.read_text(encoding="utf-8")
    assert "| Dtype |" in text
    assert "| Range |" in text
    assert "Nullable" in text
    assert "no NaNs by design" in text
    assert "`nps_score`" in text

    metrics = {
        "dummy_val": {"roc_auc": 0.5, "f1": 0.0, "average_precision": 0.2},
        "dummy_test": {"roc_auc": 0.5, "f1": 0.0, "average_precision": 0.2},
        "logreg_val": {"roc_auc": 0.9, "f1": 0.6, "average_precision": 0.7},
        "logreg_test": {"roc_auc": 0.88, "f1": 0.58, "average_precision": 0.69},
        "xgb_default_val": {"roc_auc": 0.87, "f1": 0.55, "average_precision": 0.65},
        "xgb_default_test": {"roc_auc": 0.86, "f1": 0.54, "average_precision": 0.64},
        "tuned_val": {"roc_auc": 0.91, "f1": 0.66, "average_precision": 0.72},
        "tuned_test": {"roc_auc": 0.87, "f1": 0.58, "average_precision": 0.70},
        "calibrated_val": {"roc_auc": 0.91, "f1": 0.66, "average_precision": 0.72},
        "calibrated_test": {"roc_auc": 0.86, "f1": 0.59, "average_precision": 0.70},
        "brier_raw_val": 0.1,
        "brier_calibrated_val": 0.08,
        "brier_raw_test": 0.12,
        "brier_calibrated_test": 0.10,
        "calibration_method": "isotonic",
        "n_train": 3000,
        "n_val": 1000,
        "n_test": 1000,
        "n_trials": 20,
        "churn_rate_train": 0.19,
        "average_precision_test": 0.70,
        "best_f1_threshold": 0.34,
        "best_f1_at_threshold": 0.61,
        "evaluate_test": {
            "test_roc_auc": 0.86,
            "test_average_precision": 0.70,
        },
    }
    write_model_card(metrics, card_path)
    card = card_path.read_text(encoding="utf-8")
    for name in (
        "roc_curve.png",
        "pr_curve.png",
        "calibration_curve.png",
        "confusion_matrix.png",
        "threshold_f1.png",
    ):
        assert name in card
    assert "Dummy (prior)" in card
    assert "XGBoost (default)" in card
    assert "Best F1 threshold" in card
    assert "0.8800" in card  # logreg test AUC


def test_serve_fails_loud_on_nan():
    profile = {k: v for k, v in santosh_profile().items() if k != "churned"}
    profile["nps_score"] = None
    with pytest.raises(ValueError, match="NaNs"):
        row_to_feature_frame(profile)


def test_serve_fails_loud_on_non_numeric():
    profile = {k: v for k, v in santosh_profile().items() if k != "churned"}
    profile["sessions_last_7d"] = "not-a-number"
    with pytest.raises(ValueError, match="NaNs"):
        row_to_feature_frame(profile)


def test_valid_santosh_row_has_no_nans():
    profile = {k: v for k, v in santosh_profile().items() if k != "churned"}
    X = row_to_feature_frame(profile)
    assert not X.isna().any().any()
    assert len(X) == 1
