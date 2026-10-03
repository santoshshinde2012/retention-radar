"""Docs generator + fail-loud missingness (no full Optuna run)."""

from __future__ import annotations

import pytest

from retention_radar.data.generate import santosh_profile
from retention_radar.docs_gen import (
    load_user_record_schema,
    schema_field_meta,
    write_data_dictionary,
    write_model_card,
)
from retention_radar.features.transform import row_to_feature_frame


def test_schema_meta_covers_required_fields():
    schema = load_user_record_schema()
    assert schema.get("properties")
    dtype, rng, nullable = schema_field_meta("limit_hits_14d", schema)
    assert "number" in dtype
    assert "0" in rng and "60" in rng
    assert "required" in nullable
    dtype, rng, nullable = schema_field_meta("plan_tier_code", schema)
    assert dtype == "integer"
    assert "0" in rng


def test_write_dictionary_and_model_card(tmp_path, monkeypatch):
    from retention_radar import config

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
    assert "`limit_hits_14d`" in text
    assert "dunning" in text and "cancel flow" in text  # label routing is documented
    assert "Santosh" in text or "santosh" in text

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
        "calibration_method": "sigmoid",
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
    profile = santosh_profile()
    profile["suggestion_accept_rate_28d"] = None
    with pytest.raises(ValueError, match="NaNs"):
        row_to_feature_frame(profile)


def test_serve_fails_loud_on_non_numeric():
    profile = santosh_profile()
    profile["active_days_7d"] = "not-a-number"
    with pytest.raises(ValueError, match="NaNs"):
        row_to_feature_frame(profile)


def test_valid_worked_example_row_has_no_nans():
    profile = santosh_profile()
    X = row_to_feature_frame(profile)
    assert not X.isna().any().any()
    assert len(X) == 1



def test_markdown_tables_are_not_split_by_paragraphs():
    """A table row right after a paragraph line renders as plain text on GitHub."""
    from retention_radar import config as cfg

    root = cfg.PROJECT_ROOT
    paths = [root / "README.md", *sorted((root / "docs").rglob("*.md")), *sorted((root / "results").glob("*.md"))]
    for path in paths:
        fenced = False
        lines = path.read_text(encoding="utf-8").splitlines()
        for prev, line in zip(lines, lines[1:]):
            if prev.startswith("```"):
                fenced = not fenced
            if not fenced and line.startswith("|") and prev.strip() and not prev.startswith("|"):
                raise AssertionError(f"{path.name}: table row follows a paragraph: {prev[:60]!r}")
