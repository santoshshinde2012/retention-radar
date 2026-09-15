"""Auto-generate model card and data dictionary from config + metrics + schema."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from retention_radar import config


def load_user_record_schema() -> dict:
    path = config.USER_RECORD_SCHEMA_PATH
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def schema_field_meta(col: str, schema: dict) -> tuple[str, str, str]:
    """Return (dtype, range, nullable) from JSON Schema + derived-column notes."""
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    if col == "plan_tier_code":
        return "integer", "0–3 (free…enterprise)", "no (derived at transform)"
    if col == "churned":
        return "integer (0/1)", "0 or 1", "no in training CSV; omitted on serve"

    prop = props.get(col, {})
    raw_type = prop.get("type", "")
    if isinstance(raw_type, list):
        dtype = " | ".join(str(t) for t in raw_type)
    else:
        dtype = str(raw_type) if raw_type else "unknown"

    if "enum" in prop:
        rng = "enum: " + ", ".join(str(x) for x in prop["enum"])
    elif "minimum" in prop or "maximum" in prop:
        lo = prop.get("minimum", "—")
        hi = prop.get("maximum", "—")
        rng = f"[{lo}, {hi}]"
    elif "minLength" in prop:
        rng = f"minLength={prop['minLength']}"
    else:
        hint = config.FEATURE_RANGES.get(col)
        rng = f"[{hint[0]}, {hint[1]}]" if hint else "—"

    if col in required:
        nullable = "no (required on serve)"
    elif col in props:
        nullable = "yes (not required in schema)"
    else:
        nullable = "n/a"

    return dtype, rng, nullable


def _fmt(block: dict | None, key: str = "roc_auc") -> str:
    if not block:
        return "n/a"
    val = block.get(key, float("nan"))
    try:
        return f"{float(val):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _fmt_scalar(val, digits: int = 4) -> str:
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_data_dictionary(path: Path | None = None) -> Path:
    path = path or config.DATA_DICTIONARY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    config.GUIDES_DIR.mkdir(parents=True, exist_ok=True)
    schema = load_user_record_schema()

    lines = [
        "# Data dictionary",
        "",
        "Auto-generated from `src/retention_radar/config.py`, "
        "`schemas/user_record.schema.json`, and `src/retention_radar/data/generate.py`.",
        "Synthetic AI-platform churn dataset — no real PII.",
        "",
        f"_Generated: {date.today().isoformat()} · seed={config.RANDOM_SEED}_",
        "",
        "## Missingness policy",
        "",
        "- **Synthetic generator:** every feature is populated; **no NaNs by design** "
        "(see `src/retention_radar/data/generate.py`).",
        "- **Train / serve:** `prepare_xy` and `row_to_feature_frame` **fail loud** "
        "if any model feature is NaN after encoding. There is no silent impute.",
        "- **Recommended real-data pattern:** add missingness indicators "
        "(e.g. `nps_missing`) and fit an imputer **on train only**, persist it "
        "beside the model, then apply the same transform at serve. Do not impute "
        "from a single Streamlit row.",
        "",
        "## Columns",
        "",
        "| Column | Role | Dtype | Range | Nullable | Description |",
        "|--------|------|-------|-------|----------|-------------|",
    ]

    def role(col: str) -> str:
        if col in config.ID_COLUMNS:
            return "id"
        if col == config.TARGET_COLUMN:
            return "target"
        if col in config.FEATURE_COLUMNS or col == "plan_tier_code":
            return "feature"
        return "other"

    all_cols = list(
        dict.fromkeys(
            config.ID_COLUMNS
            + config.FEATURE_COLUMNS
            + ["plan_tier_code", config.TARGET_COLUMN]
        )
    )
    for col in all_cols:
        desc = config.FEATURE_DESCRIPTIONS.get(col, "")
        dtype, rng, nullable = schema_field_meta(col, schema)
        lines.append(
            f"| `{col}` | {role(col)} | `{dtype}` | {rng} | {nullable} | {desc} |"
        )

    lines += [
        "",
        "## Model feature vector (encoded)",
        "",
        "Order used at train / infer time:",
        "",
    ]
    for i, name in enumerate(config.MODEL_FEATURE_COLUMNS):
        lines.append(f"{i + 1}. `{name}`")

    lines += [
        "",
        "## Plan tier encoding",
        "",
        "| Tier | Code |",
        "|------|------|",
    ]
    for i, name in enumerate(config.PLAN_TIER_ORDER):
        lines.append(f"| `{name}` | {i} |")

    try:
        from retention_radar.data.generate import santosh_profile

        profile = {k: v for k, v in santosh_profile().items() if k != "churned"}
        lines += [
            "",
            "## Santosh Shinde — feature contract (inference)",
            "",
            "Payload: `data/raw/santosh_shinde.json` (no `churned`).",
            "",
            "| Field | Example value |",
            "|-------|----------------|",
        ]
        for k, v in profile.items():
            lines.append(f"| `{k}` | {v} |")
        lines += [
            "",
            "See also [santosh-case-study.md](../case-study/santosh-case-study.md) and "
            "[single-record-checklist.md](../case-study/single-record-checklist.md).",
            "",
        ]
    except Exception:
        pass

    lines += [
        "",
        "## Related reading",
        "",
        "- [ARCHITECTURE.md](../ARCHITECTURE.md) — SOLID package map",
        "- [MODEL_CARD.md](../MODEL_CARD.md)",
        "- [Santosh case study](../case-study/santosh-case-study.md)",
        "- [Single-record checklist](../case-study/single-record-checklist.md)",
        "- [Data foundation / lakehouse](data-foundation-lakehouse.md)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_model_card(metrics: dict | None = None, path: Path | None = None) -> Path:
    path = path or config.MODEL_CARD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    config.GUIDES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = metrics or {}
    ev = metrics.get("evaluate_test") or {}
    cal_test = metrics.get("calibrated_test") or {}
    # Fallbacks: train may call write_model_card before evaluate_test exists.
    op_auc = ev.get("test_roc_auc")
    if op_auc is None:
        op_auc = cal_test.get("roc_auc")
    op_ap = ev.get("test_average_precision") or metrics.get("average_precision_test")
    if op_ap is None:
        op_ap = cal_test.get("average_precision")

    lines = [
        "# Model card — AI platform churn (XGBoost)",
        "",
        f"_Generated: {date.today().isoformat()} · seed={config.RANDOM_SEED}_",
        "",
        "## Overview",
        "",
        "Binary classifier predicting whether a synthetic AI-platform user will churn.",
        "Stack: XGBoost (Optuna-tuned) + post-hoc probability calibration "
        f"(`{metrics.get('calibration_method', config.CALIBRATION_METHOD)}`) "
        "on the validation set.",
        "",
        "## Intended use",
        "",
        "- Teaching / FOSS case study for churn ranking and calibrated probabilities.",
        "- Interactive Streamlit what-if on the Santosh Shinde hero profile.",
        "- **Not** for production decisions on real customers without fresh data validation.",
        "",
        "## Training data",
        "",
        "- Synthetic users from `src/retention_radar/data/generate.py` "
        "(rule-based propensity + noise). **No NaNs by design.**",
        f"- Split: train ~{(1 - config.TEST_SIZE - config.VAL_SIZE) * 100:.0f}% / "
        f"val ~{config.VAL_SIZE * 100:.0f}% / test ~{config.TEST_SIZE * 100:.0f}% "
        f"(stratified, `random_state={config.RANDOM_SEED}`).",
        f"- n_train={metrics.get('n_train', '?')} · n_val={metrics.get('n_val', '?')} · "
        f"n_test={metrics.get('n_test', '?')} · n_trials={metrics.get('n_trials', '?')}.",
        f"- Train churn rate: `{metrics.get('churn_rate_train', 'n/a')}`.",
        "",
        "## Features",
        "",
        f"{len(config.MODEL_FEATURE_COLUMNS)} numeric features after ordinal "
        "`plan_tier` encoding.",
        "See [data-dictionary.md](data/data-dictionary.md) for dtype, ranges, and nullability.",
        "",
        "## Metrics (holdout) — from `models/metrics.json`",
        "",
        "Honest ladder: **Dummy(prior) → LogReg → RF → default XGB → Optuna XGB → LightGBM → CatBoost** "
        "(calibrated XGB is the Santosh / serving hero). No simple-rule baseline is logged. "
        "GBDT trilogy peers: XGB / LightGBM / CatBoost.",
        "",
        "| Model | Val AUC | Val F1 | Test AUC | Test F1 | Test PR-AUC |",
        "|-------|---------|--------|----------|---------|-------------|",
        f"| Dummy (prior) | {_fmt(metrics.get('dummy_val'))} | "
        f"{_fmt(metrics.get('dummy_val'), 'f1')} | "
        f"{_fmt(metrics.get('dummy_test'))} | "
        f"{_fmt(metrics.get('dummy_test'), 'f1')} | "
        f"{_fmt(metrics.get('dummy_test'), 'average_precision')} |",
        f"| Logistic regression | {_fmt(metrics.get('logreg_val'))} | "
        f"{_fmt(metrics.get('logreg_val'), 'f1')} | "
        f"{_fmt(metrics.get('logreg_test'))} | "
        f"{_fmt(metrics.get('logreg_test'), 'f1')} | "
        f"{_fmt(metrics.get('logreg_test'), 'average_precision')} |",
        f"| Random Forest | {_fmt(metrics.get('rf_val'))} | "
        f"{_fmt(metrics.get('rf_val'), 'f1')} | "
        f"{_fmt(metrics.get('rf_test'))} | "
        f"{_fmt(metrics.get('rf_test'), 'f1')} | "
        f"{_fmt(metrics.get('rf_test'), 'average_precision')} |",
        f"| XGBoost (default) | {_fmt(metrics.get('xgb_default_val'))} | "
        f"{_fmt(metrics.get('xgb_default_val'), 'f1')} | "
        f"{_fmt(metrics.get('xgb_default_test'))} | "
        f"{_fmt(metrics.get('xgb_default_test'), 'f1')} | "
        f"{_fmt(metrics.get('xgb_default_test'), 'average_precision')} |",
        f"| XGBoost (Optuna, raw) | {_fmt(metrics.get('tuned_val'))} | "
        f"{_fmt(metrics.get('tuned_val'), 'f1')} | "
        f"{_fmt(metrics.get('tuned_test'))} | "
        f"{_fmt(metrics.get('tuned_test'), 'f1')} | "
        f"{_fmt(metrics.get('tuned_test'), 'average_precision')} |",
        f"| LightGBM (default) | {_fmt(metrics.get('lgbm_val'))} | "
        f"{_fmt(metrics.get('lgbm_val'), 'f1')} | "
        f"{_fmt(metrics.get('lgbm_test'))} | "
        f"{_fmt(metrics.get('lgbm_test'), 'f1')} | "
        f"{_fmt(metrics.get('lgbm_test'), 'average_precision')} |",
        f"| CatBoost (default) | {_fmt(metrics.get('catboost_val'))} | "
        f"{_fmt(metrics.get('catboost_val'), 'f1')} | "
        f"{_fmt(metrics.get('catboost_test'))} | "
        f"{_fmt(metrics.get('catboost_test'), 'f1')} | "
        f"{_fmt(metrics.get('catboost_test'), 'average_precision')} |",
        f"| XGBoost (calibrated) | {_fmt(metrics.get('calibrated_val'))} | "
        f"{_fmt(metrics.get('calibrated_val'), 'f1')} | "
        f"{_fmt(metrics.get('calibrated_test'))} | "
        f"{_fmt(metrics.get('calibrated_test'), 'f1')} | "
        f"{_fmt_scalar(op_ap)} |",
        "",
        "### Calibration (Brier — lower is better)",
        "",
        "| Split | Raw Brier | Calibrated Brier |",
        "|-------|-----------|------------------|",
        f"| val | `{_fmt_scalar(metrics.get('brier_raw_val'))}` | "
        f"`{_fmt_scalar(metrics.get('brier_calibrated_val'))}` |",
        f"| test | `{_fmt_scalar(metrics.get('brier_raw_test'))}` | "
        f"`{_fmt_scalar(metrics.get('brier_calibrated_test'))}` |",
        "",
        f"Method: `{metrics.get('calibration_method', config.CALIBRATION_METHOD)}`.",
        "",
        "### Operating point (calibrated test scores)",
        "",
        "| Quantity | Value |",
        "|----------|-------|",
        f"| Test AUC-ROC (calibrated) | `{_fmt_scalar(op_auc)}` |",
        f"| Test PR-AUC / AP | `{_fmt_scalar(op_ap)}` |",
        f"| Best F1 threshold τ | `{_fmt_scalar(metrics.get('best_f1_threshold'), 2)}` |",
        f"| Best F1 at τ | `{_fmt_scalar(metrics.get('best_f1_at_threshold'))}` |",
        f"| F1 @ 0.5 | `{_fmt(metrics.get('calibrated_test'), 'f1')}` |",
    ]
    lat = metrics.get("latency") or {}
    if lat.get("latency_ms_p50") is not None:
        lines.append(
            f"| Warm latency p50 / p95 (ms) | "
            f"`{_fmt_scalar(lat.get('latency_ms_p50'), 2)}` / "
            f"`{_fmt_scalar(lat.get('latency_ms_p95'), 2)}` |"
        )
    lines += [
        "",
        "## Result plots",
        "",
        "Committed copies live under `results/plots/` (runtime dumps in `artifacts/`).",
        "",
        "![ROC](../results/plots/roc_curve.png)",
        "",
        "![Precision–Recall](../results/plots/pr_curve.png)",
        "",
        "![Calibration](../results/plots/calibration_curve.png)",
        "",
        "![Confusion matrix](../results/plots/confusion_matrix.png)",
        "",
        "![Threshold vs F1](../results/plots/threshold_f1.png)",
        "",
        "## Artifacts",
        "",
        "| Path | Contents |",
        "|------|----------|",
        "| `models/churn_xgb.joblib` | Tuned XGBoost + metadata |",
        "| `models/calibrator.joblib` | Validation-fit probability calibrator |",
        "| `models/metrics.json` | Full metric dump (source of truth) |",
        "| `artifacts/roc_curve.png` | ROC |",
        "| `artifacts/pr_curve.png` | Precision–Recall |",
        "| `artifacts/calibration_curve.png` | Reliability diagram |",
        "| `artifacts/confusion_matrix.png` | Confusion @ 0.5 |",
        "| `artifacts/threshold_f1.png` | Threshold vs F1 / precision / recall |",
        "| `results/plots/*.png` | Committed copies of the same plots |",
        "",
        "## Ethical notes",
        "",
        "- Labels and features are synthetic; do not treat scores as real risk.",
        "- Calibration improves probability meaning but does not fix selection bias.",
        "- SHAP explains this score, not causation.",
        "- Single-record path is HITL only (`auto_action: none`) — "
        "see [single-record-checklist.md](case-study/single-record-checklist.md).",
        "",
        "## Related reading",
        "",
        "- [ARCHITECTURE.md](ARCHITECTURE.md) — SOLID package map",
        "- [data-dictionary.md](data/data-dictionary.md)",
        "- [BEST_PRACTICES.md](guides/BEST_PRACTICES.md)",
        "- [santosh-case-study.md](case-study/santosh-case-study.md)",
        "- [ALGORITHM_LANDSCAPE.md](guides/ALGORITHM_LANDSCAPE.md) — what is on the ladder vs deferred",
        "- [../results/BENCHMARKS.md](../results/BENCHMARKS.md)",
        "- [../results/SANTOSH_ANALYSIS.md](../results/SANTOSH_ANALYSIS.md)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def refresh_docs_from_metrics_file(metrics_path: Path) -> None:
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
    write_data_dictionary()
    write_model_card(metrics)


def main() -> None:
    refresh_docs_from_metrics_file(Path(config.METRICS_PATH))
    print(f"Wrote data dictionary and model card from {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
