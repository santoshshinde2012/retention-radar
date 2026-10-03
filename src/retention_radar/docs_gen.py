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
        return "integer", "0–2 (pro, pro_plus, ultra)", "no (derived at transform)"
    if col == "churned":
        return "integer (0/1)", "0 or 1", "no in training CSV; omitted on serve (renewal is ahead)"

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
        "`configs/schemas/user_record.schema.json`, and `src/retention_radar/data/generate.py`.",
        "One row per paying subscriber of a monthly AI coding assistant plan, snapshotted "
        "seven days before renewal (T-7). Synthetic, no real PII.",
        "",
        "## Label and routing",
        "",
        "- `churned` = 1 when the subscriber **voluntarily** let the plan lapse at this renewal.",
        "- Renewals that lapsed because the card failed and retries ran out are routed to "
        "**dunning** and excluded (a payments problem, not a behaviour signal).",
        "- Subscribers who had already scheduled a cancel before T-7 are routed to the "
        "**cancel flow** and excluded (the outcome is decided; keeping them would leak it).",
        "- `data/raw/renewals_all.csv` keeps every renewal with `outcome` and `route` for audit.",
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
        "(e.g. `accept_rate_missing` for users who turned inline suggestions off) and fit an imputer **on train only**, persist it "
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
        from retention_radar.data.generate import HERO_PROFILES

        lines += [
            "",
            "## Worked examples (scoring-time records, no label)",
            "",
            "| Field | " + " | ".join(HERO_PROFILES) + " |",
            "|-------|" + "|".join("---" for _ in HERO_PROFILES) + "|",
        ]
        profiles = {k: fn() for k, fn in HERO_PROFILES.items()}
        for field in config.INFERENCE_REQUIRED_KEYS:
            lines.append(
                f"| `{field}` | " + " | ".join(str(profiles[k][field]) for k in profiles) + " |"
            )
        lines += [
            "",
            "Payloads: `data/raw/subscribers/*.json`. Walkthrough: "
            "[renewal-worked-examples.md](../case-study/renewal-worked-examples.md).",
            "",
        ]
    except Exception:
        pass

    lines += [
        "",
        "## Related reading",
        "",
        "- [architecture.md](../architecture.md) — SOLID package map",
        "- [model-card.md](../model-card.md)",
        "- [Worked examples](../case-study/renewal-worked-examples.md)",
        "- [Single-record checklist](../case-study/single-record-checklist.md)",
        "- [Data foundation / lakehouse](data-foundation-lakehouse.md)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _extra_sections(metrics: dict) -> list[str]:
    """Training setup, segment and policy numbers, the lakehouse run and the holdout design (from files, never typed)."""
    out: list[str] = []
    bp = metrics.get("best_params") or {}
    if bp:
        out += ["", "## Training setup", "",
                f"Optuna ({metrics.get('n_trials', '?')} trials, objective: validation AUC, best "
                f"{_fmt_scalar(metrics.get('best_optuna_auc'))}) chose these XGBoost parameters:", "",
                "| Parameter | Value |", "|---|---|"]
        out += [f"| `{k}` | {_fmt_scalar(v) if isinstance(v, float) else v} |" for k, v in bp.items()]
        out += ["", f"The threshold τ = {_fmt_scalar(metrics.get('best_f1_threshold'), 2)} is chosen on the "
                f"{metrics.get('best_f1_threshold_source', 'validation')} split (best F1) and applied once to test."]
    top = metrics.get("capture_at_top_10pct")
    if top is not None:
        out += ["", "### Ranking (test)", "",
                f"- Base rate: {_fmt_scalar(metrics.get('base_rate_test'))}.",
                f"- Top 10% by score: precision {_fmt_scalar(metrics.get('precision_at_top_10pct'))}, "
                f"capturing {_fmt_scalar(top)} of all lapses."]
    sl = (metrics.get("slice_metrics_by_plan_tier") or {}).get("by_plan_tier") or {}
    if sl:
        out += ["", "### By plan (test, at τ)", "",
                "Segment diagnostics only, not a fairness audit.", "",
                "| Plan | n | Lapse rate | Precision | Recall | ROC AUC |", "|---|---:|---:|---:|---:|---:|"]
        for tier, v in sl.items():
            auc = _fmt_scalar(v.get("roc_auc")) if v.get("enough_samples_for_auc") else "too few"
            out.append(f"| {tier} | {v.get('n')} | {_fmt_scalar(v.get('churn_rate'))} | "
                       f"{_fmt_scalar(v.get('precision'))} | {_fmt_scalar(v.get('recall'))} | {auc} |")
    pol = metrics.get("policy_on_test") or {}
    if pol.get("actions"):
        out += ["", "### Policy on the test set", "",
                f"n = {pol.get('n')}; contacted share {_fmt_scalar(pol.get('contacted_share'))}; expected value "
                f"${pol.get('expected_value_usd_total')} (assumed playbook effects).", "",
                "| Action | Rows |", "|---|---:|"]
        out += [f"| `{a}` | {n} |" for a, n in sorted(pol["actions"].items(), key=lambda kv: -kv[1])]
    out += ["", "## Holdout and control design", "",
            f"- {config.HOLDOUT_PCT}% of eligible subscribers are a fixed control group: a subscriber is in it when the first "
            "8 hex digits of `sha256(\"holdout:<user_id>\")` mod 100 fall below the holdout share "
            "(`serving/policy.py`). The same id always lands in the same group.",
            "- Holdout rows are scored and logged with the action the policy would have taken (`would_have_sent`), "
            "but nothing is sent. Lift is the outcome gap between treated and holdout rows (`cli.outcomes`).",
            "- The worked example `sub_santosh` is in the holdout: medium risk, `would_have_sent: limit_reset`."]
    summary = config.RESULTS_DIR / "lakehouse_e2e_summary.json" if hasattr(config, "RESULTS_DIR") else None
    if summary is not None and summary.exists():
        lk = json.loads(summary.read_text(encoding="utf-8"))
        out += ["", "## Lakehouse data (separate run)", "",
                "The same pipeline also trains on the feature export of "
                "[local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) (synthetic events "
                "turned into point-in-time gold). That run writes only under `artifacts/lakehouse_run/`; the "
                "committed bundle above stays on the synthetic seed-42 data.", "",
                "| Quantity | Value |", "|---|---|",
                f"| Source | {lk.get('source')} (verified {lk.get('verified_at')}) |",
                f"| n_train / n_test | {lk.get('n_train')} / {lk.get('n_test')} |",
                f"| Train lapse rate | {_fmt_scalar(lk.get('churn_rate_train'))} |",
                f"| Best Optuna AUC (val) | {_fmt_scalar(lk.get('best_optuna_auc_val'))} |",
                f"| Calibrated test ROC AUC | {_fmt_scalar(lk.get('calibrated_test_roc_auc'))} |",
                f"| Threshold τ | {lk.get('best_f1_threshold')} |",
                "", "Source: [`results/lakehouse_e2e_summary.json`](../results/lakehouse_e2e_summary.json)."]
    return out


def _data_limits(metrics: dict) -> list[str]:
    out = []
    sl = (metrics.get("slice_metrics_by_plan_tier") or {}).get("by_plan_tier") or {}
    thin = [f"`{t}` ({v.get('n_positive')} lapses in {v.get('n')} test rows)" for t, v in sl.items()
            if (v.get("n_positive") or 0) < 20]
    if thin:
        out.append("- Thin segments: " + ", ".join(thin) + ". Treat their numbers as anecdotes.")
    stats = Path(config.METRICS_PATH).parent / "feature_stats.json"
    if stats.exists():
        fs = json.loads(stats.read_text(encoding="utf-8"))
        if not any(isinstance(v, dict) and "psi_bins" in v for v in fs.values()):
            out.append("- Drift checks fall back to the standardised mean difference: the committed "
                       "`models/feature_stats.json` has no PSI bins.")
    return out


def _versioning_section(metrics: dict) -> list[str]:
    """Bundle files with sha256 prefixes and the pinned model libraries (read from disk at generation time)."""
    import hashlib
    out = ["## Versioning", "",
           f"The served bundle is the committed `models/` directory (seed {metrics.get('random_seed', config.RANDOM_SEED)}). "
           "It changes only with a retrain, recorded in the CHANGELOG; `make reproduce` retrains into `artifacts/repro/` "
           "and diffs against `models/metrics.json`.", "",
           "| File | sha256 (first 12) |", "|---|---|"]
    models_dir = Path(config.METRICS_PATH).parent
    for f in sorted(models_dir.glob("*")):
        if f.is_file() and f.suffix in (".joblib", ".json"):
            out.append(f"| `models/{f.name}` | `{hashlib.sha256(f.read_bytes()).hexdigest()[:12]}` |")
    req = config.PROJECT_ROOT / "requirements.txt" if hasattr(config, "PROJECT_ROOT") else None
    if req is not None and req.exists():
        pins = [ln.strip() for ln in req.read_text(encoding="utf-8").splitlines() if "==" in ln and not ln.startswith("#")]
        if pins:
            out += ["", "Pinned model libraries (`requirements.txt`): " + ", ".join(f"`{p}`" for p in pins) + "."]
    return out


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
        "# Model card: voluntary lapse at renewal (AI coding assistant, XGBoost)",
        "",
        f"_Generated: {date.today().isoformat()} · seed={config.RANDOM_SEED}_",
        "",
        "## Overview",
        "",
        "Binary classifier scoring, seven days before a monthly renewal, whether a paying "
        "subscriber of a self-serve AI coding assistant will voluntarily let the plan lapse.",
        "Stack: XGBoost (Optuna-tuned) + post-hoc probability calibration "
        f"(`{metrics.get('calibration_method', config.CALIBRATION_METHOD)}`) "
        "on the validation set.",
        "",
        "## Intended use",
        "",
        "- Teaching / FOSS case study: rank T-7 renewals, calibrate, and feed an expected-value "
        "policy that picks an approved playbook, a holdout, or no action.",
        "- The score says who is at risk, not who will respond. Playbook effects in "
        "`config.PLAYBOOKS` are assumptions until a holdout measures them.",
        "- **Not** for production decisions on real customers without fresh data validation.",
        "",
        "## Training data",
        "",
        "- Synthetic renewal cohort from `src/retention_radar/data/generate.py`: unobserved "
        "causes (need, fit, price sensitivity, side-project habit, a rival tool's pull) drive "
        "observed usage and a noisy voluntary-lapse outcome. **No NaNs by design.**",
        "- Dunning (failed-payment) lapses and already-scheduled cancels are excluded; see the data dictionary.",
        f"- Split: train ~{(1 - config.TEST_SIZE - config.VAL_SIZE) * 100:.0f}% / "
        f"val ~{config.VAL_SIZE * 100:.0f}% / test ~{config.TEST_SIZE * 100:.0f}% "
        f"(stratified, `random_state={config.RANDOM_SEED}`).",
        f"- n_train={metrics.get('n_train', '?')} · n_val={metrics.get('n_val', '?')} · "
        f"n_test={metrics.get('n_test', '?')} · n_trials={metrics.get('n_trials', '?')}.",
        f"- Train voluntary-lapse rate: `{metrics.get('churn_rate_train', 'n/a')}`.",
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
        "(calibrated XGB serves). No simple-rule baseline is logged. "
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
        "F1 columns use a 0.5 threshold. Calibrated scores sit near the ~9% base rate and rarely reach 0.5, so the "
        "calibrated row shows F1 0; the served threshold τ is in the operating point below.",
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
        f"| Best F1 threshold τ (chosen on validation) | `{_fmt_scalar(metrics.get('best_f1_threshold'), 2)}` |",
        f"| F1 at τ (validation) | `{_fmt_scalar(metrics.get('val_f1_at_threshold'))}` |",
        f"| F1 at τ (test, reported once) | `{_fmt_scalar(metrics.get('best_f1_at_threshold'))}` |",
        f"| F1 @ 0.5 | `{_fmt(metrics.get('calibrated_test'), 'f1')}` |",
    ]
    lat = metrics.get("latency") or {}
    if lat.get("latency_ms_p50") is not None:
        lines.append(
            f"| Warm latency p50 / p95 (ms) | "
            f"`{_fmt_scalar(lat.get('latency_ms_p50'), 2)}` / "
            f"`{_fmt_scalar(lat.get('latency_ms_p95'), 2)}` |"
        )
    lines += _extra_sections(metrics)
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
        "- The service never executes an action (`auto_action: none`); a lifecycle tool runs "
        "human-approved playbooks, and a deterministic holdout is kept out of every playbook.",
        "- Contacting at-risk subscribers can raise churn (Ascarza et al., JMR 2016). "
        "Measure lift against the holdout before scaling any playbook.",
        "",
        "## Limitations",
        "",
        "- Trained and evaluated on synthetic data only; there is no labelled real-world outcome.",
        "- One seed (42) and one split. With about 140 lapses in test, AUC differences of a few points between ladder rows are within noise.",
        "- Playbook effects and costs in `config.PLAYBOOKS` are assumptions, so the expected value is too.",
        *_data_limits(metrics),
        "- Latency is one machine's warm in-process timing, not a service SLA.",
        "",
        *_versioning_section(metrics),
        "",
        "## Related reading",
        "",
        "- [architecture.md](architecture.md) — SOLID package map",
        "- [data-dictionary.md](data/data-dictionary.md)",
        "- [best-practices.md](guides/best-practices.md)",
        "- [renewal-worked-examples.md](case-study/renewal-worked-examples.md)",
        "- [algorithm-landscape.md](guides/algorithm-landscape.md) — what is on the ladder vs deferred",
        "- [../results/benchmarks.md](../results/benchmarks.md)",
        "- [../results/worked-examples.md](../results/worked-examples.md)",
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
