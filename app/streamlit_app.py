"""Streamlit demo: Predict / Explain / Decision / Methodology / Benchmarks.

Run from project root:
    export PYTHONPATH="$(pwd)/src"
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retention_radar import config as rr_config  # noqa: E402
from retention_radar.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CALIBRATOR_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PLAN_TIER_ORDER,
)
from retention_radar.data.ingest import (  # noqa: E402
    resolve_santosh_json,
    resolve_users_csv,
)
from retention_radar.serving.packet import build_decision_packet  # noqa: E402
from retention_radar.training.calibrate import load_calibrator  # noqa: E402


@st.cache_resource
def load_model_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_cal():
    return load_calibrator(CALIBRATOR_PATH)


def load_santosh_defaults() -> dict:
    path = resolve_santosh_json()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {
        "user_id": "santosh_shinde",
        "user_name": "Santosh Shinde",
        "days_since_signup": 420,
        "sessions_last_7d": 9,
        "sessions_last_30d": 38,
        "avg_session_minutes": 28.5,
        "models_used_count": 7,
        "api_calls_last_30d": 1850,
        "tokens_consumed_last_30d": 420000,
        "tools_used_count": 8,
        "failed_requests_rate": 0.12,
        "support_tickets_last_90d": 2,
        "plan_tier": "pro",
        "payment_failures_last_90d": 1,
        "feature_adoption_score": 0.78,
        "nps_score": 7.0,
        "last_active_days_ago": 8,
        "weekend_usage_ratio": 0.22,
        "engagement_trend": 0.9474,
        "spend_usd_last_30d": 189.0,
        "days_until_renewal": 21,
        "agent_runs_last_30d": 52,
        "ide_plugin_sessions_last_30d": 28,
        "seat_utilization": 0.72,
    }


def load_metrics() -> dict:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


USE_CASES_PATH = ROOT / "data" / "use_cases" / "personas.json"


def load_use_cases() -> list[dict]:
    """Scenario presets from data/use_cases/personas.json (empty if the pack is absent)."""
    try:
        with open(USE_CASES_PATH, encoding="utf-8") as f:
            return json.load(f).get("personas", [])
    except (OSError, ValueError):
        return []


def choose_preset() -> dict:
    """Sidebar picker: Santosh from the active data source, then the use-case pack."""
    santosh = load_santosh_defaults()
    presets = [
        {
            "id": "santosh_default",
            "title": f"Santosh Shinde ({rr_config.CHURN_DATA_SOURCE} source)",
            "story": "Hero record from the active data source.",
            "expected": None,
            "record": santosh,
        }
    ]
    for p in load_use_cases():
        if p["record"] == santosh:
            presets[0]["expected"] = p.get("expected")  # same record as the hero
            continue
        presets.append(p)
    titles = [p["title"] for p in presets]
    title = st.sidebar.selectbox("Use case", titles, index=0, key="use_case")
    preset = presets[titles.index(title)]
    st.sidebar.caption(preset.get("story", ""))
    exp = preset.get("expected")
    if exp:
        st.sidebar.caption(
            f"Committed bundle: `{exp['band']}` → `{exp['hitl_action']}` "
            f"(p_cal {exp['p_cal']:.3f}). Move a slider to explore."
        )
    return preset


def _slider(preset_id: str, label: str, key: str, lo, hi, defaults: dict, fallback, step=None, as_int=False):
    """Slider keyed per preset; bounds widen to fit the record so it is never clipped."""
    raw = defaults.get(key, fallback)
    value = int(round(float(raw))) if as_int else float(raw)
    lo, hi = min(lo, value), max(hi, value)
    kwargs = {"key": f"{preset_id}:{key}"}
    if step is not None:
        kwargs["step"] = step
    return st.slider(label, lo, hi, value, **kwargs)


def collect_user_inputs(defaults: dict, preset_id: str = "santosh_default") -> dict:
    st.sidebar.header("What-if")
    st.sidebar.caption("Pick a use case, then tweak sliders to see how risk changes.")
    user_name = st.sidebar.text_input(
        "Name", value=defaults.get("user_name", "Santosh Shinde"), key=f"{preset_id}:user_name"
    )
    plan_tier = st.sidebar.selectbox(
        "Plan tier",
        PLAN_TIER_ORDER,
        index=PLAN_TIER_ORDER.index(defaults.get("plan_tier", "pro")),
        key=f"{preset_id}:plan_tier",
    )

    def num(label, key, lo, hi, fallback, step=None, as_int=False):
        return _slider(preset_id, label, key, lo, hi, defaults, fallback, step=step, as_int=as_int)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Usage")
        days_since_signup = num("Days since signup", "days_since_signup", 14, 900, 420, as_int=True)
        sessions_last_7d = num("Sessions last 7d", "sessions_last_7d", 0, 40, 9, as_int=True)
        sessions_last_30d = num("Sessions last 30d", "sessions_last_30d", 0, 120, 38, as_int=True)
        avg_session_minutes = num("Avg session minutes", "avg_session_minutes", 1.0, 120.0, 28.5)
        api_calls_last_30d = num("API calls last 30d", "api_calls_last_30d", 0, 20000, 1850, as_int=True)
        tokens_consumed_last_30d = num(
            "Tokens last 30d", "tokens_consumed_last_30d", 0, 2_000_000, 420000, step=1000, as_int=True
        )
    with col2:
        st.subheader("Adoption")
        models_used_count = num("Models used", "models_used_count", 0, 15, 7, as_int=True)
        tools_used_count = num("Tools used", "tools_used_count", 0, 20, 8, as_int=True)
        feature_adoption_score = num("Feature adoption", "feature_adoption_score", 0.0, 1.0, 0.78)
        nps_score = num("NPS score", "nps_score", 0.0, 10.0, 7.0, step=0.1)
        weekend_usage_ratio = num("Weekend usage ratio", "weekend_usage_ratio", 0.0, 1.0, 0.22)
        last_active_days_ago = num("Last active (days ago)", "last_active_days_ago", 0, 120, 8, as_int=True)
    with col3:
        st.subheader("Friction")
        failed_requests_rate = num(
            "Failed request rate", "failed_requests_rate", 0.0, 0.95, 0.12, step=0.01
        )
        support_tickets_last_90d = num(
            "Support tickets (90d)", "support_tickets_last_90d", 0, 20, 2, as_int=True
        )
        payment_failures_last_90d = num(
            "Payment failures (90d)", "payment_failures_last_90d", 0, 10, 1, as_int=True
        )
        spend_usd_last_30d = num("Spend USD (30d)", "spend_usd_last_30d", 0.0, 2000.0, 189.0, step=1.0)
        days_until_renewal = num("Days until renewal", "days_until_renewal", 0, 365, 21, as_int=True)
    with col4:
        st.subheader("AI-native / team")
        default_trend = float(defaults.get("engagement_trend", 0.95))
        auto_trend = sessions_last_7d / max(1.0, sessions_last_30d / 4.0)
        # Auto-derive only when the loaded record already follows the formula, so a
        # preset is scored exactly as the API / batch would score it.
        record_follows_formula = abs(round(auto_trend, 4) - default_trend) < 1e-3
        use_auto = st.checkbox(
            "Auto engagement_trend from sessions",
            value=record_follows_formula,
            key=f"{preset_id}:auto_trend",
        )
        engagement_trend = (
            float(round(auto_trend, 4))
            if use_auto
            else num("Engagement trend", "engagement_trend", 0.0, 5.0, 0.95, step=0.01)
        )
        if use_auto:
            st.caption(f"engagement_trend = {engagement_trend:.4f} (≈1 stable)")
        agent_runs_last_30d = num("Agent runs (30d)", "agent_runs_last_30d", 0, 400, 52, as_int=True)
        ide_plugin_sessions_last_30d = num(
            "IDE plugin sessions (30d)", "ide_plugin_sessions_last_30d", 0, 200, 28, as_int=True
        )
        seat_utilization = num("Seat utilization", "seat_utilization", 0.0, 1.0, 0.72, step=0.01)

    return {
        "user_id": defaults.get("user_id", "santosh_shinde"),
        "user_name": user_name,
        "days_since_signup": days_since_signup,
        "sessions_last_7d": sessions_last_7d,
        "sessions_last_30d": sessions_last_30d,
        "avg_session_minutes": avg_session_minutes,
        "models_used_count": models_used_count,
        "api_calls_last_30d": api_calls_last_30d,
        "tokens_consumed_last_30d": tokens_consumed_last_30d,
        "tools_used_count": tools_used_count,
        "failed_requests_rate": failed_requests_rate,
        "support_tickets_last_90d": support_tickets_last_90d,
        "plan_tier": plan_tier,
        "payment_failures_last_90d": payment_failures_last_90d,
        "feature_adoption_score": feature_adoption_score,
        "nps_score": nps_score,
        "last_active_days_ago": last_active_days_ago,
        "weekend_usage_ratio": weekend_usage_ratio,
        "engagement_trend": engagement_trend,
        "spend_usd_last_30d": spend_usd_last_30d,
        "days_until_renewal": days_until_renewal,
        "agent_runs_last_30d": agent_runs_last_30d,
        "ide_plugin_sessions_last_30d": ide_plugin_sessions_last_30d,
        "seat_utilization": seat_utilization,
    }


def show_hold(packet: dict) -> None:
    """Invalid input: show why, never a score (same contract as API / batch)."""
    st.error("Input failed validation, so it is not scored or queued.")
    for err in packet["validation"].get("errors") or []:
        st.markdown(f"- {err}")
    st.markdown(f"**HITL action:** `{packet['hitl']['action']}`")


def tab_predict(raw, cal, display, band, user_dict):
    st.markdown("### Plain language")
    st.write(
        "This score estimates how likely this user is to **stop using the product** "
        "(churn). A calibrated probability means: if we look at many users with ~30% "
        "score, about 30% of them actually churned in the test data."
    )
    band_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}[band]
    st.markdown(
        f"## {band_color} Churn probability: **{display:.1%}**  ({band} risk)"
    )
    st.progress(min(max(display, 0.0), 1.0))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Raw model probability", f"{raw:.1%}")
    with c2:
        if cal is not None:
            st.metric("Calibrated probability", f"{cal:.1%}")
        else:
            st.metric("Calibrated probability", "n/a")

    with st.expander("Raw feature vector"):
        st.json(user_dict)


def tab_explain(top):
    st.markdown("### Why this score?")
    st.write(
        "Positive contributions push **toward churn**; negative push **toward retain**. "
        "We use SHAP when available, otherwise a simple importance heuristic."
    )
    explain_df = pd.DataFrame(top, columns=["feature", "contribution"])
    st.dataframe(explain_df, width="stretch")
    st.bar_chart(explain_df.set_index("feature")["contribution"])


def tab_decision(packet: dict, user_dict: dict, top):
    st.markdown(f"### Decision packet — {user_dict.get('user_name') or user_dict.get('user_id')}")
    st.write(
        "End-to-end single-record path: **validate → score (raw+cal) → explain → "
        "cohort compare → HITL action**. No auto-cancel."
    )

    v = packet["validation"]
    s = packet["scoring"]
    h = packet["hitl"]
    if s is None:
        st.metric("Validation", "FAIL ❌")
        for err in v.get("errors") or []:
            st.error(err)
        st.markdown(f"**HITL action:** `{h['action']}`")
        st.warning(h.get("rationale", ""))
        st.caption(f"Auto action: `{h.get('auto_action')}` · not scored, not queued")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Validation", "OK ✅" if v.get("ok") else "FAIL ❌")
    with c2:
        st.metric("Raw P(churn)", f"{s['churn_probability_raw']:.1%}")
    with c3:
        cal = s.get("churn_probability_calibrated")
        st.metric("Calibrated", f"{cal:.1%}" if cal is not None else "n/a")
    with c4:
        st.metric("Best-F1 threshold", f"{s['best_f1_threshold']:.3f}")

    st.markdown(f"**Risk band:** `{s['risk_band']}`")
    st.markdown(f"**HITL action:** `{h['action']}`")
    st.info(h.get("rationale", ""))
    st.caption(f"Auto action: `{h.get('auto_action')}` · HITL required: always")

    st.markdown("#### Cohort percentiles vs population")
    cohort = packet.get("cohort_compare") or {}
    if cohort:
        cdf = pd.DataFrame(
            [
                {
                    "feature": k,
                    "value": v["value"],
                    "percentile": v["percentile"],
                    "pop_median": v["population_median"],
                }
                for k, v in cohort.items()
            ]
        )
        st.dataframe(cdf, width="stretch")
        st.bar_chart(cdf.set_index("feature")["percentile"])
        if any(v.get("source") == "feature_stats" for v in cohort.values()):
            st.caption(
                "users.csv not present — percentiles approximated from training "
                "quantiles in `feature_stats.json`."
            )
    else:
        st.caption("No cohort stats (missing users.csv and feature_stats.json?).")

    st.markdown("#### Top drivers")
    explain_df = pd.DataFrame(top, columns=["feature", "contribution"])
    st.dataframe(explain_df.head(5), width="stretch")

    if packet.get("outliers"):
        st.warning(f"Outliers vs train p01–p99: {len(packet['outliers'])}")
        st.json(packet["outliers"])

    st.markdown(
        "Docs: [single-record checklist](../docs/case-study/single-record-checklist.md) · "
        "[Santosh case study](../docs/case-study/santosh-case-study.md) · "
        "[use cases](../data/use_cases/README.md)"
    )
    with st.expander("Full decision packet JSON"):
        st.json(packet)


def tab_methodology(metrics: dict):
    st.markdown("### How this model was built")
    st.markdown(
        """
1. **Synthetic data** — fake AI-platform users (seed 42), no real PII.
2. **Honest baselines** — dummy, logistic regression, Random Forest, LightGBM, and CatBoost peers (XGBoost remains the teaching hero).
3. **XGBoost + Optuna** — tune trees on validation AUC.
4. **Calibration** — isotonic regression on validation probabilities.
5. **Holdout test** — ROC, PR, Brier, threshold sweep for business action.
6. **Single-record packet** — validate → score → explain → cohort → HITL.

Beginner tip: **AUC** ranks users; **Brier** checks if probabilities are honest;
**threshold** is the business dial (more alerts vs fewer misses).
"""
    )
    if metrics:
        rows = []
        for name, key in [
            ("Dummy", "dummy_test"),
            ("Logistic regression", "logreg_test"),
            ("Random Forest", "rf_test"),
            ("XGBoost default", "xgb_default_test"),
            ("XGBoost tuned", "tuned_test"),
            ("LightGBM", "lgbm_test"),
            ("CatBoost", "catboost_test"),
            ("XGBoost calibrated", "calibrated_test"),
        ]:
            block = metrics.get(key) or {}
            if block:
                rows.append(
                    {
                        "model": name,
                        "test_auc": block.get("roc_auc"),
                        "test_f1": block.get("f1"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        st.write(
            f"Calibrated test Brier: **{metrics.get('brier_calibrated_test', 'n/a')}** "
            f"(raw: {metrics.get('brier_raw_test', 'n/a')})"
        )
    st.markdown(
        "Docs: [model card](../docs/MODEL_CARD.md) · "
        "[data dictionary](../docs/data/data-dictionary.md) · "
        "[architecture](../docs/ARCHITECTURE.md) · "
        "[Santosh case](../docs/case-study/santosh-case-study.md)"
    )


def tab_benchmarks(metrics: dict):
    st.markdown("### Research artifacts & latency")
    lat = metrics.get("latency") or {}
    if lat:
        st.write(
            f"Single-record inference p50 ≈ **{lat.get('latency_ms_p50', float('nan')):.3f} ms** "
            f"(p95={lat.get('latency_ms_p95', float('nan')):.3f} ms, "
            f"n={lat.get('n_runs', '?')})."
        )
    else:
        st.info("Run `python -m retention_radar.cli.benchmark` to populate latency metrics.")

    for fname, caption in [
        ("roc_curve.png", "ROC curve"),
        ("pr_curve.png", "Precision–Recall"),
        ("calibration_curve.png", "Reliability diagram"),
        ("threshold_f1.png", "Threshold vs F1 / precision / recall"),
        ("confusion_matrix.png", "Confusion matrix @ 0.5"),
    ]:
        path = ARTIFACTS_DIR / fname
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.caption(f"Missing artifact: `{fname}` — run evaluate.")


def main() -> None:
    st.set_page_config(page_title="AI Churn Risk — Santosh", layout="wide")
    st.title("AI Platform Churn Risk")
    st.caption(
        "FOSS XGBoost demo — Santosh Shinde what-if · Decision packet · "
        "synthetic data, no real PII · HITL only (`auto_action: none`)"
    )
    st.caption(
        f"Data source: `{rr_config.CHURN_DATA_SOURCE}` · "
        f"users=`{resolve_users_csv()}` · santosh=`{resolve_santosh_json()}`"
    )

    bundle = load_model_bundle()
    if bundle is None:
        st.error(
            f"Model not found at `{MODEL_PATH}`. "
            "Run `./scripts/run_all.sh` or `python -m retention_radar.cli.train` first."
        )
        return

    preset = choose_preset()
    metrics = load_metrics()
    calibrator = load_cal()
    user_dict = collect_user_inputs(preset["record"], preset["id"])
    # One packet drives every tab: validate first, so invalid what-if input is
    # held everywhere instead of being scored on Predict / Explain.
    try:
        packet = build_decision_packet(user_dict, model_bundle=bundle, calibrator=calibrator, top_k=8)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build decision packet: {exc}")
        return

    t1, t2, t3, t4, t5 = st.tabs(
        ["Predict", "Explain", "Decision", "Methodology", "Benchmarks"]
    )
    s = packet["scoring"]
    top = [
        (d["feature"], d["contribution"]) for d in (packet.get("explanation") or {}).get("top_features", [])
    ]
    with t1:
        if s is None:
            show_hold(packet)
        else:
            tab_predict(
                s["churn_probability_raw"],
                s["churn_probability_calibrated"],
                s["churn_probability"],
                s["risk_band"],
                user_dict,
            )
    with t2:
        if s is None:
            show_hold(packet)
        else:
            tab_explain(top)
    with t3:
        tab_decision(packet, user_dict, top)
    with t4:
        tab_methodology(metrics)
    with t5:
        tab_benchmarks(metrics)


if __name__ == "__main__":
    main()
