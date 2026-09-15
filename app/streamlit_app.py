"""Streamlit demo: Predict / Explain / Decision / Methodology / Benchmarks.

Run from project root:
    export PYTHONPATH="$(pwd)"
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.calibrate import load_calibrator  # noqa: E402
from src.config import (  # noqa: E402
    ARTIFACTS_DIR,
    CALIBRATOR_PATH,
    GUIDES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    PLAN_TIER_ORDER,
)
from src.infer import predict_user  # noqa: E402
from src.single_record import build_decision_packet  # noqa: E402
from src.retention_radar.data.ingest import (  # noqa: E402
    resolve_santosh_json,
    resolve_users_csv,
)
from src.retention_radar.serving.policy import risk_band  # noqa: E402
from src.retention_radar import config as rr_config  # noqa: E402


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


def collect_user_inputs(defaults: dict) -> dict:
    st.sidebar.header("Santosh what-if")
    st.sidebar.caption("Tweak sliders to see how risk changes.")
    user_name = st.sidebar.text_input(
        "Name", value=defaults.get("user_name", "Santosh Shinde")
    )
    plan_tier = st.sidebar.selectbox(
        "Plan tier",
        PLAN_TIER_ORDER,
        index=PLAN_TIER_ORDER.index(defaults.get("plan_tier", "pro")),
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Usage")
        days_since_signup = st.slider(
            "Days since signup", 14, 900, int(defaults.get("days_since_signup", 420))
        )
        sessions_last_7d = st.slider(
            "Sessions last 7d", 0, 40, int(defaults.get("sessions_last_7d", 9))
        )
        sessions_last_30d = st.slider(
            "Sessions last 30d", 0, 120, int(defaults.get("sessions_last_30d", 38))
        )
        avg_session_minutes = st.slider(
            "Avg session minutes",
            1.0,
            120.0,
            float(defaults.get("avg_session_minutes", 28.5)),
        )
        api_calls_last_30d = st.slider(
            "API calls last 30d", 0, 20000, int(defaults.get("api_calls_last_30d", 1850))
        )
        tokens_consumed_last_30d = st.slider(
            "Tokens last 30d",
            0,
            2_000_000,
            int(defaults.get("tokens_consumed_last_30d", 420000)),
            step=1000,
        )
    with col2:
        st.subheader("Adoption")
        models_used_count = st.slider(
            "Models used", 0, 15, int(defaults.get("models_used_count", 7))
        )
        tools_used_count = st.slider(
            "Tools used", 0, 20, int(defaults.get("tools_used_count", 8))
        )
        feature_adoption_score = st.slider(
            "Feature adoption",
            0.0,
            1.0,
            float(defaults.get("feature_adoption_score", 0.78)),
        )
        nps_score = st.slider(
            "NPS score", 0.0, 10.0, float(defaults.get("nps_score", 7.0)), step=0.5
        )
        weekend_usage_ratio = st.slider(
            "Weekend usage ratio",
            0.0,
            1.0,
            float(defaults.get("weekend_usage_ratio", 0.22)),
        )
        last_active_days_ago = st.slider(
            "Last active (days ago)",
            0,
            120,
            int(defaults.get("last_active_days_ago", 8)),
        )
    with col3:
        st.subheader("Friction")
        failed_requests_rate = st.slider(
            "Failed request rate",
            0.0,
            0.95,
            float(defaults.get("failed_requests_rate", 0.12)),
            step=0.01,
        )
        support_tickets_last_90d = st.slider(
            "Support tickets (90d)",
            0,
            20,
            int(defaults.get("support_tickets_last_90d", 2)),
        )
        payment_failures_last_90d = st.slider(
            "Payment failures (90d)",
            0,
            10,
            int(defaults.get("payment_failures_last_90d", 1)),
        )
        spend_usd_last_30d = st.slider(
            "Spend USD (30d)",
            0.0,
            2000.0,
            float(defaults.get("spend_usd_last_30d", 189.0)),
            step=1.0,
        )
        days_until_renewal = st.slider(
            "Days until renewal",
            0,
            365,
            int(defaults.get("days_until_renewal", 21)),
        )
    with col4:
        st.subheader("AI-native / team")
        default_trend = float(defaults.get("engagement_trend", 0.95))
        auto_trend = sessions_last_7d / max(1.0, sessions_last_30d / 4.0)
        use_auto = st.checkbox("Auto engagement_trend from sessions", value=True)
        engagement_trend = (
            float(round(auto_trend, 4))
            if use_auto
            else st.slider(
                "Engagement trend",
                0.0,
                3.0,
                default_trend,
                step=0.01,
            )
        )
        if use_auto:
            st.caption(f"engagement_trend = {engagement_trend:.4f} (≈1 stable)")
        agent_runs_last_30d = st.slider(
            "Agent runs (30d)",
            0,
            400,
            int(defaults.get("agent_runs_last_30d", 52)),
        )
        ide_plugin_sessions_last_30d = st.slider(
            "IDE plugin sessions (30d)",
            0,
            200,
            int(defaults.get("ide_plugin_sessions_last_30d", 28)),
        )
        seat_utilization = st.slider(
            "Seat utilization",
            0.0,
            1.0,
            float(defaults.get("seat_utilization", 0.72)),
            step=0.01,
        )

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


def score_user(bundle, calibrator, user_dict):
    """Thin adapter: scoring + HITL bands live in serving, not in the UI."""
    result = predict_user(user_dict, bundle, calibrator=calibrator, top_k=8)
    return (
        result["churn_probability_raw"],
        result["churn_probability_calibrated"],
        result["churn_probability"],
        result["top_features"],
    )


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
    st.dataframe(explain_df, use_container_width=True)
    st.bar_chart(explain_df.set_index("feature")["contribution"])


def tab_decision(bundle, calibrator, user_dict, top):
    st.markdown("### Santosh case — Decision packet")
    st.write(
        "End-to-end single-record path: **validate → score (raw+cal) → explain → "
        "cohort compare → HITL action**. No auto-cancel."
    )
    try:
        packet = build_decision_packet(
            user_dict, model_bundle=bundle, calibrator=calibrator
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build decision packet: {exc}")
        return

    v = packet["validation"]
    s = packet["scoring"]
    h = packet["hitl"]

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
        st.dataframe(cdf, use_container_width=True)
        st.bar_chart(cdf.set_index("feature")["percentile"])
    else:
        st.caption("No cohort stats (missing users.csv?).")

    st.markdown("#### Top drivers")
    explain_df = pd.DataFrame(top, columns=["feature", "contribution"])
    st.dataframe(explain_df.head(5), use_container_width=True)

    if packet.get("outliers"):
        st.warning(f"Outliers vs train p01–p99: {len(packet['outliers'])}")
        st.json(packet["outliers"])

    checklist = GUIDES_DIR / "single-record-checklist.md"
    case_study = GUIDES_DIR / "santosh-case-study.md"
    st.markdown(
        f"Docs: [single-record checklist]({checklist.as_posix()}) · "
        f"[Santosh case study]({case_study.as_posix()})"
    )
    with st.expander("Full decision packet JSON"):
        st.json(packet)


def tab_methodology(metrics: dict):
    st.markdown("### How this model was built")
    st.markdown(
        """
1. **Synthetic data** — fake AI-platform users (seed 42), no real PII.
2. **Honest baselines** — dummy, logistic regression, Random Forest, and LightGBM peers (XGBoost remains the teaching hero).
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
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.write(
            f"Calibrated test Brier: **{metrics.get('brier_calibrated_test', 'n/a')}** "
            f"(raw: {metrics.get('brier_raw_test', 'n/a')})"
        )
    st.markdown(
        "Docs: [model card](../MODEL_CARD.md) · "
        "[data dictionary](../docs/data-dictionary.md) · "
        "[architecture](../ARCHITECTURE.md) · "
        "[Santosh case](../docs/santosh-case-study.md)"
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
        st.info("Run `python -m src.benchmark` to populate latency metrics.")

    for fname, caption in [
        ("roc_curve.png", "ROC curve"),
        ("pr_curve.png", "Precision–Recall"),
        ("calibration_curve.png", "Reliability diagram"),
        ("threshold_f1.png", "Threshold vs F1 / precision / recall"),
        ("confusion_matrix.png", "Confusion matrix @ 0.5"),
    ]:
        path = ARTIFACTS_DIR / fname
        if path.exists():
            st.image(str(path), caption=caption, use_container_width=True)
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
            "Run `./scripts/run_all.sh` or `python -m src.train` first."
        )
        return

    defaults = load_santosh_defaults()
    metrics = load_metrics()
    calibrator = load_cal()
    user_dict = collect_user_inputs(defaults)
    raw, cal, display, top = score_user(bundle, calibrator, user_dict)
    band = risk_band(display)

    t1, t2, t3, t4, t5 = st.tabs(
        ["Predict", "Explain", "Decision", "Methodology", "Benchmarks"]
    )
    with t1:
        tab_predict(raw, cal, display, band, user_dict)
    with t2:
        tab_explain(top)
    with t3:
        tab_decision(bundle, calibrator, user_dict, top)
    with t4:
        tab_methodology(metrics)
    with t5:
        tab_benchmarks(metrics)


if __name__ == "__main__":
    main()
