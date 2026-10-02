"""Streamlit demo: Predict / Explain / Decision / Methodology / Benchmarks.

One subscriber of a monthly AI coding assistant plan, scored seven days before
renewal. Pick a worked example or a use-case scenario, move the sliders, and
watch the calibrated score, the band and the suggested action change.

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
    PLAN_PRICE_USD,
    PLAN_TIER_ORDER,
)
from retention_radar.data.generate import HERO_PROFILES  # noqa: E402
from retention_radar.data.ingest import resolve_hero_json, resolve_users_csv  # noqa: E402
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


def load_hero(name: str) -> dict:
    path = resolve_hero_json(name)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return HERO_PROFILES[name]()


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
    """Sidebar picker: the worked examples first, then the use-case pack."""
    presets = []
    for name in rr_config.HEROES:
        record = load_hero(name)
        presets.append(
            {
                "id": f"hero_{name}",
                "title": record.get("user_name", name),
                "story": "Worked example scored at T-7 (no label: the renewal is ahead).",
                "expected": None,
                "record": record,
            }
        )
    hero_records = [p["record"] for p in presets]
    for p in load_use_cases():
        if p["record"] in hero_records:
            presets[hero_records.index(p["record"])]["expected"] = p.get("expected")
            presets[hero_records.index(p["record"])]["story"] = p.get("story", "")
            continue
        presets.append(p)
    titles = [p["title"] for p in presets]
    title = st.sidebar.selectbox("Subscriber", titles, index=0, key="use_case")
    preset = presets[titles.index(title)]
    st.sidebar.caption(preset.get("story", ""))
    exp = preset.get("expected")
    if exp:
        st.sidebar.caption(
            f"Committed bundle: `{exp['band']}` → `{exp['action']}` "
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


def collect_user_inputs(defaults: dict, preset_id: str = "hero_maya") -> dict:
    st.sidebar.header("What-if")
    st.sidebar.caption("Pick a subscriber, then move sliders to see how risk and action change.")
    plan_tier = st.sidebar.selectbox(
        "Plan",
        PLAN_TIER_ORDER,
        index=PLAN_TIER_ORDER.index(defaults.get("plan_tier", "pro")),
        format_func=lambda t: f"{t} (${PLAN_PRICE_USD[t]:.0f}/mo)",
        key=f"{preset_id}:plan_tier",
    )

    def num(label, key, lo, hi, fallback, step=None, as_int=False):
        return _slider(preset_id, label, key, lo, hi, defaults, fallback, step=step, as_int=as_int)

    def flag(label, key):
        return int(st.checkbox(label, value=bool(defaults.get(key, 0)), key=f"{preset_id}:{key}"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader("Habit")
        renewals_completed = num("Renewals already paid", "renewals_completed", 0, 36, 3, as_int=True)
        active_days_28d = num("Active days (28d)", "active_days_28d", 0, 28, 15, as_int=True)
        active_days_7d = num("Active days (7d)", "active_days_7d", 0, 7, 2, as_int=True)
        active_days_7d = min(active_days_7d, active_days_28d)
        last_active_days_ago = num("Last active (days ago)", "last_active_days_ago", 0, 60, 2, as_int=True)
        weekend_usage_ratio = num("Weekend share", "weekend_usage_ratio", 0.0, 1.0, 0.2, step=0.01)
    with col2:
        st.subheader("Against the cap")
        allowance_used_pct = num("Allowance used", "allowance_used_pct", 0.0, 3.0, 1.0, step=0.01)
        limit_hits_14d = num("Cap hits (14d)", "limit_hits_14d", 0, 20, 3, as_int=True)
        cheap_model_share_28d = num("Cheap-model share", "cheap_model_share_28d", 0.0, 1.0, 0.6, step=0.01)
        agent_requests_28d = num("Agent requests (28d)", "agent_requests_28d", 0, 5000, 470, as_int=True)
        overage_usd_28d = num("Overage USD (28d)", "overage_usd_28d", 0.0, 500.0, 0.0, step=1.0)
        overage_toggled_off = flag("Switched overage off", "overage_toggled_off")
    with col3:
        st.subheader("Quality")
        suggestion_accept_rate_28d = num(
            "Suggestion accept rate", "suggestion_accept_rate_28d", 0.0, 0.9, 0.29, step=0.01
        )
        accept_rate_change = num("Accept rate vs last period", "accept_rate_change", 0.3, 2.0, 1.0, step=0.01)
        agent_task_success_rate = num(
            "Agent tasks kept", "agent_task_success_rate", 0.0, 1.0, 0.6, step=0.01
        )
        failed_requests_rate = num("Failed requests", "failed_requests_rate", 0.0, 0.5, 0.05, step=0.01)
        incident_exposed_28d = flag("Hit an incident window", "incident_exposed_28d")
        support_tickets_90d = num("Support tickets (90d)", "support_tickets_90d", 0, 10, 0, as_int=True)
    with col4:
        st.subheader("Surface / pricing")
        ide_sessions_28d = num("IDE sessions (28d)", "ide_sessions_28d", 0, 120, 14, as_int=True)
        cli_sessions_28d = num("CLI sessions (28d)", "cli_sessions_28d", 0, 120, 19, as_int=True)
        first_renewal_after_pricing_change = flag(
            "First renewal since the cap change", "first_renewal_after_pricing_change"
        )
        engagement_trend = round(active_days_7d / max(1.0, active_days_28d / 4.0), 4)
        st.caption(f"engagement_trend = {engagement_trend:.4f} (~1 steady, <1 fading)")

    return {
        "user_id": defaults.get("user_id", "sub_maya"),
        "user_name": defaults.get("user_name", "Maya (worked example)"),
        "plan_tier": plan_tier,
        "renewals_completed": renewals_completed,
        "active_days_7d": active_days_7d,
        "active_days_28d": active_days_28d,
        "engagement_trend": engagement_trend,
        "last_active_days_ago": last_active_days_ago,
        "agent_requests_28d": agent_requests_28d,
        "allowance_used_pct": allowance_used_pct,
        "limit_hits_14d": limit_hits_14d,
        "cheap_model_share_28d": cheap_model_share_28d,
        "overage_usd_28d": overage_usd_28d,
        "overage_toggled_off": overage_toggled_off,
        "suggestion_accept_rate_28d": suggestion_accept_rate_28d,
        "accept_rate_change": accept_rate_change,
        "agent_task_success_rate": agent_task_success_rate,
        "failed_requests_rate": failed_requests_rate,
        "incident_exposed_28d": incident_exposed_28d,
        "support_tickets_90d": support_tickets_90d,
        "ide_sessions_28d": ide_sessions_28d,
        "cli_sessions_28d": cli_sessions_28d,
        "weekend_usage_ratio": weekend_usage_ratio,
        "first_renewal_after_pricing_change": first_renewal_after_pricing_change,
    }


def show_hold(packet: dict) -> None:
    """Invalid input: show why, never a score (same contract as API / batch)."""
    st.error("Input failed validation, so it is not scored or queued.")
    for err in packet["validation"].get("errors") or []:
        st.markdown(f"- {err}")
    st.markdown(f"**Action:** `{packet['decision']['action']}`")


def tab_predict(raw, cal, display, band, user_dict):
    st.markdown("### What the number means")
    st.write(
        "The chance this subscriber lets the plan lapse at the renewal seven days from now. "
        "Calibrated means: among many subscribers scored around 15%, about 15% lapsed in the "
        "test data. The base rate is about 10%."
    )
    band_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}[band]
    st.markdown(f"## {band_color} P(lapse at renewal): **{display:.1%}**  ({band})")
    st.progress(min(max(display, 0.0), 1.0))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Raw model probability", f"{raw:.1%}")
    with c2:
        st.metric("Calibrated probability", f"{cal:.1%}" if cal is not None else "n/a")
    st.caption(
        "Raw is inflated on purpose: training reweights lapses (scale_pos_weight). "
        "Only the calibrated number is a probability."
    )
    with st.expander("Feature vector"):
        st.json(user_dict)


def tab_explain(top):
    st.markdown("### Why this score?")
    st.write(
        "SHAP contributions in log-odds: positive pushes toward lapse, negative toward renewal. "
        "They explain the score, not the cause, and not what an intervention would do."
    )
    explain_df = pd.DataFrame(top, columns=["feature", "contribution"])
    st.dataframe(explain_df, width="stretch")
    st.bar_chart(explain_df.set_index("feature")["contribution"])


def tab_decision(packet: dict, user_dict: dict, top):
    st.markdown(f"### Decision packet: {user_dict.get('user_name') or user_dict.get('user_id')}")
    st.write(
        "validate → score (raw + calibrated) → explain → compare to the cohort → pick an "
        "approved playbook, the holdout, or nothing. The service never sends anything."
    )

    v = packet["validation"]
    s = packet["scoring"]
    d = packet["decision"]
    if s is None:
        st.metric("Validation", "FAIL ❌")
        for err in v.get("errors") or []:
            st.error(err)
        st.markdown(f"**Action:** `{d['action']}`")
        st.warning(d.get("rationale", ""))
        st.caption(f"Auto action: `{d.get('auto_action')}` · not scored, not queued")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Validation", "OK ✅" if v.get("ok") else "FAIL ❌")
    with c2:
        st.metric("Raw P(lapse)", f"{s['churn_probability_raw']:.1%}")
    with c3:
        cal = s.get("churn_probability_calibrated")
        st.metric("Calibrated", f"{cal:.1%}" if cal is not None else "n/a")
    with c4:
        st.metric("τ (contact threshold)", f"{s['best_f1_threshold']:.3f}")

    st.markdown(f"**Risk band:** `{s['risk_band']}`")
    st.markdown(f"**Action:** `{d['action']}`")
    st.info(d.get("rationale", ""))
    if d.get("candidates"):
        st.markdown("#### Playbooks considered (expected value, assumption-based)")
        st.dataframe(pd.DataFrame(d["candidates"]), width="stretch")
    who = "a person" if d.get("hitl_required") else "the lifecycle tool, after a human approved the playbook"
    st.caption(f"Auto action: `{d.get('auto_action')}` · executed by {who} · holdout: {d.get('holdout')}")

    st.markdown("#### Cohort percentiles (T-7 renewal table)")
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
                "renewals_t7.csv not present: percentiles approximated from training "
                "quantiles in `feature_stats.json`."
            )
    else:
        st.caption("No cohort stats (missing renewals_t7.csv and feature_stats.json?).")

    st.markdown("#### Top drivers")
    st.dataframe(pd.DataFrame(top, columns=["feature", "contribution"]).head(5), width="stretch")

    if packet.get("outliers"):
        st.warning(f"Outside training p01–p99: {len(packet['outliers'])}")
        st.json(packet["outliers"])

    st.markdown(
        "Docs: [worked examples](../docs/case-study/renewal-worked-examples.md) · "
        "[single-record checklist](../docs/case-study/single-record-checklist.md) · "
        "[use cases](../data/use_cases/README.md)"
    )
    with st.expander("Full decision packet JSON"):
        st.json(packet)


def tab_methodology(metrics: dict):
    st.markdown("### How this model was built")
    st.markdown(
        """
1. **Label from billing, not activity**: voluntary lapse at renewal. Failed-card lapses go to
   dunning; subscribers who already scheduled a cancel go to the cancel flow. Neither trains the model.
2. **Snapshot at T-7**: every feature is computed as of seven days before the renewal.
3. **Honest ladder**: dummy, logistic regression, Random Forest, XGBoost (default + Optuna),
   LightGBM, CatBoost. Logistic regression is allowed to win, and on this run it does.
4. **Calibration**: Platt (sigmoid) on validation. Isotonic was tried and rejected: with ~140
   validation lapses it made a staircase and scored some subscribers at exactly 0%.
5. **Policy**: contact only above τ; pick the approved playbook with the best expected value;
   hold 10% out so lift can be measured.
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
                        "test_pr_auc": block.get("average_precision"),
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch")
        st.write(
            f"Calibrated test Brier: **{metrics.get('brier_calibrated_test', 'n/a')}** "
            f"(raw: {metrics.get('brier_raw_test', 'n/a')})"
        )
    st.markdown(
        "Docs: [model card](../docs/model-card.md) · "
        "[data dictionary](../docs/data/data-dictionary.md) · "
        "[architecture](../docs/architecture.md)"
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
        if not path.exists():
            path = ROOT / "results" / "plots" / fname
        if path.exists():
            st.image(str(path), caption=caption, width="stretch")
        else:
            st.caption(f"Missing artifact: `{fname}`: run evaluate.")


def main() -> None:
    st.set_page_config(page_title="Renewal risk: AI coding assistant", layout="wide")
    st.title("Renewal risk at T-7")
    st.caption(
        "Monthly AI coding assistant plan (Pro $20 · Pro+ $60 · Ultra $200). "
        "Synthetic data, no real PII. The service suggests; it never sends (`auto_action: none`)."
    )
    st.caption(
        f"Data source: `{rr_config.CHURN_DATA_SOURCE}` · renewals=`{resolve_users_csv()}`"
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
    try:
        packet = build_decision_packet(user_dict, model_bundle=bundle, calibrator=calibrator, top_k=8)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build decision packet: {exc}")
        return

    t1, t2, t3, t4, t5 = st.tabs(["Predict", "Explain", "Decision", "Methodology", "Benchmarks"])
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
