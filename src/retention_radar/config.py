"""Project paths, seed, and the renewal-scoring contract.

Use case: a self-serve AI coding assistant (IDE extension + CLI agent) sold as
monthly Pro / Pro+ / Ultra plans. Each row is one paying subscriber scored
seven days before a renewal (T-7). The label is *voluntary* lapse at that
renewal; failed-payment (involuntary) churn and subscribers who already
scheduled a cancel are routed to other tracks, not to this model.

All paths are relative to PROJECT_ROOT so scripts work no matter
where you launch them from (as long as ``src`` is on PYTHONPATH, or the
package is installed editable).

Canonical module: ``retention_radar.config``.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"  # lakehouse gold exports
INTERIM_DIR = DATA_DIR / "interim"  # CDS parity (empty in teaching path)
PROCESSED_DIR = DATA_DIR / "processed"  # CDS parity (features usually in-memory)
# Committed seed-42 serve bundle (never overwrite via lakehouse E2E).
SEED_MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"
RESEARCH_DIR = DOCS_DIR  # compat alias (teaching docs live under docs/)
ARTICLES_DIR = PROJECT_ROOT / "articles"  # unused in this teaching repo
SCHEMAS_DIR = PROJECT_ROOT / "configs" / "schemas"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"

# Key files (data paths are not redirected by artifact override)
# One row per subscriber at T-7 before renewal (the model's training table).
USERS_CSV = RAW_DIR / "renewals_t7.csv"
# Every renewal in the cohort, with outcome + routing columns (label audit trail).
RENEWALS_ALL_CSV = RAW_DIR / "renewals_all.csv"
# Scoring-time records for the two worked examples (no label: the renewal is ahead).
HERO_DIR = RAW_DIR / "subscribers"
LAKEHOUSE_FEATURES_CSV = EXTERNAL_DIR / "churn_user_features.csv"
LAKEHOUSE_HERO_JSON = EXTERNAL_DIR / "hero_inference_record.json"
# auto | synthetic | lakehouse — auto prefers external lakehouse exports when present
CHURN_DATA_SOURCE = os.environ.get("CHURN_DATA_SOURCE", "auto")
USER_RECORD_SCHEMA_PATH = SCHEMAS_DIR / "user_record.schema.json"
GUIDES_DIR = DOCS_DIR  # docs root; nested guides/data/case-study below


def _resolve_artifact_root(raw: str | None = None) -> Path | None:
    """Return absolute artifact root from env/arg, or None for default seed paths."""
    value = (raw if raw is not None else os.environ.get("RETENTION_RADAR_ARTIFACT_DIR", "")).strip()
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def apply_artifact_dir(artifact_dir: str | Path | None = None) -> Path | None:
    """Rebind model / docs / runtime artifact paths.

    When ``RETENTION_RADAR_ARTIFACT_DIR`` (or ``artifact_dir``) is set, train /
    evaluate / docs_gen write under that directory instead of committed
    ``models/`` and published ``docs/model-card.md`` / data-dictionary.
    Pass ``None`` with env unset (or empty string) to restore seed defaults.

    Returns the resolved artifact root, or ``None`` when using seed paths.
    """
    global MODELS_DIR, ARTIFACTS_DIR, MODEL_PATH, CALIBRATOR_PATH, METRICS_PATH
    global FEATURE_NAMES_PATH, FEATURE_STATS_PATH, MODEL_CARD_PATH, DATA_DICTIONARY_PATH

    if artifact_dir is None:
        root = _resolve_artifact_root()
    elif str(artifact_dir).strip() == "":
        root = None
    else:
        root = _resolve_artifact_root(str(artifact_dir))

    if root is None:
        MODELS_DIR = SEED_MODELS_DIR
        ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
        MODEL_CARD_PATH = DOCS_DIR / "model-card.md"
        DATA_DICTIONARY_PATH = DOCS_DIR / "data" / "data-dictionary.md"
    else:
        root.mkdir(parents=True, exist_ok=True)
        MODELS_DIR = root
        ARTIFACTS_DIR = root
        MODEL_CARD_PATH = root / "model-card.md"
        DATA_DICTIONARY_PATH = root / "data-dictionary.md"

    MODEL_PATH = MODELS_DIR / "churn_xgb.joblib"
    CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"
    METRICS_PATH = MODELS_DIR / "metrics.json"
    FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
    FEATURE_STATS_PATH = MODELS_DIR / "feature_stats.json"
    return root


# Initialize from env at import time (lakehouse E2E exports the override before Python).
MODELS_DIR = SEED_MODELS_DIR
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = MODELS_DIR / "churn_xgb.joblib"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
FEATURE_STATS_PATH = MODELS_DIR / "feature_stats.json"


def runtime_log_dir() -> Path:
    """Where the API and action-log tools append runtime logs (prediction log, action log).

    ``RETENTION_RADAR_LOG_DIR`` overrides it without moving the model bundle
    (``RETENTION_RADAR_ARTIFACT_DIR`` moves both); default is ``ARTIFACTS_DIR``.
    """
    override = os.environ.get("RETENTION_RADAR_LOG_DIR", "").strip()
    return Path(override).expanduser().resolve() if override else ARTIFACTS_DIR
MODEL_CARD_PATH = DOCS_DIR / "model-card.md"
DATA_DICTIONARY_PATH = DOCS_DIR / "data" / "data-dictionary.md"
apply_artifact_dir()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Plans (fictional product, priced like the 2025-26 AI coding assistant market)
# ---------------------------------------------------------------------------
PLAN_TIER_ORDER = ["pro", "pro_plus", "ultra"]
PLAN_TIER_MAP = {name: idx for idx, name in enumerate(PLAN_TIER_ORDER)}
PLAN_PRICE_USD = {"pro": 20.0, "pro_plus": 60.0, "ultra": 200.0}

# ---------------------------------------------------------------------------
# Schema: 22 serve fields (+ user_id, user_name) known at T-7
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "plan_tier",  # encoded to plan_tier_code
    "renewals_completed",
    "active_days_7d",
    "active_days_28d",
    "engagement_trend",
    "last_active_days_ago",
    "agent_requests_28d",
    "allowance_used_pct",
    "limit_hits_14d",
    "cheap_model_share_28d",
    "overage_usd_28d",
    "overage_toggled_off",
    "suggestion_accept_rate_28d",
    "accept_rate_change",
    "agent_task_success_rate",
    "failed_requests_rate",
    "incident_exposed_28d",
    "support_tickets_90d",
    "ide_sessions_28d",
    "cli_sessions_28d",
    "weekend_usage_ratio",
    "first_renewal_after_pricing_change",
]

MODEL_FEATURE_COLUMNS = ["plan_tier_code"] + FEATURE_COLUMNS[1:]

FEATURE_DESCRIPTIONS = {
    "user_id": "Stable synthetic subscriber id (not a feature).",
    "user_name": "Display name for the worked examples (not a feature).",
    "plan_tier": "Monthly plan: pro ($20) | pro_plus ($60) | ultra ($200).",
    "plan_tier_code": "Ordinal encoding of plan_tier (pro=0, pro_plus=1, ultra=2).",
    "renewals_completed": "Monthly renewals already paid. 0 = this is the first renewal (the cliff).",
    "active_days_7d": "Days with any coding activity in the 7 days before T-7 (0-7).",
    "active_days_28d": "Days with any coding activity in the 28 days before T-7 (0-28).",
    "engagement_trend": "active_days_7d / max(1, active_days_28d / 4): ~1 steady, <1 fading.",
    "last_active_days_ago": "Days since the last coding activity, measured at T-7.",
    "agent_requests_28d": "Agent / chat requests sent to frontier models in 28 days.",
    "allowance_used_pct": "Share of the plan's included usage consumed in 28 days (can exceed 1 with overage).",
    "limit_hits_14d": "Times a 5-hour or weekly usage cap blocked a request in 14 days.",
    "cheap_model_share_28d": "Share of requests routed to a cheaper / auto model (rationing signal).",
    "overage_usd_28d": "Usage billed above the plan in 28 days (USD).",
    "overage_toggled_off": "1 if paid overage was switched off or capped after being on.",
    "suggestion_accept_rate_28d": "Accepted / shown inline suggestions in 28 days.",
    "accept_rate_change": "Accept rate this 28 days / previous 28 days (1 = unchanged).",
    "agent_task_success_rate": "Agent tasks that ended with the change kept (not reverted) in 28 days.",
    "failed_requests_rate": "Share of requests that errored or timed out in 28 days.",
    "incident_exposed_28d": "1 if the subscriber had requests during a declared incident window.",
    "support_tickets_90d": "Support tickets opened in 90 days.",
    "ide_sessions_28d": "IDE-extension sessions in 28 days.",
    "cli_sessions_28d": "CLI-agent sessions in 28 days.",
    "weekend_usage_ratio": "Share of activity on weekends (side-project signal).",
    "first_renewal_after_pricing_change": "1 if this is the subscriber's first renewal since the last limit / pricing change.",
    "churned": "Label: 1 = voluntarily let the plan lapse at this renewal (training only).",
}

INFERENCE_REQUIRED_KEYS = ["user_id", "user_name"] + FEATURE_COLUMNS

BINARY_FEATURES = [
    "overage_toggled_off",
    "incident_exposed_28d",
    "first_renewal_after_pricing_change",
]
RATE_FEATURES = [
    "cheap_model_share_28d",
    "suggestion_accept_rate_28d",
    "agent_task_success_rate",
    "failed_requests_rate",
    "weekend_usage_ratio",
]

FEATURE_RANGES = {
    "renewals_completed": (0, 60),
    "active_days_7d": (0, 7),
    "active_days_28d": (0, 28),
    "engagement_trend": (0.0, 4.0),
    "last_active_days_ago": (0, 90),
    "agent_requests_28d": (0, 50000),
    "allowance_used_pct": (0.0, 3.0),
    "limit_hits_14d": (0, 60),
    "cheap_model_share_28d": (0.0, 1.0),
    "overage_usd_28d": (0.0, 5000.0),
    "overage_toggled_off": (0, 1),
    "suggestion_accept_rate_28d": (0.0, 1.0),
    "accept_rate_change": (0.0, 3.0),
    "agent_task_success_rate": (0.0, 1.0),
    "failed_requests_rate": (0.0, 1.0),
    "incident_exposed_28d": (0, 1),
    "support_tickets_90d": (0, 50),
    "ide_sessions_28d": (0, 500),
    "cli_sessions_28d": (0, 500),
    "weekend_usage_ratio": (0.0, 1.0),
    "first_renewal_after_pricing_change": (0, 1),
}

COHORT_COMPARE_FEATURES = [
    "active_days_28d",
    "engagement_trend",
    "limit_hits_14d",
    "cheap_model_share_28d",
    "allowance_used_pct",
    "suggestion_accept_rate_28d",
    "agent_task_success_rate",
    "renewals_completed",
]

ID_COLUMNS = ["user_id", "user_name"]
TARGET_COLUMN = "churned"

# Renewal outcomes in renewals_all.csv. Only rows routed to "model" train/score.
OUTCOME_RENEWED = "renewed"
OUTCOME_VOLUNTARY = "voluntary_lapse"
OUTCOME_INVOLUNTARY = "involuntary_lapse"
ROUTE_MODEL = "model"
ROUTE_DUNNING = "dunning"  # card failed and retries ran out: payments problem
ROUTE_CANCEL_FLOW = "cancel_flow"  # cancel already scheduled before T-7

# ---------------------------------------------------------------------------
# Worked examples (scoring-time records, never in the training table)
# ---------------------------------------------------------------------------
HEROES = {
    "maya": "maya.json",  # Pro, first renewal since the weekly-cap cut, rationing
    "arjun": "arjun.json",  # steady Pro+ user; the one we deliberately leave alone
}
DEFAULT_HERO = "maya"

# ---------------------------------------------------------------------------
# Decision policy. Every number here is an ASSUMPTION until a holdout measures it.
# ---------------------------------------------------------------------------
# Risk bands on calibrated p (base rate is ~9%, so "high" starts well below 0.5).
RISK_BAND_EDGES = (0.10, 0.30)
# Share of eligible subscribers kept out of every playbook to measure lift.
HOLDOUT_PCT = 10
# Months a saved subscriber stays on average after a save (Churnkey reports ~5).
MONTHS_RETAINED_AFTER_SAVE = 5.0
# Playbooks a retention lead has approved. effect = share of would-be churners the
# playbook keeps; cost_usd = paid per send; discount_usd = paid per acceptance,
# including by subscribers who would have renewed anyway (sure_thing_accept).
PLAYBOOKS = {
    "in_app_usage_tips": {
        "label": "In-app message: how to stretch the allowance",
        "effect": 0.04,
        "cost_usd": 0.02,
        "discount_usd": 0.0,
        "sure_thing_accept": 0.0,
    },
    "limit_reset": {
        "label": "One-time usage-limit reset with a short note",
        "effect": 0.25,
        "cost_usd": 2.00,  # compute for one reset of the 5-hour window on Pro
        "discount_usd": 0.0,
        "sure_thing_accept": 0.0,
    },
    "pause_offer": {
        "label": "Offer a one-month pause instead of cancel",
        "effect": 0.15,
        "cost_usd": 0.02,
        "discount_usd": 0.0,
        "sure_thing_accept": 0.02,
        "pause_months": 1.0,
    },
    "cancel_flow_discount": {
        "label": "Arm a 3-month 20%-off offer in the cancel flow",
        "effect": 0.12,
        "cost_usd": 0.0,
        "discount_usd_pct_of_price": 0.60,  # 20% x 3 months = 0.6 of one month
        "sure_thing_accept": 0.03,
    },
    # The only playbook with a person in it: worth ~15 minutes of staff time only
    # when the plan is big enough to pay for it.
    "personal_email": {
        "label": "Personal email from the team, written by a person",
        "effect": 0.25,
        "cost_usd": 15.0,
        "discount_usd": 0.0,
        "sure_thing_accept": 0.0,
        "plans": ["ultra"],
    },
}

# Train / val / test fractions (of the model table)
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# Cohort size (renewals generated before routing) / Optuna trials. Env overrides for CI.
N_USERS = int(os.environ.get("N_USERS", "8000"))
N_OPTUNA_TRIALS = int(os.environ.get("N_OPTUNA_TRIALS", "20"))

# Platt (sigmoid), not isotonic: with 142 lapses in validation, isotonic fits
# a few dozen flat steps, ties most of the queue and hands some subscribers P = 0.000.
CALIBRATION_METHOD = "sigmoid"

LATENCY_WARMUP = 20
LATENCY_RUNS = 200
