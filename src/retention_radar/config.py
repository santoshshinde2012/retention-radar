"""Project paths, seed, and feature column definitions.

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
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DOCS_DIR = PROJECT_ROOT / "docs"
RESEARCH_DIR = DOCS_DIR  # compat alias (teaching docs live under docs/)
ARTICLES_DIR = PROJECT_ROOT / "articles"  # unused in this teaching repo
SCHEMAS_DIR = PROJECT_ROOT / "configs" / "schemas"

# Key files
USERS_CSV = RAW_DIR / "users.csv"
SANTOSH_JSON = RAW_DIR / "santosh_shinde.json"
LAKEHOUSE_FEATURES_CSV = EXTERNAL_DIR / "churn_user_features.csv"
LAKEHOUSE_SANTOSH_JSON = EXTERNAL_DIR / "santosh_inference_record.json"
# auto | synthetic | lakehouse — auto prefers external lakehouse exports when present
CHURN_DATA_SOURCE = os.environ.get("CHURN_DATA_SOURCE", "auto")
MODEL_PATH = MODELS_DIR / "churn_xgb.joblib"
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.json"
FEATURE_STATS_PATH = MODELS_DIR / "feature_stats.json"
USER_RECORD_SCHEMA_PATH = SCHEMAS_DIR / "user_record.schema.json"
GUIDES_DIR = DOCS_DIR  # docs root; nested guides/data/case-study below
MODEL_CARD_PATH = DOCS_DIR / "MODEL_CARD.md"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
DATA_DICTIONARY_PATH = DOCS_DIR / "data" / "data-dictionary.md"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# Columns used as model features (after encoding plan_tier → plan_tier_code)
FEATURE_COLUMNS = [
    "days_since_signup",
    "sessions_last_7d",
    "sessions_last_30d",
    "avg_session_minutes",
    "models_used_count",
    "api_calls_last_30d",
    "tokens_consumed_last_30d",
    "tools_used_count",
    "failed_requests_rate",
    "support_tickets_last_90d",
    "plan_tier",  # encoded to ordinal int in features
    "payment_failures_last_90d",
    "feature_adoption_score",
    "nps_score",
    "last_active_days_ago",
    "weekend_usage_ratio",
    "engagement_trend",
    "spend_usd_last_30d",
    "days_until_renewal",
    "agent_runs_last_30d",
    "ide_plugin_sessions_last_30d",
    "seat_utilization",
]

# Numeric features after plan_tier is encoded
MODEL_FEATURE_COLUMNS = [
    "days_since_signup",
    "sessions_last_7d",
    "sessions_last_30d",
    "avg_session_minutes",
    "models_used_count",
    "api_calls_last_30d",
    "tokens_consumed_last_30d",
    "tools_used_count",
    "failed_requests_rate",
    "support_tickets_last_90d",
    "plan_tier_code",
    "payment_failures_last_90d",
    "feature_adoption_score",
    "nps_score",
    "last_active_days_ago",
    "weekend_usage_ratio",
    "engagement_trend",
    "spend_usd_last_30d",
    "days_until_renewal",
    "agent_runs_last_30d",
    "ide_plugin_sessions_last_30d",
    "seat_utilization",
]

# Human-readable descriptions for data dictionary / model card
FEATURE_DESCRIPTIONS = {
    "user_id": "Stable synthetic user identifier.",
    "user_name": "Display name (synthetic; Santosh is the hero profile).",
    "days_since_signup": "Days since account creation.",
    "sessions_last_7d": "Product sessions in the last 7 days.",
    "sessions_last_30d": "Product sessions in the last 30 days.",
    "avg_session_minutes": "Average session length in minutes.",
    "models_used_count": "Distinct AI models the user has invoked.",
    "api_calls_last_30d": "API calls in the last 30 days.",
    "tokens_consumed_last_30d": "Token usage in the last 30 days.",
    "tools_used_count": "Distinct tools / integrations used.",
    "failed_requests_rate": "Fraction of failed requests (0–1).",
    "support_tickets_last_90d": "Support tickets opened in last 90 days.",
    "plan_tier": "Subscription tier: free | starter | pro | enterprise.",
    "plan_tier_code": "Ordinal encoding of plan_tier (free=0 … enterprise=3).",
    "payment_failures_last_90d": "Failed payment attempts in last 90 days.",
    "feature_adoption_score": "0–1 score of how broadly features are used.",
    "nps_score": "Net Promoter Score style rating (0–10).",
    "last_active_days_ago": "Days since last observed activity.",
    "weekend_usage_ratio": "Share of usage that happens on weekends (0–1).",
    "engagement_trend": (
        "sessions_last_7d / max(1, sessions_last_30d/4); "
        "≈1 stable, <1 cooling, >1 accelerating."
    ),
    "spend_usd_last_30d": "Synthetic monthly spend in USD (last 30 days).",
    "days_until_renewal": "Days until next billing / renewal (B2B-ish).",
    "agent_runs_last_30d": "Agent / automation runs in the last 30 days (AI-native).",
    "ide_plugin_sessions_last_30d": "IDE plugin sessions in the last 30 days.",
    "seat_utilization": "0–1 seats used vs seats provisioned (team signal).",
    "churned": "Binary label: 1 = churned, 0 = retained (training only).",
}

# Inference payload required keys (no churned label)
INFERENCE_REQUIRED_KEYS = [
    "user_id",
    "user_name",
    "days_since_signup",
    "sessions_last_7d",
    "sessions_last_30d",
    "avg_session_minutes",
    "models_used_count",
    "api_calls_last_30d",
    "tokens_consumed_last_30d",
    "tools_used_count",
    "failed_requests_rate",
    "support_tickets_last_90d",
    "plan_tier",
    "payment_failures_last_90d",
    "feature_adoption_score",
    "nps_score",
    "last_active_days_ago",
    "weekend_usage_ratio",
    "engagement_trend",
    "spend_usd_last_30d",
    "days_until_renewal",
    "agent_runs_last_30d",
    "ide_plugin_sessions_last_30d",
    "seat_utilization",
]

# Soft range hints for single-record validation
FEATURE_RANGES = {
    "days_since_signup": (1, 2000),
    "sessions_last_7d": (0, 60),
    "sessions_last_30d": (0, 200),
    "avg_session_minutes": (0.5, 240.0),
    "models_used_count": (0, 30),
    "api_calls_last_30d": (0, 100000),
    "tokens_consumed_last_30d": (0, 50_000_000),
    "tools_used_count": (0, 40),
    "failed_requests_rate": (0.0, 1.0),
    "support_tickets_last_90d": (0, 50),
    "payment_failures_last_90d": (0, 20),
    "feature_adoption_score": (0.0, 1.0),
    "nps_score": (0.0, 10.0),
    "last_active_days_ago": (0, 365),
    "weekend_usage_ratio": (0.0, 1.0),
    "engagement_trend": (0.0, 5.0),
    "spend_usd_last_30d": (0.0, 100000.0),
    "days_until_renewal": (0, 730),
    "agent_runs_last_30d": (0, 5000),
    "ide_plugin_sessions_last_30d": (0, 500),
    "seat_utilization": (0.0, 1.0),
}

# Cohort comparison keys shown in decision packet / Streamlit
COHORT_COMPARE_FEATURES = [
    "sessions_last_30d",
    "engagement_trend",
    "feature_adoption_score",
    "failed_requests_rate",
    "last_active_days_ago",
    "spend_usd_last_30d",
    "days_until_renewal",
    "agent_runs_last_30d",
    "ide_plugin_sessions_last_30d",
    "seat_utilization",
    "nps_score",
]

ID_COLUMNS = ["user_id", "user_name"]
TARGET_COLUMN = "churned"

# Ordered plan tiers (higher = more valuable)
PLAN_TIER_ORDER = ["free", "starter", "pro", "enterprise"]
PLAN_TIER_MAP = {name: idx for idx, name in enumerate(PLAN_TIER_ORDER)}

# Train / val / test fractions (of full data)
# TEST_SIZE=0.20 → 20% test. VAL_SIZE=0.20 of full → ~25% of remaining → 20% val, 60% train.
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# Dataset size / Optuna (env overrides for CI speed; local defaults unchanged)
N_USERS = int(os.environ.get("N_USERS", "5000"))
N_OPTUNA_TRIALS = int(os.environ.get("N_OPTUNA_TRIALS", "20"))

# Calibration method: "isotonic" (preferred) or "sigmoid"
CALIBRATION_METHOD = "isotonic"

# Latency benchmark defaults
LATENCY_WARMUP = 20
LATENCY_RUNS = 200
