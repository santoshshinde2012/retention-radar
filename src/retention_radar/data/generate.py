"""Generate synthetic AI-platform user data and Santosh Shinde's profile.

Run from project root:
    python -m retention_radar.cli.generate_data
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from retention_radar import config


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def engagement_trend_from_sessions(
    sessions_last_7d: np.ndarray | float,
    sessions_last_30d: np.ndarray | float,
) -> np.ndarray | float:
    """sessions_last_7d / max(1, sessions_last_30d/4); ≈1 stable, <1 cooling."""
    s7 = np.asarray(sessions_last_7d, dtype=float)
    s30 = np.asarray(sessions_last_30d, dtype=float)
    denom = np.maximum(1.0, s30 / 4.0)
    out = s7 / denom
    if np.ndim(sessions_last_7d) == 0:
        return float(out)
    return out


def generate_users(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Create n synthetic users with a rule-based churn label + noise."""
    n = config.N_USERS if n is None else n
    seed = config.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    # Latent engagement (higher = healthier usage)
    engagement = rng.beta(2.2, 1.6, size=n)

    # Plan tier biased toward free/starter
    plan_probs = [0.40, 0.30, 0.22, 0.08]
    plan_tier = rng.choice(config.PLAN_TIER_ORDER, size=n, p=plan_probs)
    plan_boost = np.array(
        [{"free": 0.0, "starter": 0.15, "pro": 0.35, "enterprise": 0.55}[p] for p in plan_tier]
    )

    days_since_signup = rng.integers(14, 900, size=n)
    sessions_last_30d = _clip(
        (engagement + plan_boost) * rng.uniform(8, 45, size=n) + rng.normal(0, 3, size=n),
        0,
        120,
    ).astype(int)
    sessions_last_7d = _clip(
        sessions_last_30d * rng.uniform(0.15, 0.40, size=n) + rng.normal(0, 1, size=n),
        0,
        40,
    ).astype(int)
    avg_session_minutes = _clip(
        5 + engagement * 40 + rng.normal(0, 5, size=n), 1, 120
    ).round(1)
    models_used_count = _clip(
        engagement * 8 + plan_boost * 4 + rng.normal(0, 1.5, size=n), 0, 15
    ).astype(int)
    api_calls_last_30d = _clip(
        sessions_last_30d * rng.uniform(5, 40, size=n) * (0.5 + engagement),
        0,
        20000,
    ).astype(int)
    tokens_consumed_last_30d = _clip(
        api_calls_last_30d * rng.uniform(80, 600, size=n), 0, 5_000_000
    ).astype(int)
    tools_used_count = _clip(
        engagement * 10 + plan_boost * 5 + rng.normal(0, 1.5, size=n), 0, 20
    ).astype(int)

    # Friction signals (inversely related to engagement, with noise)
    failed_requests_rate = _clip(
        (1 - engagement) * 0.35 + rng.uniform(0, 0.08, size=n), 0.0, 0.95
    ).round(4)
    support_tickets_last_90d = _clip(
        (1 - engagement) * 6 + rng.poisson(0.4, size=n), 0, 20
    ).astype(int)
    payment_failures_last_90d = _clip(
        (1 - engagement) * 3 + rng.poisson(0.2, size=n) * (plan_tier != "free"),
        0,
        10,
    ).astype(int)

    feature_adoption_score = _clip(
        engagement * 0.7 + plan_boost * 0.25 + rng.normal(0, 0.05, size=n), 0.0, 1.0
    ).round(4)
    nps_score = _clip(
        3 + engagement * 7 + rng.normal(0, 1.2, size=n), 0, 10
    ).round(1)
    last_active_days_ago = _clip(
        (1 - engagement) * 45 + rng.exponential(3, size=n), 0, 120
    ).astype(int)
    weekend_usage_ratio = _clip(rng.beta(2, 5, size=n), 0.0, 1.0).round(4)

    # --- AI-churn / B2B extras ---
    engagement_trend = engagement_trend_from_sessions(sessions_last_7d, sessions_last_30d)
    engagement_trend = _clip(
        np.asarray(engagement_trend, dtype=float) + rng.normal(0, 0.05, size=n),
        0.0,
        4.0,
    ).round(4)

    base_spend = np.array(
        [{"free": 0.0, "starter": 29.0, "pro": 99.0, "enterprise": 499.0}[p] for p in plan_tier]
    )
    spend_usd_last_30d = _clip(
        base_spend * (0.6 + engagement) + rng.normal(0, 15, size=n) * (plan_tier != "free"),
        0.0,
        5000.0,
    ).round(2)

    days_until_renewal = _clip(
        rng.integers(1, 365, size=n).astype(float)
        + (1 - engagement) * rng.uniform(-20, 10, size=n),
        0,
        365,
    ).astype(int)

    agent_runs_last_30d = _clip(
        engagement * 40 + plan_boost * 30 + rng.normal(0, 8, size=n), 0, 400
    ).astype(int)
    ide_plugin_sessions_last_30d = _clip(
        engagement * 25 + plan_boost * 15 + rng.normal(0, 5, size=n), 0, 200
    ).astype(int)
    seat_utilization = _clip(
        0.25 + engagement * 0.55 + plan_boost * 0.15 + rng.normal(0, 0.08, size=n),
        0.0,
        1.0,
    ).round(4)

    # Rule-based churn propensity (then Bernoulli + noise)
    logit = (
        -1.8
        + 0.045 * last_active_days_ago
        + 2.8 * failed_requests_rate
        + 0.22 * support_tickets_last_90d
        + 0.35 * payment_failures_last_90d
        - 0.012 * sessions_last_30d
        - 0.018 * sessions_last_7d
        - 1.6 * feature_adoption_score
        - 0.12 * nps_score
        - 0.04 * models_used_count
        - 0.03 * tools_used_count
        - 0.15 * plan_boost
        - 0.55 * (engagement_trend - 1.0)  # cooling (<1) raises risk
        - 0.0015 * spend_usd_last_30d  # invested spend is protective
        + 0.004 * np.maximum(0, 45 - days_until_renewal)  # near renewal ↑ risk
        - 0.008 * agent_runs_last_30d
        - 0.012 * ide_plugin_sessions_last_30d
        - 0.9 * seat_utilization
    )
    # Additive noise so the model cannot recover the exact formula
    logit = logit + rng.normal(0, 0.55, size=n)
    propensity = 1.0 / (1.0 + np.exp(-logit))
    churned = (rng.random(n) < propensity).astype(int)

    user_ids = [f"user_{i:05d}" for i in range(n)]
    first_names = ["Alex", "Jordan", "Sam", "Riley", "Casey", "Morgan", "Taylor", "Quinn"]
    last_names = ["Lee", "Patel", "Garcia", "Kim", "Nguyen", "Brown", "Davis", "Wilson"]
    user_names = [
        f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)
    ]

    df = pd.DataFrame(
        {
            "user_id": user_ids,
            "user_name": user_names,
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
            "churned": churned,
        }
    )
    return df


def santosh_profile() -> dict:
    """Power-user metrics with mild risk (slightly higher inactivity / failures)."""
    sessions_last_7d = 9
    sessions_last_30d = 38
    trend = round(
        float(engagement_trend_from_sessions(sessions_last_7d, sessions_last_30d)), 4
    )
    return {
        "user_id": "santosh_shinde",
        "user_name": "Santosh Shinde",
        "days_since_signup": 420,
        "sessions_last_7d": sessions_last_7d,
        "sessions_last_30d": sessions_last_30d,
        "avg_session_minutes": 28.5,
        "models_used_count": 7,
        "api_calls_last_30d": 1850,
        "tokens_consumed_last_30d": 420000,
        "tools_used_count": 8,
        "failed_requests_rate": 0.12,  # mild friction
        "support_tickets_last_90d": 2,
        "plan_tier": "pro",
        "payment_failures_last_90d": 1,
        "feature_adoption_score": 0.78,
        "nps_score": 7.0,
        "last_active_days_ago": 8,  # slightly colder than a hot power user
        "weekend_usage_ratio": 0.22,
        "engagement_trend": trend,  # ~0.947 — mild cooling
        "spend_usd_last_30d": 189.0,
        "days_until_renewal": 21,  # approaching renewal window
        "agent_runs_last_30d": 52,
        "ide_plugin_sessions_last_30d": 28,
        "seat_utilization": 0.72,  # solid but not fully seated
        "churned": 0,
    }


def inject_santosh(df: pd.DataFrame) -> pd.DataFrame:
    """Replace first row with Santosh so the hero is always in the CSV."""
    profile = santosh_profile()
    for col, val in profile.items():
        df.at[0, col] = val
    return df


def main(n_users: int | None = None) -> None:
    """Write users.csv + santosh_shinde.json.

    Size resolution (first wins):
      1. explicit ``n_users`` arg
      2. ``--n-users`` CLI
      3. ``N_USERS`` env / config default (5000)
    """
    parser = argparse.ArgumentParser(description="Generate synthetic AI-platform users")
    parser.add_argument(
        "--n-users",
        type=int,
        default=None,
        help="Number of users (default: N_USERS env or 5000)",
    )
    args, _ = parser.parse_known_args()

    if n_users is not None:
        n = int(n_users)
    elif args.n_users is not None:
        n = int(args.n_users)
    else:
        n = int(os.environ.get("N_USERS", config.N_USERS))

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n} synthetic users (seed={config.RANDOM_SEED})...")
    df = generate_users(n=n)
    df = inject_santosh(df)

    df.to_csv(config.USERS_CSV, index=False)
    print(
        f"Wrote {config.USERS_CSV}  shape={df.shape}  "
        f"churn_rate={df['churned'].mean():.3f}"
    )

    profile = santosh_profile()
    infer_payload = {k: v for k, v in profile.items() if k != "churned"}
    with open(config.SANTOSH_JSON, "w", encoding="utf-8") as f:
        json.dump(infer_payload, f, indent=2)
    print(f"Wrote {config.SANTOSH_JSON}")


if __name__ == "__main__":
    main()
