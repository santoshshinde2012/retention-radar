"""Generate a synthetic renewal cohort for a $20/month AI coding assistant.

Each row is one paying subscriber seven days before a monthly renewal (T-7).
The story the generator encodes comes from the public 2025-26 record of AI
coding tools (usage caps tightened, pricing moved to usage, quality incidents,
developers running two or three tools at once):

* Unobserved causes: how much the subscriber codes with AI (``need``), how well
  the assistant fits their work (``fit``), price sensitivity, a side-project
  habit, and whether a rival tool has started taking their work (``pull``).
* Observed footprints: active days, agent requests, allowance used, cap hits,
  cheap-model rationing, overage switched off, accept rate, agent success,
  incidents, tickets, IDE vs CLI sessions.
* Outcome at renewal: renewed, voluntary lapse, or involuntary lapse (the card
  failed and retries ran out). Some voluntary churners schedule the cancel
  before T-7; those rows are already decided and go to the cancel flow.

Only rows routed to the model (not dunning, not cancel flow) are written to the
training table ``renewals_t7.csv``. ``renewals_all.csv`` keeps every renewal with
its outcome and route so the label can be audited.

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

# Relative size of each plan's included usage (Pro = 1). Ultra is "20x".
_ALLOWANCE = {"pro": 1.0, "pro_plus": 3.0, "ultra": 20.0}
# Frontier-model requests one full allowance buys in 28 days.
_REQUESTS_PER_ALLOWANCE = 550
# The most recent limit change cut every plan's cap by 17%.
_CAP_AFTER_CHANGE = 0.83


def _clip(arr, lo, hi):
    return np.clip(arr, lo, hi)


def engagement_trend_from_days(active_days_7d, active_days_28d):
    """active_days_7d / max(1, active_days_28d / 4); ~1 steady, <1 fading."""
    d7 = np.asarray(active_days_7d, dtype=float)
    d28 = np.asarray(active_days_28d, dtype=float)
    out = d7 / np.maximum(1.0, d28 / 4.0)
    if np.ndim(active_days_7d) == 0:
        return float(out)
    return out


def generate_renewals(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Return every renewal in the cohort with features, outcome and route."""
    n = config.N_USERS if n is None else n
    seed = config.RANDOM_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    # --- unobserved causes --------------------------------------------------
    need = rng.beta(2.0, 2.2, size=n)
    fit = rng.normal(0.0, 1.0, size=n)
    price_sensitive = rng.beta(2.0, 3.0, size=n)
    side_project = rng.random(n) < 0.30
    pull = np.where(rng.random(n) < 0.24, rng.uniform(0.3, 1.0, size=n), 0.0)

    # Heavy users self-select into bigger plans; price-sensitive ones stay on Pro.
    heavy = need + rng.normal(0, 0.15, size=n) - 0.4 * price_sensitive
    plan_tier = np.where(heavy > 0.80, "ultra", np.where(heavy > 0.55, "pro_plus", "pro"))
    allowance = np.array([_ALLOWANCE[p] for p in plan_tier])
    price = np.array([config.PLAN_PRICE_USD[p] for p in plan_tier])

    # Tenure in paid months: most subscribers are young (AI tourists churn early).
    renewals_completed = _clip(rng.geometric(0.16, size=n) - 1, 0, 48).astype(int)
    # Whose renewal is the first one since the cap was cut (grandfathered until now).
    first_after_change = (rng.random(n) < 0.35).astype(int)

    # --- activity -----------------------------------------------------------
    base_rate = _clip(0.12 + 0.62 * need - 0.25 * pull + rng.normal(0, 0.05, n), 0.02, 0.95)
    # A side project goes quiet between projects; a rival tool takes recent work.
    lull = side_project & (rng.random(n) < 0.35)
    recent_rate = _clip(
        base_rate * (1 - 0.75 * pull) * np.where(lull, 0.25, 1.0) + rng.normal(0, 0.04, n),
        0.0,
        0.98,
    )
    active_days_28d = rng.binomial(28, base_rate)
    active_days_7d = np.minimum(rng.binomial(7, recent_rate), active_days_28d)
    engagement_trend = _clip(
        engagement_trend_from_days(active_days_7d, active_days_28d), 0.0, 4.0
    ).round(4)
    last_active_days_ago = np.where(
        active_days_7d > 0,
        rng.integers(0, 3, size=n),
        7 + _clip(rng.exponential(6 + 10 * pull), 0, 60).astype(int),
    )
    last_active_days_ago = _clip(last_active_days_ago, 0, 90).astype(int)

    weekend_usage_ratio = np.where(
        side_project, rng.beta(5, 4, size=n), rng.beta(2, 8, size=n)
    ).round(4)
    ide_share = rng.beta(3, 2, size=n)
    ide_sessions_28d = rng.poisson(
        active_days_28d * 2.2 * ide_share * (1 - 0.5 * pull)
    ).clip(0, 500)
    cli_sessions_28d = rng.poisson(active_days_28d * 1.8 * (1 - ide_share)).clip(0, 500)

    # --- usage against the cap ------------------------------------------------
    demand = need * rng.lognormal(0.0, 0.55, size=n) * 1.6 * (1 - 0.6 * pull)
    effective_allowance = allowance * _CAP_AFTER_CHANGE
    allowance_used_pct = _clip(demand / effective_allowance, 0.0, 3.0)
    # Caps (5-hour / weekly) bite once someone runs near or past their allowance.
    limit_hits_14d = rng.poisson(np.maximum(0.0, allowance_used_pct - 0.75) * 5.5).clip(0, 60)
    # Some heavy users switch overage on; a surprise bill makes many switch it off.
    overage_on = (allowance_used_pct > 0.9) & (rng.random(n) < 0.35)
    overage_usd_28d = np.where(
        overage_on,
        np.maximum(0.0, allowance_used_pct - 1.0) * price * rng.uniform(0.6, 1.4, n),
        0.0,
    ).round(2)
    overage_toggled_off = (
        overage_on
        & (rng.random(n) < _clip(0.15 + 0.6 * price_sensitive + overage_usd_28d / 150, 0, 0.95))
    ).astype(int)
    # When the cap bites, people ration by routing work to the cheaper model.
    cheap_model_share_28d = _clip(
        0.12 + 0.30 * price_sensitive + 0.07 * np.minimum(limit_hits_14d, 6)
        + rng.normal(0, 0.06, n),
        0.0,
        1.0,
    ).round(4)
    agent_requests_28d = rng.poisson(
        np.minimum(allowance_used_pct, 1.0 + (overage_usd_28d > 0))
        * effective_allowance
        * _REQUESTS_PER_ALLOWANCE
    ).clip(0, 50000)

    # --- quality and friction -------------------------------------------------
    incident_exposed_28d = (rng.random(n) < 0.30).astype(int)
    suggestion_accept_rate_28d = _clip(
        0.27 + 0.05 * fit + rng.normal(0, 0.05, n), 0.02, 0.9
    ).round(4)
    accept_rate_change = _clip(
        1.0 + 0.07 * fit - 0.06 * incident_exposed_28d - 0.10 * pull + rng.normal(0, 0.07, n),
        0.3,
        2.0,
    ).round(4)
    agent_task_success_rate = _clip(
        0.60 + 0.09 * fit - 0.06 * incident_exposed_28d + rng.normal(0, 0.07, n), 0.0, 1.0
    ).round(4)
    failed_requests_rate = _clip(
        0.02 + 0.05 * incident_exposed_28d + rng.exponential(0.02, n), 0.0, 1.0
    ).round(4)
    support_tickets_90d = rng.poisson(
        0.15 + 0.35 * incident_exposed_28d + 0.25 * (limit_hits_14d >= 3) + 0.5 * overage_toggled_off
    ).clip(0, 50)

    # --- voluntary lapse at this renewal --------------------------------------
    tenure_risk = np.select(
        [renewals_completed == 0, renewals_completed == 1, renewals_completed <= 3],
        [0.95, 0.5, 0.2],
        default=-0.3 * np.log1p(np.maximum(renewals_completed - 3, 0)),
    )
    cap_pain = np.minimum(limit_hits_14d, 6) * np.where(plan_tier == "pro", 0.30, 0.14)
    logit = (
        -3.75
        + tenure_risk
        + 1.7 * pull
        - 0.50 * fit
        + 0.9 * price_sensitive
        - 1.1 * (need - 0.5)
        + 0.55 * (side_project & (active_days_7d == 0))
        + cap_pain * (1.0 + 0.9 * first_after_change)
        + 0.35 * first_after_change * (plan_tier == "pro")
        + 0.85 * overage_toggled_off
        + 0.30 * incident_exposed_28d * (fit < 0)
        + rng.normal(0, 0.55, n)
    )
    p_voluntary = 1.0 / (1.0 + np.exp(-logit))
    voluntary = rng.random(n) < p_voluntary

    # Involuntary lapse: the card failed and smart retries ran out. Mostly unrelated
    # to how the product is used; slightly higher on the first renewal.
    p_involuntary = 0.038 + 0.02 * (renewals_completed == 0)
    involuntary = (~voluntary) & (rng.random(n) < p_involuntary)
    # About a third of voluntary churners schedule the cancel before T-7.
    scheduled_before_t7 = voluntary & (rng.random(n) < 0.35)

    outcome = np.where(
        voluntary,
        config.OUTCOME_VOLUNTARY,
        np.where(involuntary, config.OUTCOME_INVOLUNTARY, config.OUTCOME_RENEWED),
    )
    route = np.where(
        involuntary,
        config.ROUTE_DUNNING,
        np.where(scheduled_before_t7, config.ROUTE_CANCEL_FLOW, config.ROUTE_MODEL),
    )

    # Renewal dates across one quarter; features are snapshotted 7 days earlier.
    renewal_date = pd.Timestamp("2026-07-01") + pd.to_timedelta(
        rng.integers(0, 91, size=n), unit="D"
    )
    as_of_date = renewal_date - pd.Timedelta(days=7)

    first_names = ["Alex", "Jordan", "Sam", "Riley", "Casey", "Morgan", "Taylor", "Quinn"]
    last_names = ["Lee", "Patel", "Garcia", "Kim", "Nguyen", "Brown", "Davis", "Wilson"]
    user_names = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n)]

    return pd.DataFrame(
        {
            "user_id": [f"sub_{i:05d}" for i in range(n)],
            "user_name": user_names,
            "plan_tier": plan_tier,
            "renewals_completed": renewals_completed,
            "active_days_7d": active_days_7d.astype(int),
            "active_days_28d": active_days_28d.astype(int),
            "engagement_trend": engagement_trend,
            "last_active_days_ago": last_active_days_ago,
            "agent_requests_28d": agent_requests_28d.astype(int),
            "allowance_used_pct": allowance_used_pct.round(4),
            "limit_hits_14d": limit_hits_14d.astype(int),
            "cheap_model_share_28d": cheap_model_share_28d,
            "overage_usd_28d": overage_usd_28d,
            "overage_toggled_off": overage_toggled_off,
            "suggestion_accept_rate_28d": suggestion_accept_rate_28d,
            "accept_rate_change": accept_rate_change,
            "agent_task_success_rate": agent_task_success_rate,
            "failed_requests_rate": failed_requests_rate,
            "incident_exposed_28d": incident_exposed_28d,
            "support_tickets_90d": support_tickets_90d.astype(int),
            "ide_sessions_28d": ide_sessions_28d.astype(int),
            "cli_sessions_28d": cli_sessions_28d.astype(int),
            "weekend_usage_ratio": weekend_usage_ratio,
            "first_renewal_after_pricing_change": first_after_change,
            "as_of_date": as_of_date.strftime("%Y-%m-%d"),
            "renewal_date": renewal_date.strftime("%Y-%m-%d"),
            "outcome": outcome,
            "route": route,
            config.TARGET_COLUMN: voluntary.astype(int),
        }
    )


def model_table(renewals: pd.DataFrame) -> pd.DataFrame:
    """Rows the churn model trains on: not in dunning, cancel not already scheduled."""
    keep = renewals["route"] == config.ROUTE_MODEL
    cols = config.ID_COLUMNS + config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    return renewals.loc[keep, cols].reset_index(drop=True)


# Back-compat name used by older notebooks.
def generate_users(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    return model_table(generate_renewals(n=n, seed=seed))


def _with_trend(record: dict) -> dict:
    record["engagement_trend"] = round(
        float(engagement_trend_from_days(record["active_days_7d"], record["active_days_28d"])), 4
    )
    return record


def maya_profile() -> dict:
    """Pro subscriber, fourth renewal, the first since the weekly cap was cut.

    She hit the cap four times in two weeks, moved most work to the cheaper
    model, and her active days fell off while CLI sessions held up. Scored at
    T-7; the renewal has not happened, so there is no label.
    """
    return _with_trend(
        {
            "user_id": "sub_maya",
            "user_name": "Maya (worked example)",
            "plan_tier": "pro",
            "renewals_completed": 3,
            "active_days_7d": 2,
            "active_days_28d": 15,
            "last_active_days_ago": 2,
            "agent_requests_28d": 470,
            "allowance_used_pct": 1.02,
            "limit_hits_14d": 4,
            "cheap_model_share_28d": 0.68,
            "overage_usd_28d": 0.0,
            "overage_toggled_off": 0,
            "suggestion_accept_rate_28d": 0.29,
            "accept_rate_change": 0.96,
            "agent_task_success_rate": 0.58,
            "failed_requests_rate": 0.05,
            "incident_exposed_28d": 1,
            "support_tickets_90d": 0,
            "ide_sessions_28d": 14,
            "cli_sessions_28d": 19,
            "weekend_usage_ratio": 0.21,
            "first_renewal_after_pricing_change": 1,
        }
    )


def arjun_profile() -> dict:
    """Steady Pro+ subscriber, a year in, well inside the allowance."""
    return _with_trend(
        {
            "user_id": "sub_arjun",
            "user_name": "Arjun (worked example)",
            "plan_tier": "pro_plus",
            "renewals_completed": 12,
            "active_days_7d": 5,
            "active_days_28d": 20,
            "last_active_days_ago": 0,
            "agent_requests_28d": 910,
            "allowance_used_pct": 0.66,
            "limit_hits_14d": 0,
            "cheap_model_share_28d": 0.18,
            "overage_usd_28d": 0.0,
            "overage_toggled_off": 0,
            "suggestion_accept_rate_28d": 0.33,
            "accept_rate_change": 1.03,
            "agent_task_success_rate": 0.69,
            "failed_requests_rate": 0.03,
            "incident_exposed_28d": 1,
            "support_tickets_90d": 1,
            "ide_sessions_28d": 41,
            "cli_sessions_28d": 12,
            "weekend_usage_ratio": 0.12,
            "first_renewal_after_pricing_change": 0,
        }
    )


HERO_PROFILES = {"maya": maya_profile, "arjun": arjun_profile}


def write_heroes() -> None:
    config.HERO_DIR.mkdir(parents=True, exist_ok=True)
    for key, fn in HERO_PROFILES.items():
        path = config.HERO_DIR / config.HEROES[key]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fn(), f, indent=2)
        print(f"Wrote {path}")


def main(n_users: int | None = None) -> None:
    """Write renewals_all.csv, renewals_t7.csv and the worked-example JSONs.

    Size resolution (first wins): explicit arg, ``--n-users``, ``N_USERS`` env,
    config default (8000 renewals before routing).
    """
    parser = argparse.ArgumentParser(description="Generate a synthetic renewal cohort")
    parser.add_argument("--n-users", type=int, default=None)
    args, _ = parser.parse_known_args()

    if n_users is not None:
        n = int(n_users)
    elif args.n_users is not None:
        n = int(args.n_users)
    else:
        n = int(os.environ.get("N_USERS", config.N_USERS))

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n} renewals (seed={config.RANDOM_SEED})...")
    renewals = generate_renewals(n=n)
    renewals.to_csv(config.RENEWALS_ALL_CSV, index=False)
    routes = renewals["route"].value_counts().to_dict()
    outcomes = renewals["outcome"].value_counts().to_dict()
    print(f"Wrote {config.RENEWALS_ALL_CSV}  outcomes={outcomes}  routes={routes}")

    table = model_table(renewals)
    table.to_csv(config.USERS_CSV, index=False)
    print(
        f"Wrote {config.USERS_CSV}  rows={len(table)}  "
        f"voluntary_lapse_rate={table[config.TARGET_COLUMN].mean():.3f}"
    )
    write_heroes()


if __name__ == "__main__":
    main()
