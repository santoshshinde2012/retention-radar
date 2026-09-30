"""Risk bands and the renewal decision policy: single source of truth.

A $20/month subscriber never gets a person reading their profile. The score
picks which *approved playbook* (if any) a lifecycle tool should run, and a
fixed share of eligible subscribers is held out so the lift can be measured.
The service itself never sends anything: ``auto_action`` is always ``none``;
it writes a queue that a messaging tool consumes after a human approved the
playbook. Only the Ultra plan has a playbook with a person in it.

Every effect / cost number lives in ``config.PLAYBOOKS`` and is an assumption
until a holdout replaces it with a measurement.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from retention_radar import config

NO_ACTION = "no_action"
HOLDOUT = "holdout"


def risk_band(prob: float) -> str:
    """Map a calibrated probability onto low / medium / high (fixed edges, not τ)."""
    if not math.isfinite(prob):
        raise ValueError(f"risk_band needs a finite probability, got {prob!r}")
    low, high = config.RISK_BAND_EDGES
    if prob < low:
        return "low"
    if prob < high:
        return "medium"
    return "high"


def in_holdout(user_id: str | None, pct: int | None = None) -> bool:
    """Deterministic holdout assignment from a hash of the subscriber id."""
    if not user_id:
        return False
    pct = config.HOLDOUT_PCT if pct is None else pct
    digest = hashlib.sha256(f"holdout:{user_id}".encode()).hexdigest()
    return int(digest[:8], 16) % 100 < pct


def _price(record: dict | None) -> float:
    tier = (record or {}).get("plan_tier", "pro")
    return float(config.PLAN_PRICE_USD.get(tier, config.PLAN_PRICE_USD["pro"]))


def eligible_playbooks(prob: float, threshold: float, record: dict | None) -> list[str]:
    """Which approved playbooks apply to this subscriber at all."""
    r = record or {}
    tier = r.get("plan_tier", "pro")
    out = ["in_app_usage_tips"]
    if float(r.get("limit_hits_14d", 0)) >= 2:
        out.append("limit_reset")
    if float(r.get("weekend_usage_ratio", 0)) >= 0.4 and float(r.get("active_days_7d", 7)) <= 1:
        out.append("pause_offer")
    if prob >= threshold:
        out.append("cancel_flow_discount")
        if tier in config.PLAYBOOKS["personal_email"].get("plans", []):
            out.append("personal_email")
    return out


def expected_value(name: str, prob: float, record: dict | None) -> float:
    """Expected net revenue (USD) of running playbook ``name`` for one subscriber.

    saved     = p(lapse) x effect             (would-be churners the playbook keeps)
    revenue   = saved x price x months retained after a save
    cost      = per-send cost + discount x (saved + (1 - p) x sure-thing acceptance)

    The discount term is the part teams forget: subscribers who would have renewed
    anyway also take the offer.
    """
    pb = config.PLAYBOOKS[name]
    price = _price(record)
    discount = float(pb.get("discount_usd", 0.0))
    discount += float(pb.get("discount_usd_pct_of_price", 0.0)) * price
    discount += float(pb.get("pause_months", 0.0)) * price
    saved = prob * float(pb["effect"])
    revenue = saved * price * config.MONTHS_RETAINED_AFTER_SAVE
    cost = float(pb["cost_usd"]) + discount * (saved + (1 - prob) * float(pb["sure_thing_accept"]))
    return revenue - cost


def decide(prob: float, threshold: float, band: str, record: dict | None = None) -> dict[str, Any]:
    """Pick one action for a subscriber scored at T-7.

    1. Below τ: no action. τ is the validation best-F1 threshold; below it most
       contacted subscribers would have renewed anyway, contact has a cost, and a
       "we noticed you're quieter" message can remind someone to cancel.
    2. Otherwise, if the subscriber is in the holdout: no contact, but logged, so
       the playbook's lift can be measured against them.
    3. Otherwise run the eligible playbook with the highest expected value, or
       nothing if none pays back.
    """
    base = {
        "risk_band": band,
        "auto_action": "none",
        "holdout": False,
        "expected_value_usd": None,
        "candidates": [],
        "would_have_sent": None,
        "hitl_required": False,
    }
    if prob < threshold:
        return {
            **base,
            "action": NO_ACTION,
            "rationale": (
                f"Calibrated P(lapse)={prob:.3f} is below τ ({threshold:.3f}). "
                "No contact: the renewal is likely and outreach has a cost."
            ),
        }

    uid = (record or {}).get("user_id")
    candidates = [
        {"playbook": name, "expected_value_usd": round(expected_value(name, prob, record), 2)}
        for name in eligible_playbooks(prob, threshold, record)
    ]
    candidates.sort(key=lambda c: c["expected_value_usd"], reverse=True)
    base["candidates"] = candidates

    best_positive = next((c["playbook"] for c in candidates if c["expected_value_usd"] > 0), None)
    base["would_have_sent"] = best_positive
    if in_holdout(uid):
        return {
            **base,
            "action": HOLDOUT,
            "holdout": True,
            "rationale": (
                f"Eligible (P(lapse)={prob:.3f}) but in the {config.HOLDOUT_PCT}% holdout. "
                "No contact, so the playbooks' lift can be measured against this group."
            ),
        }

    best = candidates[0] if candidates else None
    if best is None or best["expected_value_usd"] <= 0:
        return {
            **base,
            "action": NO_ACTION,
            "rationale": (
                f"P(lapse)={prob:.3f}, but no approved playbook has a positive expected "
                "value at this risk and price."
            ),
        }
    name = best["playbook"]
    human = name == "personal_email"
    return {
        **base,
        "action": name,
        "playbook_label": config.PLAYBOOKS[name]["label"],
        "expected_value_usd": best["expected_value_usd"],
        "hitl_required": human,
        "rationale": (
            f"P(lapse)={prob:.3f} (τ={threshold:.3f}). Highest expected value among "
            f"{[c['playbook'] for c in candidates]}: {config.PLAYBOOKS[name]['label']} "
            f"(≈${best['expected_value_usd']:.2f}, assumption-based)."
            + (" A person writes this one." if human else " Sent by the lifecycle tool.")
        ),
    }


HOLD_ACTION = "hold: fix input data"


def validation_hold(errors: list[str]) -> dict[str, Any]:
    """Decision block for a record that failed validation: never scored, never queued."""
    return {
        "action": HOLD_ACTION,
        "rationale": "Input failed validation, so it is not scored or queued: " + "; ".join(errors),
        "auto_action": "none",
        "holdout": False,
        "hitl_required": True,
        "blocked_by_validation": True,
    }


class RenewalDecisionPolicy:
    """``DecisionPolicy`` implementation used by packet, batch, API and UI."""

    def decide(
        self, prob: float, threshold: float, band: str, record: dict | None = None
    ) -> dict[str, Any]:
        return decide(prob, threshold, band, record)


# Back-compat aliases for older imports.
HitlDecisionPolicy = RenewalDecisionPolicy


def hitl_action(prob: float, threshold: float, band: str, record: dict | None = None) -> dict[str, Any]:
    return decide(prob, threshold, band, record)
