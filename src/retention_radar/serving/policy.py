"""HITL decision policy and risk bands — single source of truth.

Serving CLI, packet builder, and Streamlit must call these helpers rather than
duplicating band cutoffs or auto-action rules.
"""

from __future__ import annotations

import math
from typing import Any


def risk_band(prob: float) -> str:
    """Map a (usually calibrated) probability onto low / medium / high.

    Edges are fixed product constants (not τ): low < 0.30 ≤ medium < 0.60 ≤ high.
    A non-finite probability raises instead of silently landing in "high".
    """
    if not math.isfinite(prob):
        raise ValueError(f"risk_band needs a finite probability, got {prob!r}")
    if prob < 0.30:
        return "low"
    if prob < 0.60:
        return "medium"
    return "high"


def hitl_action(prob: float, threshold: float, band: str) -> dict[str, Any]:
    """Map calibrated probability + risk band → recommended HITL action.

    Rules (best_f1_threshold from metrics.json):
      - below 0.5*threshold → monitor
      - below threshold → nurture / check-in
      - above threshold → retention outreach (human review)
      - high band → escalate (still HITL, no auto-cancel)

    ``auto_action`` is always ``none``.
    """
    if band == "high":
        action = "escalate"
        rationale = (
            "High risk band — escalate to human retention owner; "
            "no auto-cancel / no auto-email."
        )
    elif prob < 0.5 * threshold:
        action = "monitor"
        rationale = (
            f"Calibrated P(churn)={prob:.3f} < 0.5×threshold ({0.5 * threshold:.3f}); "
            "continue passive monitoring."
        )
    elif prob < threshold:
        action = "nurture / check-in"
        rationale = (
            f"Calibrated P(churn)={prob:.3f} below best-F1 threshold ({threshold:.3f}) "
            "but above half-threshold — light nurture / CS check-in."
        )
    else:
        action = "retention outreach (human review)"
        rationale = (
            f"Calibrated P(churn)={prob:.3f} ≥ best-F1 threshold ({threshold:.3f}); "
            "queue human retention outreach (HITL)."
        )
    return {
        "action": action,
        "rationale": rationale,
        "auto_action": "none",
        "hitl_required": True,
    }


class HitlDecisionPolicy:
    """``DecisionPolicy`` implementation used by packet + UI."""

    def decide(self, prob: float, threshold: float, band: str) -> dict[str, Any]:
        return hitl_action(prob, threshold, band)
