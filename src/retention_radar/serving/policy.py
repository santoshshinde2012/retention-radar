"""HITL decision policy and risk bands — single source of truth.

Serving CLI, packet builder, and Streamlit must call these helpers rather than
duplicating band cutoffs or auto-action rules.
"""

from __future__ import annotations

import math
from typing import Any

ESCALATE_PROB = 0.60
"""Calibrated probability at or above which a high-band account is escalated.

A fixed product constant (not learned). Everything else is derived from τ.
"""


def _check_inputs(prob: float, threshold: float) -> None:
    if not math.isfinite(prob):
        raise ValueError(f"risk_band needs a finite probability, got {prob!r}")
    if not (math.isfinite(threshold) and 0.0 < threshold < 1.0):
        raise ValueError(f"threshold τ must be in (0, 1), got {threshold!r}")


def risk_band(prob: float, threshold: float) -> str:
    """Map a calibrated probability onto low / medium / high, aligned with τ.

    Bands and actions share the same edges, so a band always implies its action:

    - low:    p < τ/2          → monitor
    - medium: τ/2 ≤ p < τ      → nurture / check-in
    - high:   p ≥ τ            → retention outreach (human review),
                                 or escalate when p ≥ ``ESCALATE_PROB``

    ``threshold`` is ``best_f1_threshold`` from metrics.json (chosen on validation).
    A non-finite probability raises instead of silently landing in "high".
    """
    _check_inputs(prob, threshold)
    if prob < 0.5 * threshold:
        return "low"
    if prob < threshold:
        return "medium"
    return "high"


def hitl_action(prob: float, threshold: float, band: str | None = None) -> dict[str, Any]:
    """Map calibrated probability (+ τ) → recommended HITL action.

    Rules (``threshold`` = best_f1_threshold from metrics.json):
      - p < 0.5×τ                 → monitor            (band low)
      - 0.5×τ ≤ p < τ             → nurture / check-in (band medium)
      - τ ≤ p < ESCALATE_PROB     → retention outreach (human review) (band high)
      - p ≥ ESCALATE_PROB         → escalate           (band high)

    ``band`` is optional; when given it must match ``risk_band(prob, threshold)``
    (a mismatch means a caller used stale edges, so it raises).
    ``auto_action`` is always ``none``.
    """
    expected = risk_band(prob, threshold)
    if band is not None and band != expected:
        raise ValueError(
            f"band {band!r} does not match risk_band({prob:.4f}, τ={threshold:.2f}) = {expected!r}"
        )
    half = 0.5 * threshold
    if prob >= ESCALATE_PROB:
        action = "escalate"
        rationale = (
            f"Calibrated P(churn)={prob:.3f} ≥ {ESCALATE_PROB:.2f}: escalate to the human "
            "retention owner; no auto-cancel / no auto-email."
        )
    elif expected == "high":
        action = "retention outreach (human review)"
        rationale = (
            f"Calibrated P(churn)={prob:.3f} ≥ best-F1 threshold ({threshold:.3f}); "
            "queue human retention outreach (HITL)."
        )
    elif expected == "medium":
        action = "nurture / check-in"
        rationale = (
            f"Calibrated P(churn)={prob:.3f} between 0.5×threshold ({half:.3f}) and the "
            f"best-F1 threshold ({threshold:.3f}): light nurture / CS check-in."
        )
    else:
        action = "monitor"
        rationale = (
            f"Calibrated P(churn)={prob:.3f} < 0.5×threshold ({half:.3f}); "
            "continue passive monitoring."
        )
    return {
        "action": action,
        "rationale": rationale,
        "auto_action": "none",
        "hitl_required": True,
    }


HOLD_ACTION = "hold: fix input data"


def validation_hold(errors: list[str]) -> dict[str, Any]:
    """HITL block for a record that failed validation: never scored, never queued."""
    return {
        "action": HOLD_ACTION,
        "rationale": "Input failed validation, so no score or outreach is recommended: "
        + "; ".join(errors),
        "auto_action": "none",
        "hitl_required": True,
        "blocked_by_validation": True,
    }


class HitlDecisionPolicy:
    """``DecisionPolicy`` implementation used by packet + UI."""

    def decide(self, prob: float, threshold: float, band: str | None = None) -> dict[str, Any]:
        return hitl_action(prob, threshold, band)
