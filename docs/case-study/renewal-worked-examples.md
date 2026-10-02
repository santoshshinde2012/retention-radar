# Two subscribers, seven days before renewal

A walkthrough of the serve path for the two worked examples:
**validate → score (raw + calibrated) → explain → compare to the cohort → decide**.

Reproduce: `./scripts/run_all.sh` writes `artifacts/maya_decision_packet.json` and
`artifacts/arjun_decision_packet.json`. Synthetic data, seed 42.

## Maya

Pro plan, $20/month, three renewals paid. Her fourth renewal is the first since the weekly
cap was cut. In the last 14 days she hit the cap four times (90th percentile). She now routes
68% of requests to the cheaper model (96th percentile of the renewal table). She was active on 15 of the last
28 days but only 2 of the last 7. CLI sessions (19) now outnumber IDE sessions (14).

| Step | Result |
|------|--------|
| Validate | 24 fields present, ranges and enums OK, `active_days_7d ≤ active_days_28d`, `engagement_trend` matches the active-days formula |
| Score | raw 0.717 → calibrated **0.288** (base rate 0.096) |
| Band | medium (edges 0.10 / 0.30) |
| Explain (SHAP, log-odds) | `limit_hits_14d` +0.63 · `cheap_model_share_28d` +0.30 · `first_renewal_after_pricing_change` +0.21 · `renewals_completed` +0.09 · `active_days_28d` −0.08 |
| Decide | above τ = 0.16 → eligible; not in the holdout; candidates `limit_reset` $5.20 · `cancel_flow_discount` $2.79 · `in_app_usage_tips` $1.13 → **limit_reset** |
| Executed by | the lifecycle tool, after a retention lead approved the playbook. `auto_action: none` |

What the drivers do and do not say: four cap hits and rationing to the cheap model are the
strongest pushes toward lapse in *this model's* view of her. That is not evidence that a limit reset will
change her decision. The expected value of $5.20 uses an **assumed** 25% effect; only the
holdout can say whether the reset works.

## Arjun

Pro+ plan, $60/month, twelve renewals paid. He was active on 20 of the last 28 days and 5 of
the last 7. He used 66% of his allowance with no cap hits. His agent tasks are kept 69% of the
time (82nd percentile).

| Step | Result |
|------|--------|
| Validate | OK. Outlier flag: `ide_sessions_28d` = 41, just above the training p99 (40). Scored anyway; the flag is information, not a hold |
| Score | raw 0.124 → calibrated **0.025** |
| Band | low |
| Explain | `renewals_completed` −0.53 · `cheap_model_share_28d` −0.22 · `accept_rate_change` −0.16 · `agent_task_success_rate` −0.16 |
| Decide | below τ → **no_action** |

Leaving Arjun alone is a decision, not an absence of one. Contacting renewers costs money
and can prompt a cancel. In one field experiment, proactive plan outreach raised 3-month
churn from 6% to 10% (Ascarza, Iyengar & Schleicher, JMR 2016).

## Packet shape

```json
{
  "validation": {"ok": true, "errors": [], "warnings": []},
  "scoring": {"churn_probability_raw": 0.7174, "churn_probability_calibrated": 0.2880,
              "risk_band": "medium", "best_f1_threshold": 0.16},
  "explanation": {"top_features": [{"feature": "limit_hits_14d", "contribution": 0.633}]},
  "cohort_compare": {"limit_hits_14d": {"value": 4.0, "percentile": 90.4}},
  "decision": {"action": "limit_reset", "holdout": false, "expected_value_usd": 5.2,
               "candidates": [{"playbook": "limit_reset", "expected_value_usd": 5.2}],
               "auto_action": "none", "hitl_required": false}
}
```

Full sample: [`results/maya_decision_packet.sample.json`](../../results/maya_decision_packet.sample.json).

## Six more scenarios

[`data/use_cases/`](../../data/use_cases/README.md) holds real test-split subscribers for
the other paths through the policy:
- a first renewal that is already fading (discount armed in the cancel flow)
- a side-project user between projects (pause offer)
- an overage shock whose email the messaging tool suppressed
- a quiet week on a long tenure (no action)
- a holdout
- the one Ultra subscriber who gets a person-written email
