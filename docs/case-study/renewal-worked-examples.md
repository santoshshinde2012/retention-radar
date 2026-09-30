# Two subscribers, seven days before renewal

A walkthrough of the serve path for the two worked examples:
**validate → score (raw + calibrated) → explain → compare to the cohort → decide**.

Reproduce: `./scripts/run_all.sh` writes `artifacts/maya_decision_packet.json` and
`artifacts/arjun_decision_packet.json`. Synthetic data, seed 42.

## Maya

Pro plan, $20/month, three renewals paid. Her fourth renewal is the first since the weekly
cap was cut. In the last 14 days she hit the cap three times. She now routes 64% of requests
to the cheaper model (95th percentile of the renewal table). She was active on 15 of the last
28 days but only 2 of the last 7. CLI sessions (19) now outnumber IDE sessions (14).

| Step | Result |
|------|--------|
| Validate | 24 fields present, ranges and enums OK, `active_days_7d ≤ active_days_28d`, `engagement_trend` matches the active-days formula |
| Score | raw 0.554 → calibrated **0.153** (base rate 0.096) |
| Band | medium (edges 0.10 / 0.30) |
| Explain (SHAP, log-odds) | `cheap_model_share_28d` +0.45 · `first_renewal_after_pricing_change` +0.16 · `cli_sessions_28d` −0.11 · `ide_sessions_28d` −0.10 · `accept_rate_change` −0.08 |
| Decide | above τ = 0.14 → eligible; not in the holdout; candidates `limit_reset` $1.83 · `cancel_flow_discount` $1.31 · `in_app_usage_tips` $0.59 → **limit_reset** |
| Executed by | the lifecycle tool, after a retention lead approved the playbook. `auto_action: none` |

What the drivers do and do not say: rationing to the cheap model is the strongest push
toward lapse in *this model's* view of her. That is not evidence that a limit reset will
change her decision. The expected value of $1.83 uses an **assumed** 25% effect; only the
holdout can say whether the reset works.

## Arjun

Pro+ plan, $60/month, twelve renewals paid. He was active on 20 of the last 28 days and 5 of
the last 7. He used 66% of his allowance with no cap hits. His agent tasks are kept 69% of the
time (82nd percentile).

| Step | Result |
|------|--------|
| Validate | OK. Outlier flag: `ide_sessions_28d` = 41, just above the training p99 (40). Scored anyway; the flag is information, not a hold |
| Score | raw 0.066 → calibrated **0.023** |
| Band | low |
| Explain | `renewals_completed` −0.74 · `agent_task_success_rate` −0.29 · `accept_rate_change` −0.27 · `cheap_model_share_28d` −0.26 |
| Decide | below τ → **no_action** |

Leaving Arjun alone is a decision, not an absence of one. Contacting renewers costs money
and can prompt a cancel. In one field experiment, proactive plan outreach raised 3-month
churn from 6% to 10% (Ascarza, Iyengar & Schleicher, JMR 2016).

## Packet shape

```json
{
  "validation": {"ok": true, "errors": [], "warnings": []},
  "scoring": {"churn_probability_raw": 0.5535, "churn_probability_calibrated": 0.1532,
              "risk_band": "medium", "best_f1_threshold": 0.14},
  "explanation": {"top_features": [{"feature": "cheap_model_share_28d", "contribution": 0.453}]},
  "cohort_compare": {"limit_hits_14d": {"value": 3.0, "percentile": 87.6}},
  "decision": {"action": "limit_reset", "holdout": false, "expected_value_usd": 1.83,
               "candidates": [{"playbook": "limit_reset", "expected_value_usd": 1.83}],
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
