# Worked examples: outcome (seed 42)

Two scoring-time records, seven days before renewal. No label: the renewal has not
happened. Walkthrough: [`../docs/case-study/renewal-worked-examples.md`](../docs/case-study/renewal-worked-examples.md).

```bash
python -m retention_radar.cli.single_record --user santosh    # → artifacts/santosh_decision_packet.json
python -m retention_radar.cli.single_record --user arjun
```

Sample output: [`santosh_decision_packet.sample.json`](santosh_decision_packet.sample.json)

| | Santosh | Arjun |
|--|------|-------|
| Plan · renewals paid | Pro · 3 | Pro+ · 12 |
| First renewal since the cap cut | yes | no |
| Cap hits (14d) · cheap-model share | 4 (90th pct) · 0.68 (96th pct) | 0 · 0.18 |
| Active days 7d / 28d · trend | 2 / 15 · 0.53 | 5 / 20 · 1.00 |
| Raw → calibrated P(lapse) | 0.717 → **0.288** | 0.124 → **0.025** |
| Band (edges 0.10 / 0.30) | medium | low |
| τ | 0.16 | 0.16 |
| Action | **holdout** (10% control group); would have sent **limit_reset** (EV ≈ $5.20) | **no_action** |
| Other playbooks considered | cancel_flow_discount ($2.79), in_app_usage_tips ($1.13) | none (below τ) |
| Holdout bucket | 8 of 100 (< 10, so held back) | not checked (below τ) |
| Top SHAP drivers (log-odds) | `limit_hits_14d` +0.63, `cheap_model_share_28d` +0.30, `first_renewal_after_pricing_change` +0.21, `renewals_completed` +0.09 | `renewals_completed` −0.53, `cheap_model_share_28d` −0.22, `accept_rate_change` −0.16, `agent_task_success_rate` −0.16 |
| Validation | OK | OK; `ide_sessions_28d`=41 flagged above training p99 (40) |
| `auto_action` | none | none |

Santosh's raw 0.717 is not a probability: training reweights lapses with `scale_pos_weight`
(≈9.4), so raw scores run high. Platt calibration maps it to 0.288, about three times the
9.6% base rate. The limit reset beats the discount because his trouble is the cap, and a
reset costs about $2 whether or not he would have renewed anyway. An armed discount costs
nothing until someone clicks cancel, but then some subscribers who would have renewed take
it too.

He does not get the reset. `sub_santosh` hashes into bucket 8 of 100 (sha256 of
`holdout:sub_santosh`, first 8 hex digits mod 100), which is inside the 10% holdout. The
policy sends nothing and logs `action=holdout` with `would_have_sent=limit_reset`. That row is
the point of the holdout: the limit reset's lift is measured by comparing subscribers who got
it with held-back subscribers like Santosh who would have got it. The decision is
deterministic, so he is held out on every run.
