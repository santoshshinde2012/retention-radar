# Worked examples: outcome (seed 42)

Two scoring-time records, seven days before renewal. No label: the renewal has not
happened. Walkthrough: [`../docs/case-study/renewal-worked-examples.md`](../docs/case-study/renewal-worked-examples.md).

```bash
python -m retention_radar.cli.single_record --user maya    # → artifacts/maya_decision_packet.json
python -m retention_radar.cli.single_record --user arjun
```

Sample packet: [`maya_decision_packet.sample.json`](maya_decision_packet.sample.json)

| | Maya | Arjun |
|--|------|-------|
| Plan · renewals paid | Pro · 3 | Pro+ · 12 |
| First renewal since the cap cut | yes | no |
| Cap hits (14d) · cheap-model share | 3 (87th pct) · 0.64 (95th pct) | 0 · 0.18 |
| Active days 7d / 28d · trend | 2 / 15 · 0.53 | 5 / 20 · 1.00 |
| Raw → calibrated P(lapse) | 0.554 → **0.153** | 0.066 → **0.023** |
| Band (edges 0.10 / 0.30) | medium | low |
| τ | 0.14 | 0.14 |
| Action | **limit_reset** (EV ≈ $1.83) | **no_action** |
| Other playbooks considered | cancel_flow_discount ($1.31), in_app_usage_tips ($0.59) | none (below τ) |
| Top SHAP drivers (log-odds) | `cheap_model_share_28d` +0.45, `first_renewal_after_pricing_change` +0.16, `cli_sessions_28d` −0.11, `ide_sessions_28d` −0.10 | `renewals_completed` −0.74, `agent_task_success_rate` −0.29, `accept_rate_change` −0.27 |
| Validation | OK | OK; `ide_sessions_28d`=41 flagged above training p99 (40) |
| `auto_action` | none | none |

Maya's raw 0.554 is not a probability: training reweights lapses with `scale_pos_weight`
(≈9.4), so raw scores run high. Platt calibration maps it to 0.153, which is about 1.6× the
9.6% base rate. The limit reset beats the discount because her trouble is the cap, and a
reset costs about $2 whether or not she would have renewed anyway. An armed discount costs
nothing until someone clicks cancel, but then some subscribers who would have renewed take
it too.
