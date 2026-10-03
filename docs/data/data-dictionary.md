# Data dictionary

Auto-generated from `src/retention_radar/config.py`, `configs/schemas/user_record.schema.json`, and `src/retention_radar/data/generate.py`.
One row per paying subscriber of a monthly AI coding assistant plan, snapshotted seven days before renewal (T-7). Synthetic, no real PII.

## Label and routing

- `churned` = 1 when the subscriber **voluntarily** let the plan lapse at this renewal.
- Renewals that lapsed because the card failed and retries ran out are routed to **dunning** and excluded (a payments problem, not a behaviour signal).
- Subscribers who had already scheduled a cancel before T-7 are routed to the **cancel flow** and excluded (the outcome is decided; keeping them would leak it).
- `data/raw/renewals_all.csv` keeps every renewal with `outcome` and `route` for audit.

_Generated: 2026-09-30 · seed=42_

## Missingness policy

- **Synthetic generator:** every feature is populated; **no NaNs by design** (see `src/retention_radar/data/generate.py`).
- **Train / serve:** `prepare_xy` and `row_to_feature_frame` **fail loud** if any model feature is NaN after encoding. There is no silent impute.
- **Recommended real-data pattern:** add missingness indicators (e.g. `accept_rate_missing` for users who turned inline suggestions off) and fit an imputer **on train only**, persist it beside the model, then apply the same transform at serve. Do not impute from a single Streamlit row.

## Columns

| Column | Role | Dtype | Range | Nullable | Description |
|--------|------|-------|-------|----------|-------------|
| `user_id` | id | `string` | minLength=1 | no (required on serve) | Stable synthetic subscriber id (not a feature). |
| `user_name` | id | `string` | minLength=1 | no (required on serve) | Display name for the worked examples (not a feature). |
| `plan_tier` | feature | `string` | enum: pro, pro_plus, ultra | no (required on serve) | Monthly plan: pro ($20) | pro_plus ($60) | ultra ($200). |
| `renewals_completed` | feature | `number` | [0, 60] | no (required on serve) | Monthly renewals already paid. 0 = this is the first renewal (the cliff). |
| `active_days_7d` | feature | `number` | [0, 7] | no (required on serve) | Days with any coding activity in the 7 days before T-7 (0-7). |
| `active_days_28d` | feature | `number` | [0, 28] | no (required on serve) | Days with any coding activity in the 28 days before T-7 (0-28). |
| `engagement_trend` | feature | `number` | [0.0, 4.0] | no (required on serve) | active_days_7d / max(1, active_days_28d / 4): ~1 steady, <1 fading. |
| `last_active_days_ago` | feature | `number` | [0, 90] | no (required on serve) | Days since the last coding activity, measured at T-7. |
| `agent_requests_28d` | feature | `number` | [0, 50000] | no (required on serve) | Agent / chat requests sent to frontier models in 28 days. |
| `allowance_used_pct` | feature | `number` | [0.0, 3.0] | no (required on serve) | Share of the plan's included usage consumed in 28 days (can exceed 1 with overage). |
| `limit_hits_14d` | feature | `number` | [0, 60] | no (required on serve) | Times a 5-hour or weekly usage cap blocked a request in 14 days. |
| `cheap_model_share_28d` | feature | `number` | [0.0, 1.0] | no (required on serve) | Share of requests routed to a cheaper / auto model (rationing signal). |
| `overage_usd_28d` | feature | `number` | [0.0, 5000.0] | no (required on serve) | Usage billed above the plan in 28 days (USD). |
| `overage_toggled_off` | feature | `number` | enum: 0, 1 | no (required on serve) | 1 if paid overage was switched off or capped after being on. |
| `suggestion_accept_rate_28d` | feature | `number` | [0.0, 1.0] | no (required on serve) | Accepted / shown inline suggestions in 28 days. |
| `accept_rate_change` | feature | `number` | [0.0, 3.0] | no (required on serve) | Accept rate this 28 days / previous 28 days (1 = unchanged). |
| `agent_task_success_rate` | feature | `number` | [0.0, 1.0] | no (required on serve) | Agent tasks that ended with the change kept (not reverted) in 28 days. |
| `failed_requests_rate` | feature | `number` | [0.0, 1.0] | no (required on serve) | Share of requests that errored or timed out in 28 days. |
| `incident_exposed_28d` | feature | `number` | enum: 0, 1 | no (required on serve) | 1 if the subscriber had requests during a declared incident window. |
| `support_tickets_90d` | feature | `number` | [0, 50] | no (required on serve) | Support tickets opened in 90 days. |
| `ide_sessions_28d` | feature | `number` | [0, 500] | no (required on serve) | IDE-extension sessions in 28 days. |
| `cli_sessions_28d` | feature | `number` | [0, 500] | no (required on serve) | CLI-agent sessions in 28 days. |
| `weekend_usage_ratio` | feature | `number` | [0.0, 1.0] | no (required on serve) | Share of activity on weekends (side-project signal). |
| `first_renewal_after_pricing_change` | feature | `number` | enum: 0, 1 | no (required on serve) | 1 if this is the subscriber's first renewal since the last limit / pricing change. |
| `plan_tier_code` | feature | `integer` | 0–2 (pro, pro_plus, ultra) | no (derived at transform) | Ordinal encoding of plan_tier (pro=0, pro_plus=1, ultra=2). |
| `churned` | target | `integer (0/1)` | 0 or 1 | no in training CSV; omitted on serve (renewal is ahead) | Label: 1 = voluntarily let the plan lapse at this renewal (training only). |

## Model feature vector (encoded)

Order used at train / infer time:

1. `plan_tier_code`
2. `renewals_completed`
3. `active_days_7d`
4. `active_days_28d`
5. `engagement_trend`
6. `last_active_days_ago`
7. `agent_requests_28d`
8. `allowance_used_pct`
9. `limit_hits_14d`
10. `cheap_model_share_28d`
11. `overage_usd_28d`
12. `overage_toggled_off`
13. `suggestion_accept_rate_28d`
14. `accept_rate_change`
15. `agent_task_success_rate`
16. `failed_requests_rate`
17. `incident_exposed_28d`
18. `support_tickets_90d`
19. `ide_sessions_28d`
20. `cli_sessions_28d`
21. `weekend_usage_ratio`
22. `first_renewal_after_pricing_change`

## Plan tier encoding

| Tier | Code |
|------|------|
| `pro` | 0 |
| `pro_plus` | 1 |
| `ultra` | 2 |

## Worked examples (scoring-time records, no label)

| Field | santosh | arjun |
|-------|---|---|
| `user_id` | sub_santosh | sub_arjun |
| `user_name` | Santosh (worked example) | Arjun (worked example) |
| `plan_tier` | pro | pro_plus |
| `renewals_completed` | 3 | 12 |
| `active_days_7d` | 2 | 5 |
| `active_days_28d` | 15 | 20 |
| `engagement_trend` | 0.5333 | 1.0 |
| `last_active_days_ago` | 2 | 0 |
| `agent_requests_28d` | 470 | 910 |
| `allowance_used_pct` | 1.02 | 0.66 |
| `limit_hits_14d` | 4 | 0 |
| `cheap_model_share_28d` | 0.68 | 0.18 |
| `overage_usd_28d` | 0.0 | 0.0 |
| `overage_toggled_off` | 0 | 0 |
| `suggestion_accept_rate_28d` | 0.29 | 0.33 |
| `accept_rate_change` | 0.96 | 1.03 |
| `agent_task_success_rate` | 0.58 | 0.69 |
| `failed_requests_rate` | 0.05 | 0.03 |
| `incident_exposed_28d` | 1 | 1 |
| `support_tickets_90d` | 0 | 1 |
| `ide_sessions_28d` | 14 | 41 |
| `cli_sessions_28d` | 19 | 12 |
| `weekend_usage_ratio` | 0.21 | 0.12 |
| `first_renewal_after_pricing_change` | 1 | 0 |

Payloads: `data/raw/subscribers/*.json`. Walkthrough: [renewal-worked-examples.md](../case-study/renewal-worked-examples.md).


## Related reading

- [architecture.md](../architecture.md) — SOLID package map
- [model-card.md](../model-card.md)
- [Worked examples](../case-study/renewal-worked-examples.md)
- [Single-record checklist](../case-study/single-record-checklist.md)
- [Data foundation / lakehouse](data-foundation-lakehouse.md)
