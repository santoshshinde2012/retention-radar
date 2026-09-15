# Data dictionary

Auto-generated from `src/retention_radar/config.py`, `configs/schemas/user_record.schema.json`, and `src/retention_radar/data/generate.py`.
Synthetic AI-platform churn dataset — no real PII.

_Generated: 2026-09-15 · seed=42_

## Missingness policy

- **Synthetic generator:** every feature is populated; **no NaNs by design** (see `src/retention_radar/data/generate.py`).
- **Train / serve:** `prepare_xy` and `row_to_feature_frame` **fail loud** if any model feature is NaN after encoding. There is no silent impute.
- **Recommended real-data pattern:** add missingness indicators (e.g. `nps_missing`) and fit an imputer **on train only**, persist it beside the model, then apply the same transform at serve. Do not impute from a single Streamlit row.

## Columns

| Column | Role | Dtype | Range | Nullable | Description |
|--------|------|-------|-------|----------|-------------|
| `user_id` | id | `string` | minLength=1 | no (required on serve) | Stable synthetic user identifier. |
| `user_name` | id | `string` | minLength=1 | no (required on serve) | Display name (synthetic; Santosh is the hero profile). |
| `days_since_signup` | feature | `number` | [1, 2000] | no (required on serve) | Days since account creation. |
| `sessions_last_7d` | feature | `number` | [0, 60] | no (required on serve) | Product sessions in the last 7 days. |
| `sessions_last_30d` | feature | `number` | [0, 200] | no (required on serve) | Product sessions in the last 30 days. |
| `avg_session_minutes` | feature | `number` | [0.5, 240] | no (required on serve) | Average session length in minutes. |
| `models_used_count` | feature | `number` | [0, 30] | no (required on serve) | Distinct AI models the user has invoked. |
| `api_calls_last_30d` | feature | `number` | [0, 100000] | no (required on serve) | API calls in the last 30 days. |
| `tokens_consumed_last_30d` | feature | `number` | [0, 50000000] | no (required on serve) | Token usage in the last 30 days. |
| `tools_used_count` | feature | `number` | [0, 40] | no (required on serve) | Distinct tools / integrations used. |
| `failed_requests_rate` | feature | `number` | [0, 1] | no (required on serve) | Fraction of failed requests (0–1). |
| `support_tickets_last_90d` | feature | `number` | [0, 50] | no (required on serve) | Support tickets opened in last 90 days. |
| `plan_tier` | feature | `string` | enum: free, starter, pro, enterprise | no (required on serve) | Subscription tier: free | starter | pro | enterprise. |
| `payment_failures_last_90d` | feature | `number` | [0, 20] | no (required on serve) | Failed payment attempts in last 90 days. |
| `feature_adoption_score` | feature | `number` | [0, 1] | no (required on serve) | 0–1 score of how broadly features are used. |
| `nps_score` | feature | `number` | [0, 10] | no (required on serve) | Net Promoter Score style rating (0–10). |
| `last_active_days_ago` | feature | `number` | [0, 365] | no (required on serve) | Days since last observed activity. |
| `weekend_usage_ratio` | feature | `number` | [0, 1] | no (required on serve) | Share of usage that happens on weekends (0–1). |
| `engagement_trend` | feature | `number` | [0, 5] | no (required on serve) | sessions_last_7d / max(1, sessions_last_30d/4); ≈1 stable, <1 cooling, >1 accelerating. |
| `spend_usd_last_30d` | feature | `number` | [0, 100000] | no (required on serve) | Synthetic monthly spend in USD (last 30 days). |
| `days_until_renewal` | feature | `number` | [0, 730] | no (required on serve) | Days until next billing / renewal (B2B-ish). |
| `agent_runs_last_30d` | feature | `number` | [0, 5000] | no (required on serve) | Agent / automation runs in the last 30 days (AI-native). |
| `ide_plugin_sessions_last_30d` | feature | `number` | [0, 500] | no (required on serve) | IDE plugin sessions in the last 30 days. |
| `seat_utilization` | feature | `number` | [0, 1] | no (required on serve) | 0–1 seats used vs seats provisioned (team signal). |
| `plan_tier_code` | feature | `integer` | 0–3 (free…enterprise) | no (derived at transform) | Ordinal encoding of plan_tier (free=0 … enterprise=3). |
| `churned` | target | `integer (0/1)` | 0 or 1 | no in training CSV; omitted on serve | Binary label: 1 = churned, 0 = retained (training only). |

## Model feature vector (encoded)

Order used at train / infer time:

1. `days_since_signup`
2. `sessions_last_7d`
3. `sessions_last_30d`
4. `avg_session_minutes`
5. `models_used_count`
6. `api_calls_last_30d`
7. `tokens_consumed_last_30d`
8. `tools_used_count`
9. `failed_requests_rate`
10. `support_tickets_last_90d`
11. `plan_tier_code`
12. `payment_failures_last_90d`
13. `feature_adoption_score`
14. `nps_score`
15. `last_active_days_ago`
16. `weekend_usage_ratio`
17. `engagement_trend`
18. `spend_usd_last_30d`
19. `days_until_renewal`
20. `agent_runs_last_30d`
21. `ide_plugin_sessions_last_30d`
22. `seat_utilization`

## Plan tier encoding

| Tier | Code |
|------|------|
| `free` | 0 |
| `starter` | 1 |
| `pro` | 2 |
| `enterprise` | 3 |

## Santosh Shinde — feature contract (inference)

Payload: `data/raw/santosh_shinde.json` (no `churned`).

| Field | Example value |
|-------|----------------|
| `user_id` | santosh_shinde |
| `user_name` | Santosh Shinde |
| `days_since_signup` | 420 |
| `sessions_last_7d` | 9 |
| `sessions_last_30d` | 38 |
| `avg_session_minutes` | 28.5 |
| `models_used_count` | 7 |
| `api_calls_last_30d` | 1850 |
| `tokens_consumed_last_30d` | 420000 |
| `tools_used_count` | 8 |
| `failed_requests_rate` | 0.12 |
| `support_tickets_last_90d` | 2 |
| `plan_tier` | pro |
| `payment_failures_last_90d` | 1 |
| `feature_adoption_score` | 0.78 |
| `nps_score` | 7.0 |
| `last_active_days_ago` | 8 |
| `weekend_usage_ratio` | 0.22 |
| `engagement_trend` | 0.9474 |
| `spend_usd_last_30d` | 189.0 |
| `days_until_renewal` | 21 |
| `agent_runs_last_30d` | 52 |
| `ide_plugin_sessions_last_30d` | 28 |
| `seat_utilization` | 0.72 |

See also [santosh-case-study.md](../case-study/santosh-case-study.md) and [single-record-checklist.md](../case-study/single-record-checklist.md).


## Related reading

- [ARCHITECTURE.md](../ARCHITECTURE.md) — SOLID package map
- [MODEL_CARD.md](../MODEL_CARD.md)
- [Santosh case study](../case-study/santosh-case-study.md)
- [Single-record checklist](../case-study/single-record-checklist.md)
- [Data foundation / lakehouse](data-foundation-lakehouse.md)
