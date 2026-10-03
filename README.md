# Retention Radar

Renewal-risk scoring for a monthly AI coding assistant plan (Pro $20 · Pro+ $60 · Ultra $200).

Seven days before each renewal, every paying subscriber is scored for the chance they
voluntarily let the plan lapse. The score picks one human-approved playbook (a limit
reset, a pause offer, a cancel-flow discount, or for Ultra a personal email), keeps a 10%
holdout so the playbook's lift can be measured, or does nothing. The service suggests;
it never sends (`auto_action: none`).

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](runtime.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repo:** [santoshshinde2012/retention-radar](https://github.com/santoshshinde2012/retention-radar)

---

## Why this use case

AI coding assistants spent 2025–26 changing the deal under their subscribers: request
caps became usage-based pricing, weekly limits arrived and were cut again, quality
incidents came and went, and most developers ended up paying for two or three tools. In
ChartMogul's December 2025 retention report, AI-native products under $50/month kept
**23%** of their revenue over a year. At $20/month nobody reads a subscriber's profile,
so the score has to drive automated, measurable actions. That is what this repo builds.

The data is synthetic (seed **42**, no real PII), but the mechanisms behind it are the
ones in the public record: cap hits, rationing to a cheaper model, a surprise overage
bill, the first renewal after a pricing change, a rival tool taking the work. Sources:
[docs/use-case.md](docs/use-case.md).

## What this is

| This project | Not this project |
|--------------|------------------|
| Train → evaluate → serve a T-7 renewal score | A CRM, billing system or messaging tool |
| Label from billing: voluntary lapse only; failed cards → dunning; scheduled cancels → cancel flow | "Inactive for 30 days = churned" |
| Expected-value policy + deterministic 10% holdout | A claim that any playbook works |
| Synthetic renewals (seed 42), no real PII | ROI or fairness claims |

**Worked examples (seed 42):**

| Subscriber | Raw → calibrated P(lapse) | Band | Action |
|------------|---------------------------|------|--------|
| Santosh: Pro, 4 cap hits in 14 days, 68% of requests on the cheap model, first renewal since the cap cut | 0.717 → **0.288** | medium | `holdout`: in the 10% control group, so nothing is sent; would have got `limit_reset` (EV ≈ $5.20) |
| Arjun: Pro+, 12 renewals, 66% of allowance used, no cap hits | 0.124 → **0.025** | low | `no_action` |

**Honest ladder (test AUC, 7,329 T-7 rows, 9.6% base rate):** LogReg **0.781** · CatBoost
**0.765** · RF **0.758** · Optuna XGB **0.757** · LightGBM **0.736** · default XGB **0.729** ·
Dummy 0.500. Calibrated XGBoost serves for now; on this result a real deployment should swap in the linear model (see BENCHMARKS). Full tables:
[`models/metrics.json`](models/metrics.json) · [results/benchmarks.md](results/benchmarks.md).

---

## Related repos

| Repo | Role |
|------|------|
| **This repo** | Public code, benchmarks and results |
| [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) | Data foundation: bronze billing + usage events → T-7 gold features |

---

## Requirements

- Python **3.12+**. The committed bundle was trained with XGBoost **3.4.1** and scikit-learn
  **1.9.1**; `requirements.txt` pins the model-affecting libraries so `make run` re-creates
  every non-latency value in `models/metrics.json` exactly on Linux x86-64, where the committed
  bundle was trained (CI checks this). On other CPUs, such as Apple Silicon, a retrain differs from the
  third decimal. Snapshot: [`requirements.lock`](requirements.lock).
- CPU only. Linux, macOS, or Windows (WSL).
- **macOS:** `brew install libomp` if XGBoost or LightGBM fail to load.

## Install

```bash
git clone https://github.com/santoshshinde2012/retention-radar.git
cd retention-radar
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Or: `make setup`

## Quick start

```bash
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

| Step | Command / file |
|------|----------------|
| Cite metrics | [`models/metrics.json`](models/metrics.json) |
| Read benchmarks | [results/benchmarks.md](results/benchmarks.md) |
| Score Santosh | `make infer` → `artifacts/santosh_decision_packet.json` |
| Walk one renewal day | `make use-cases`: [data/use_cases/](data/use_cases/README.md) daily T-7 batch → action queue (+ rejects) → packets → held records → send export → renewal outcomes + lift vs holdout |
| Daily batch | `python -m retention_radar.cli.batch_score --csv data/raw/renewals_t7.csv` → ranked queue + `*_rejected.csv` |
| Lift vs holdout | after `make use-cases`: `python -m retention_radar.cli.outcomes --log artifacts/use_cases/action_log.csv --labels data/use_cases/renewal_outcomes.csv` |
| Thin local API | `uvicorn retention_radar.serving.api:app --app-dir src` → `POST /v1/churn/score`, `/v1/churn/batch`, `/v1/churn/actions` |
| Open UI | `make ui` (committed models only, no fit on load) |
| Run tests | `make test` |
| Verify everything locally | `make e2e-local` |

Faster smoke (committed `models/` untouched):

```bash
RETENTION_RADAR_ARTIFACT_DIR=artifacts/smoke N_USERS=800 N_OPTUNA_TRIALS=5 CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
```

### Make targets

| Target | What it does |
|--------|----------------|
| `make setup` | Create venv and install |
| `make run` | Synthetic full pipeline |
| `make test` | Pytest (synthetic) |
| `make infer` | Santosh's decision packet |
| `make ui` | Streamlit UI |
| `make api` | Thin local FastAPI (teaching; no auth) |
| `make run-lakehouse` | Lakehouse E2E (needs a local-data-lakehouse checkout) |
| `make use-cases` | One renewal day on `data/use_cases/` |
| `make reproduce` | Retrain into `artifacts/repro/` and diff against `models/metrics.json` |
| `make e2e-local` | Everything: reproduce, tests, CLI + live API + live Streamlit, lakehouse (if cloned beside), isolation check |
| `make lint` | Ruff on `src/ tests/ app/` |

---

## The data contract

One row per paying subscriber at **T-7** (seven days before a monthly renewal): 22 serve
fields plus `user_id` / `user_name`. Every field must exist in the live record at score
time; anything knowable only after the renewal stays out.

| Family | Fields |
|--------|--------|
| Plan and tenure | `plan_tier`, `renewals_completed`, `first_renewal_after_pricing_change` |
| Habit | `active_days_7d`, `active_days_28d`, `engagement_trend`, `last_active_days_ago`, `weekend_usage_ratio` |
| Against the cap | `agent_requests_28d`, `allowance_used_pct`, `limit_hits_14d`, `cheap_model_share_28d`, `overage_usd_28d`, `overage_toggled_off` |
| Quality | `suggestion_accept_rate_28d`, `accept_rate_change`, `agent_task_success_rate`, `failed_requests_rate`, `incident_exposed_28d`, `support_tickets_90d` |
| Surface | `ide_sessions_28d`, `cli_sessions_28d` |

Label `churned` = voluntary lapse at this renewal. Renewals lost to a failed card
(dunning) and subscribers who had already scheduled a cancel before T-7 are excluded;
`data/raw/renewals_all.csv` keeps them with `outcome` and `route` for audit. Full
dictionary: [docs/data/data-dictionary.md](docs/data/data-dictionary.md).

## Data paths

| Path | How | Notes |
|------|-----|-------|
| **Synthetic** (CI / published numbers) | `CHURN_DATA_SOURCE=synthetic` | Seed-42 renewal cohort; committed `models/` |
| **Lakehouse gold** | `./scripts/run_lakehouse_e2e.sh /path/to/local-data-lakehouse` | Writes under `artifacts/lakehouse_run/` only |
| **Sync only** | `./scripts/sync_lakehouse_exports.sh …/data/export` | No retrain |

Details: [docs/data/data-foundation-lakehouse.md](docs/data/data-foundation-lakehouse.md)

---

## Project structure

```text
retention-radar/
├── src/retention_radar/     # Package + CLI
├── app/                     # Streamlit UI
├── scripts/                 # run_all, use cases, lakehouse sync, local E2E
├── configs/                 # Record schema + action-log schema/template
├── data/                    # raw (renewals + worked examples) · use_cases · external
├── models/                  # Seed-42 serve bundle + metrics
├── results/                 # Benchmarks, worked examples, lakehouse consume run, plots
├── artifacts/               # Runtime output (gitignored)
├── docs/                    # Use case, guides, architecture, model card
└── tests/
```

## Docs

| Doc | Purpose |
|-----|---------|
| [docs/use-case.md](docs/use-case.md) | The use case, the evidence behind it, and what the synthetic data can and cannot show |
| [results/benchmarks.md](results/benchmarks.md) | Ladder, calibration, operating point, policy, latency |
| [results/worked-examples.md](results/worked-examples.md) | Santosh and Arjun, end to end |
| [results/lakehouse-consume-e2e.md](results/lakehouse-consume-e2e.md) | Radar main scoring the local-data-lakehouse export: 7,387 rows, actions, tests, CI |
| [docs/model-card.md](docs/model-card.md) | Intended use + metrics |
| [docs/guides/start-here.md](docs/guides/start-here.md) | Clone → one command → what to read |
| [docs/architecture.md](docs/architecture.md) | Train ≠ serve design |
| [docs/guides/algorithm-landscape.md](docs/guides/algorithm-landscape.md) | What we use vs defer |

## License

MIT © Santosh Shinde, see [LICENSE](LICENSE).
