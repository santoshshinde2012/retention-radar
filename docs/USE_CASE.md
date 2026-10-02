# The use case: renewals of a monthly AI coding assistant plan

This page explains what the repo models, why, and what the synthetic data can and cannot
show. Every market fact below has a dated source; the product itself is fictional.

## The product

A self-serve AI coding assistant: an IDE extension plus a CLI agent, billed monthly by card.

| Plan | Price | Included usage |
|------|-------|----------------|
| Pro | $20/mo | Unlimited tab completion; a monthly allowance of frontier-model requests; a 5-hour rolling window and a weekly cap on agent use |
| Pro+ | $60/mo | ~3× Pro |
| Ultra | $200/mo | ~20× Pro |

Teams plans exist but are **out of scope**: their risk is seat contraction, and they are
the one segment where a person reviewing an account pays for itself. A Teams record sent
to this service is held by validation (`plan_tier` enum).

## The question

> Seven days before a subscriber's monthly renewal, how likely are they to let the plan
> lapse on purpose, and which approved action, if any, is worth taking?

T-7 is chosen so an email, an in-app message or an armed cancel-flow offer has time to
land before the charge.

## Why this is a real problem

| When | What happened | Source |
|------|---------------|--------|
| Jun 16 → Jul 4, 2025 | Cursor Pro moved from 500 fast requests to $20 of usage at API rates; the CEO apologised and unexpected charges were refunded | https://cursor.com/blog/june-2025-pricing · https://techcrunch.com/2025/07/07/cursor-apologizes-for-unclear-pricing-changes-that-upset-users/ |
| Jun 18, 2025 | GitHub Copilot started enforcing monthly premium-request allowances | https://github.blog/changelog/2025-06-18-update-to-github-copilot-consumptive-billing-experience/ |
| Jul 28 → Aug 28, 2025 | Anthropic added weekly limits to Claude Pro / Max for Claude Code | https://venturebeat.com/ai/anthropic-throttles-claude-rate-limits-devs-call-foul · https://news.ycombinator.com/item?id=44713757 |
| Sep 2025 | Replit Agent 3 users reported surprise bills (effort-based pricing) | https://www.theregister.com/software/2025/09/18/replit-infuriating-customers-with-surprise-cost-overruns/1006671 |
| Sep 17, 2025 | Anthropic postmortem: ~30% of Claude Code users had at least one misrouted request during three infrastructure bugs | https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues |
| Apr 23, 2026 | Second quality postmortem; usage limits reset for all subscribers as the remedy | https://www.anthropic.com/engineering/april-23-postmortem |
| Jun 1, 2026 | Copilot moved to token-based credits and retired annual plans | https://github.blog/news-insights/company-news/github-copilot-is-moving-to-usage-based-billing/ |
| Sep 2026 | Claude Code weekly limits cut 17% from the summer level | https://www.bleepingcomputer.com/news/artificial-intelligence/anthropic-is-cutting-claude-codes-current-weekly-limits-by-17-percent/ |

Switching is gradual, not a slammed door:

- 70% of developers use 2–4 AI tools (Pragmatic Engineer, n=906, Feb 2026): https://newsletter.pragmaticengineer.com/p/ai-tooling-2026
- Work adoption moved within six months of 2026: Copilot 29%→21%, Cursor 18%→12%, Claude Code 18%→39%, Codex 3%→16% (JetBrains, Jan vs May–Jul 2026): https://blog.jetbrains.com/research/2026/08/ai-coding-agent-adoption-2026/
- 46% of developers distrust AI output accuracy; 66% cite "almost right, but not quite" (Stack Overflow 2025): https://survey.stackoverflow.co/2025/ai

The scale of churn at this price:

- AI-native products under $50/month: **23%** gross revenue retention over a year (ChartMogul, Dec 2025): https://chartmogul.com/reports/saas-retention-the-ai-churn-wave/. That is roughly 11–12% a month.
- Monthly churn by tenure (Churnkey, Stripe 2024 data): 12.0% under 3 months, 7.4% at 3–6 months, 3.2% at 12+: https://churnkey.co/blog/voluntary-churn-benchmarks/
- 25% of lapsed subscriptions are purely payment failures (Stripe): https://stripe.com/blog/how-we-built-it-smart-retries

## How the pieces map to that record

| Real mechanism | Where it lives in this repo |
|----------------|-----------------------------|
| Caps bite mid-task | `limit_hits_14d`, `allowance_used_pct` |
| Rationing after a cap | `cheap_model_share_28d` |
| Surprise overage bill | `overage_usd_28d`, `overage_toggled_off` |
| Grandfathered until renewal, then the new deal | `first_renewal_after_pricing_change` |
| Quality incidents | `incident_exposed_28d`, `accept_rate_change`, `agent_task_success_rate` |
| A rival tool takes the work | unobserved in the data; footprint in `active_days_7d` vs `active_days_28d`, `ide_sessions_28d` |
| First renewals are the cliff | `renewals_completed` |
| Side projects go quiet between projects | `weekend_usage_ratio`; the pause-offer playbook |
| Failed cards | routed to dunning, excluded from the label |
| Cancel already scheduled | routed to the cancel flow, excluded (the outcome is decided) |

## Who acts, and how

Gainsight's tech-touch guidance puts accounts under roughly $10–15K ARR on automated
email, in-app guidance and self-serve content, with a person only when a signal calls for
one (https://www.gainsight.com/blog/tech-touch-customer-success/). A $240/year subscriber is
far below that line. So:

- A retention lead approves a small set of **playbooks** (`config.PLAYBOOKS`).
- The daily T-7 batch picks the playbook with the highest **expected value** for each
  subscriber above τ, or nothing.
- A deterministic **10% holdout** of eligible subscribers gets nothing, so each playbook's
  lift can be measured.
- Only **Ultra** has a playbook with a person in it (a written email), because only at
  $200/month does fifteen minutes of staff time pay back.

## Why a holdout is not optional

- Targeting by risk is not targeting by response. In field data, targeting by uplift beat
  targeting by churn risk (Ascarza, JMR 2018): https://journals.sagepub.com/doi/10.1509/jmr.16.0163
- Proactive "better plan" outreach raised 3-month churn from 6% to 10% (Ascarza, Iyengar,
  Schleicher, JMR 2016): https://journals.sagepub.com/doi/abs/10.1509/jmr.13.0483
- "Properly randomised control groups … provide the only proven and reliable way of
  assessing the true impact of retention activity" (Radcliffe & Simpson, 2007):
  https://stochasticsolutions.com/pdf/SavedAndDrivenAway.pdf

## What the synthetic data can and cannot show

- **Can:** a correct label pipeline, a T-7 feature contract, an honest model ladder,
  calibration, an expected-value policy, a holdout design, and how wide a lift interval is
  at a given size.
- **Cannot:** real effect sizes. The generator's label comes from unobserved causes plus
  the observed footprints above, with noise; the ladder is a result about that generator.
  Playbook effects in `config.PLAYBOOKS` are assumptions, and the use-case pack simulates
  outcomes with exactly those assumptions. Nothing here measures whether a limit reset
  saves anyone.
