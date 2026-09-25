# Single-Record Completeness Checklist

Exhaustive E2E checklist for one JSON user record through this project.
Marks reflect **this repo after the AI-churn feature expansion + `single_record` path** (seed 42),
plus FOSS-friendly CI / batch / drift / slice gaps closed.

Legend: ✅ done in this project · ⬜ not in scope / deferred

---

## A. Data factors

| # | Item | Status |
|---|------|--------|
| A1 | Stable `user_id` / `user_name` (out of model X) | ✅ |
| A2 | Tenure (`days_since_signup`) | ✅ |
| A3 | Short-term sessions (`sessions_last_7d`) | ✅ |
| A4 | Medium-term sessions (`sessions_last_30d`) | ✅ |
| A5 | Session depth (`avg_session_minutes`) | ✅ |
| A6 | Model breadth (`models_used_count`) | ✅ |
| A7 | API intensity (`api_calls_last_30d`) | ✅ |
| A8 | Token intensity (`tokens_consumed_last_30d`) | ✅ |
| A9 | Tool / integration breadth (`tools_used_count`) | ✅ |
| A10 | Friction rate (`failed_requests_rate`) | ✅ |
| A11 | Support load (`support_tickets_last_90d`) | ✅ |
| A12 | Plan tier enum + ordinal encode | ✅ |
| A13 | Payment / dunning (`payment_failures_last_90d`) | ✅ |
| A14 | Feature adoption score | ✅ |
| A15 | NPS-like satisfaction | ✅ |
| A16 | Recency (`last_active_days_ago`) | ✅ |
| A17 | Weekend habit ratio | ✅ |
| A18 | Engagement trend (7d vs 30d pace) | ✅ |
| A19 | Spend USD (30d) | ✅ |
| A20 | Days until renewal (B2B) | ✅ |
| A21 | Agent / automation runs (AI-native) | ✅ |
| A22 | IDE plugin sessions | ✅ |
| A23 | Seat utilization (team) | ✅ |
| A24 | Synthetic label with noise (no perfect formula) | ✅ |
| A25 | Hero profile JSON without label | ✅ |
| A26 | Data dictionary auto-generated | ✅ |
| A27 | Draft-07 JSON Schema for inference payload | ✅ |
| A28 | Real production event-log warehouse | ⬜ |
| A29 | Survival / time-to-churn labels | ⬜ |
| A30 | Graph / org hierarchy features | ⬜ |

**Data subtotal:** 27 ✅ / 3 ⬜ → **90%**

---

## B. ML / scoring

| # | Item | Status |
|---|------|--------|
| B1 | Stratified train/val/test, fixed seed | ✅ |
| B2 | Dummy + LogReg honest baselines | ✅ |
| B3 | Default XGB + Optuna-tuned XGB | ✅ |
| B4 | `scale_pos_weight` for imbalance | ✅ |
| B5 | Validation-fit probability calibration | ✅ |
| B6 | Holdout ROC / PR / Brier / confusion | ✅ |
| B7 | Best-F1 threshold sweep | ✅ |
| B8 | Feature stats (percentiles) saved at train | ✅ |
| B9 | Raw + calibrated single-record score | ✅ |
| B10 | Risk bands (low/medium/high) | ✅ |
| B11 | Local explanation (SHAP / fallback) | ✅ |
| B12 | Latency benchmark (p50/p95) | ✅ |
| B13 | Model card + metrics.json | ✅ |
| B14 | Online learning / continual retrain job | ⬜ |
| B15 | Multi-model ensemble serving | ⬜ |
| B16 | Causal uplift / treatment effect model | ⬜ |

**ML subtotal:** 13 ✅ / 3 ⬜ → **81%**

---

## C. UX / single-record path

| # | Item | Status |
|---|------|--------|
| C1 | CLI infer for Santosh | ✅ |
| C2 | Decision packet CLI (`single_record`) | ✅ |
| C3 | Payload validation (keys/ranges/enum) | ✅ |
| C4 | Outlier flags vs train percentiles | ✅ |
| C5 | Cohort percentile compare | ✅ |
| C6 | HITL action mapping from threshold | ✅ |
| C7 | Streamlit Predict tab | ✅ |
| C8 | Streamlit Explain tab | ✅ |
| C9 | Streamlit Decision tab (Santosh case) | ✅ |
| C10 | What-if sliders including new features | ✅ |
| C11 | Human-readable summary printed | ✅ |
| C12 | Packet JSON artifact under `artifacts/` | ✅ |
| C13 | Case-study doc with real numbers | ✅ |
| C14 | Production CRM / ticketing push | ⬜ |
| C15 | Batch single-record CLI (`--dir` → JSONL) | ✅ |

**UX subtotal:** 14 ✅ / 1 ⬜ → **93%**

Notes:
- **C15** — `python -m retention_radar.cli.single_record --dir path/to/jsons` writes `artifacts/batch_decision_packets.jsonl` (one packet per file). Single `--user santosh` unchanged. Not a multi-user Streamlit batch UI / CRM push (those remain enterprise).

---

## D. Ethics / HITL

| # | Item | Status |
|---|------|--------|
| D1 | Synthetic / no real PII disclaimer | ✅ |
| D2 | No auto-cancel / no auto-email | ✅ |
| D3 | Explicit `auto_action: none` in packet | ✅ |
| D4 | Escalate still requires human | ✅ |
| D5 | Model card ethical notes | ✅ |
| D6 | Explanation framed as correlation not causation | ✅ |
| D7 | Educational slice metrics by `plan_tier` | ✅ |
| D8 | Legal review / DPIA for production | ⬜ |

**Ethics subtotal:** 7 ✅ / 1 ⬜ → **88%**

Notes:
- **D7** — Test metrics by `plan_tier` (precision / recall / AUC when enough samples) saved under `metrics.json → slice_metrics_by_plan_tier`. **This is NOT a protected-class fairness audit** — `plan_tier` is a commercial segment, not a protected attribute. Formal fairness / disparate-impact / DPIA remains ⬜ (D8).

---

## E. Ops / reproducibility

| # | Item | Status |
|---|------|--------|
| E1 | `scripts/run_all.sh` E2E | ✅ |
| E2 | `run_all` ends with Santosh decision packet | ✅ |
| E3 | pytest smoke (features + packet keys) | ✅ |
| E4 | PYTHONPATH=src / venv documented | ✅ |
| E5 | Feature contract file (`feature_names.json`) | ✅ |
| E6 | Schema file under `configs/schemas/` | ✅ |
| E7 | Research spine + article links | ✅ |
| E8 | MIT license | ✅ |
| E9 | CI on GitHub Actions | ✅ |
| E10 | Model registry / versioned deploys | ⬜ |
| E11 | Lite drift check (train stats vs CSV) | ✅ |
| E12 | Push to remotes as part of demo | ⬜ (intentionally omitted) |

**Ops subtotal:** 10 ✅ / 2 ⬜ → **83%**

Notes:
- **E9** — `.github/workflows/ci.yml`: push/PR to `main`, Python 3.11, `N_USERS=800`, `N_OPTUNA_TRIALS=5`, seed-42 canary on the committed bundle → ruff (`src/ tests/ app/`) → `run_all.sh` (generate → train → evaluate → single_record) → pytest → strict drift_check. Local defaults remain 5000 users / 20 trials when env unset.
- **E11** — `python -m retention_radar.cli.drift_check` compares CSV means to `models/feature_stats.json` (abs mean z-score). Writes `artifacts/drift_report.json`. **Exits 0 by default** (demo); `--strict` exits 1 only on severe drift. Wired non-fatally in `run_all.sh` before the Santosh decision packet (E2: script still ends with the packet). Not a production Evidently / GE monitor (enterprise).

Enterprise deferred (still ⬜): **A28–A30, B14–B16, C14, D8, E10, E12**.

**FOSS lakehouse note (does not flip A28):** optional dual-path SoR via `local-data-lakehouse` → `data/external/` (`CHURN_DATA_SOURCE=lakehouse|auto`) is documented in [data-foundation-lakehouse.md](../data/data-foundation-lakehouse.md) and [e2e-free-platforms.md](../guides/e2e-free-platforms.md) §5b. That is a teaching gold export, **not** a production event-log warehouse (A28 remains ⬜). Published seed-42 metrics stay synthetic.

---

## Overall completeness

| Area | ✅ | ⬜ | % |
|------|---:|---:|---:|
| Data factors | 27 | 3 | 90% |
| ML / scoring | 13 | 3 | 81% |
| UX / single-record | 14 | 1 | 93% |
| Ethics / HITL | 7 | 1 | 88% |
| Ops | 10 | 2 | 83% |
| **Total** | **71** | **10** | **≈ 88%** |

In-scope FOSS teaching goals for this case study are essentially complete; remaining ⬜ items are production/enterprise extensions deliberately deferred.

---

## Quick verify commands

```bash
./scripts/run_all.sh
pytest -q
python -m retention_radar.cli.single_record --user santosh --out artifacts/santosh_decision_packet.json
python -m retention_radar.cli.single_record --dir /tmp/batch_jsons --out artifacts/batch_decision_packets.jsonl
python -m retention_radar.cli.drift_check
python -c "import app.streamlit_app"  # or: streamlit run app/streamlit_app.py
```
