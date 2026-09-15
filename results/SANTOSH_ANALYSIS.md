# Santosh Shinde — single-record outcome analysis

End-to-end outcome for the hero JSON through Retention Radar (seed **42**, synthetic only).

**Reproduce:** `PYTHONPATH=src python -m retention_radar.cli.single_record --user santosh --out artifacts/santosh_decision_packet.json`  
**Sample packet:** [`santosh_decision_packet.sample.json`](santosh_decision_packet.sample.json)  
**Case study (longer walkthrough):** [`../docs/case-study/santosh-case-study.md`](../docs/case-study/santosh-case-study.md)

---

## Outcome (committed seed-42 serve bundle)

| Field | Value |
|-------|-------|
| Raw P(churn) | **0.043** |
| Calibrated P(churn) | **0.017** |
| Risk band | **low** |
| Best-F1 τ (frozen) | **0.34** |
| Half-τ (monitor gate) | **0.17** |
| HITL action | **monitor** |
| `auto_action` | **none** |
| Features | **22** (contract from `models/feature_names.json`) |

Validation: schema OK, required keys present, no hard range failures on the committed profile.

### Top SHAP drivers (direction ≈ lower risk)

Typical top contributors on the reference packet (signs = model contribution toward churn):

| Feature | Contribution (approx.) | Reading |
|---------|------------------------|---------|
| `feature_adoption_score` | **−1.08** | Broad adoption pushes risk **down** |
| `support_tickets_last_90d` | **−0.40** | Mild ticket count still net protective vs cohort patterns in this synthetic fit |
| `last_active_days_ago` | **−0.40** | Recent enough activity |
| `agent_runs_last_30d` | **−0.28** | Agent usage signal |
| `failed_requests_rate` | **−0.26** | Friction present but not dominant |

Exact floats live in the packet JSON; do not treat SHAP as causation.

---

## Why this packet exists

1. **Canary for train≠serve** — same 22-name contract as training; serve only loads artifacts.  
2. **Calibration teaching** — raw **0.043** → cal **0.017** both belong on the UI.  
3. **HITL policy** — calibrated p ≪ 0.5×τ → **monitor**, never auto-cancel.  
4. **UX honesty** — engaged-looking user + mild friction → still **low**; do not over-alarm.

---

## Dual world (do not mix captions)

| World | Santosh scores | Where |
|-------|----------------|-------|
| **Synthetic (published)** | raw ≈ 0.043 / cal ≈ 0.017 · low · monitor | This analysis + `models/` |
| **Lakehouse gold** | different as-of profile (see summary JSON) | [`lakehouse-e2e-summary.json`](lakehouse-e2e-summary.json) |

Published Medium / README numbers are the **synthetic** row.

---

## Related

- [BENCHMARKS.md](BENCHMARKS.md) — ladder, Brier, τ, latency  
- [docs/guides/ALGORITHM_LANDSCAPE.md](../docs/guides/ALGORITHM_LANDSCAPE.md) — CatBoost IN; TabPFN/survival/conformal DEFER  
- [docs/case-study/single-record-checklist.md](../docs/case-study/single-record-checklist.md)  
- [docs/MODEL_CARD.md](../docs/MODEL_CARD.md)  
- Articles: authored separately (**internal**); this repo is the public code / results home  
