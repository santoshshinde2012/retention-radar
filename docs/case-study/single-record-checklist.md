# Single-record checklist

What one T-7 record goes through in this repo, and what is deliberately left out.

## In the repo

| Stage | What happens | Where |
|-------|--------------|-------|
| Contract | 24 fields (22 features + ids); JSON Schema; frozen feature order in the bundle | `configs/schemas/user_record.schema.json`, `models/feature_names.json` |
| Normalise | extra fields dropped with a warning; `plan_tier` case/whitespace normalised; numeric text parsed | `serving/packet.py::normalize_record` |
| Validate | missing/null keys, non-finite numbers, booleans, ranges, plan enum (Teams is not a plan here), binary flags, `active_days_7d ≤ active_days_28d`, trend-formula warning | `serving/packet.py::validate_payload` |
| Hold | any error → never scored, never queued: `hold: fix input data` | `serving/policy.py::validation_hold` |
| Score | raw `predict_proba` + Platt-calibrated probability | `serving/infer.py`, `serving/scoring.py` |
| Band | fixed edges 0.10 / 0.30 on calibrated p (not τ) | `serving/policy.py::risk_band` |
| Explain | top SHAP drivers (log-odds); gain-based fallback | `serving/explain.py` |
| Context | percentile vs the renewal table (or training quantiles when the table is absent); training p01–p99 outlier flags | `serving/packet.py` |
| Decide | below τ → no action; holdout (10%, hash of id) → nothing sent; else best-EV approved playbook | `serving/policy.py::decide` |
| Log | what was actually done, including holdout and suppressions | `serving/action_log.py` |
| Outcome | lapse by band; lift vs holdout per playbook with a Newcombe interval, no verdict under 30 per group | `serving/outcomes.py` |

## Deliberately out of scope

- Teams / seat-contraction model and its human review queue.
- An uplift (treatment-effect) model: needs randomised treatment data first, which the holdout is there to produce.
- Survival / time-to-lapse modelling across several renewals.
- A real event warehouse. The lakehouse repo is a local teaching foundation.
- Online retraining, a model registry, a messaging-tool integration, a DPIA.

## Verify

```bash
./scripts/run_all.sh
pytest -q
python -m retention_radar.cli.single_record --user maya
python -m retention_radar.cli.single_record --dir data/use_cases/invalid   # all held
make use-cases
```
