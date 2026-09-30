# Model card: voluntary lapse at renewal (AI coding assistant, XGBoost)

_Generated: 2026-09-30 · seed=42_

## Overview

Binary classifier scoring, seven days before a monthly renewal, whether a paying subscriber of a self-serve AI coding assistant will voluntarily let the plan lapse.
Stack: XGBoost (Optuna-tuned) + post-hoc probability calibration (`sigmoid`) on the validation set.

## Intended use

- Teaching / FOSS case study: rank T-7 renewals, calibrate, and feed an expected-value policy that picks an approved playbook, a holdout, or no action.
- The score says who is at risk, not who will respond. Playbook effects in `config.PLAYBOOKS` are assumptions until a holdout measures them.
- **Not** for production decisions on real customers without fresh data validation.

## Training data

- Synthetic renewal cohort from `src/retention_radar/data/generate.py`: unobserved causes (need, fit, price sensitivity, side-project habit, a rival tool's pull) drive observed usage and a noisy voluntary-lapse outcome. **No NaNs by design.**
- Dunning (failed-payment) lapses and already-scheduled cancels are excluded; see the data dictionary.
- Split: train ~60% / val ~20% / test ~20% (stratified, `random_state=42`).
- n_train=4397 · n_val=1466 · n_test=1466 · n_trials=20.
- Train voluntary-lapse rate: `0.09642938367068456`.

## Features

22 numeric features after ordinal `plan_tier` encoding.
See [data-dictionary.md](data/data-dictionary.md) for dtype, ranges, and nullability.

## Metrics (holdout) — from `models/metrics.json`

Honest ladder: **Dummy(prior) → LogReg → RF → default XGB → Optuna XGB → LightGBM → CatBoost** (calibrated XGB serves). No simple-rule baseline is logged. GBDT trilogy peers: XGB / LightGBM / CatBoost.

| Model | Val AUC | Val F1 | Test AUC | Test F1 | Test PR-AUC |
|-------|---------|--------|----------|---------|-------------|
| Dummy (prior) | 0.5000 | 0.0000 | 0.5000 | 0.0000 | 0.0962 |
| Logistic regression | 0.7706 | 0.3130 | 0.7811 | 0.3333 | 0.2812 |
| Random Forest | 0.7653 | 0.3831 | 0.7576 | 0.3030 | 0.2672 |
| XGBoost (default) | 0.7411 | 0.3132 | 0.7290 | 0.3288 | 0.2336 |
| XGBoost (Optuna, raw) | 0.7702 | 0.3505 | 0.7573 | 0.3108 | 0.2697 |
| LightGBM (default) | 0.7525 | 0.3185 | 0.7359 | 0.3093 | 0.2430 |
| CatBoost (default) | 0.7694 | 0.3171 | 0.7649 | 0.3302 | 0.2757 |
| XGBoost (calibrated) | 0.7702 | 0.0000 | 0.7573 | 0.0000 | 0.2697 |

### Calibration (Brier — lower is better)

| Split | Raw Brier | Calibrated Brier |
|-------|-----------|------------------|
| val | `0.1607` | `0.0774` |
| test | `0.1627` | `0.0794` |

Method: `sigmoid`.

### Operating point (calibrated test scores)

| Quantity | Value |
|----------|-------|
| Test AUC-ROC (calibrated) | `0.7573` |
| Test PR-AUC / AP | `0.2697` |
| Best F1 threshold τ (chosen on validation) | `0.16` |
| F1 at τ (validation) | `0.3800` |
| F1 at τ (test, reported once) | `0.3290` |
| F1 @ 0.5 | `0.0000` |
| Warm latency p50 / p95 (ms) | `2.73` / `3.28` |

## Result plots

Committed copies live under `results/plots/` (runtime dumps in `artifacts/`).

![ROC](../results/plots/roc_curve.png)

![Precision–Recall](../results/plots/pr_curve.png)

![Calibration](../results/plots/calibration_curve.png)

![Confusion matrix](../results/plots/confusion_matrix.png)

![Threshold vs F1](../results/plots/threshold_f1.png)

## Artifacts

| Path | Contents |
|------|----------|
| `models/churn_xgb.joblib` | Tuned XGBoost + metadata |
| `models/calibrator.joblib` | Validation-fit probability calibrator |
| `models/metrics.json` | Full metric dump (source of truth) |
| `artifacts/roc_curve.png` | ROC |
| `artifacts/pr_curve.png` | Precision–Recall |
| `artifacts/calibration_curve.png` | Reliability diagram |
| `artifacts/confusion_matrix.png` | Confusion @ 0.5 |
| `artifacts/threshold_f1.png` | Threshold vs F1 / precision / recall |
| `results/plots/*.png` | Committed copies of the same plots |

## Ethical notes

- Labels and features are synthetic; do not treat scores as real risk.
- Calibration improves probability meaning but does not fix selection bias.
- SHAP explains this score, not causation.
- The service never executes an action (`auto_action: none`); a lifecycle tool runs human-approved playbooks, and a deterministic holdout is kept out of every playbook.
- Contacting at-risk subscribers can raise churn (Ascarza et al., JMR 2016). Measure lift against the holdout before scaling any playbook.

## Related reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — SOLID package map
- [data-dictionary.md](data/data-dictionary.md)
- [BEST_PRACTICES.md](guides/BEST_PRACTICES.md)
- [renewal-worked-examples.md](case-study/renewal-worked-examples.md)
- [ALGORITHM_LANDSCAPE.md](guides/ALGORITHM_LANDSCAPE.md) — what is on the ladder vs deferred
- [../results/BENCHMARKS.md](../results/BENCHMARKS.md)
- [../results/WORKED_EXAMPLES.md](../results/WORKED_EXAMPLES.md)
