# Model card: voluntary lapse at renewal (AI coding assistant, XGBoost)

_Generated: 2026-10-03 · seed=42_

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

F1 columns use a 0.5 threshold. Calibrated scores sit near the ~9% base rate and rarely reach 0.5, so the calibrated row shows F1 0; the served threshold τ is in the operating point below.

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

## Training setup

Optuna (20 trials, objective: validation AUC, best 0.7702) chose these XGBoost parameters:

| Parameter | Value |
|---|---|
| `n_estimators` | 91 |
| `max_depth` | 5 |
| `learning_rate` | 0.0247 |
| `subsample` | 0.7992 |
| `colsample_bytree` | 0.8421 |
| `min_child_weight` | 2 |
| `gamma` | 1.6822 |
| `reg_lambda` | 3.9919 |

The threshold τ = 0.16 is chosen on the validation split (best F1) and applied once to test.

### Ranking (test)

- Base rate: 0.0962.
- Top 10% by score: precision 0.2789, capturing 0.2908 of all lapses.

### By plan (test, at τ)

Segment diagnostics only, not a fairness audit.

| Plan | n | Lapse rate | Precision | Recall | ROC AUC |
|---|---:|---:|---:|---:|---:|
| pro | 1163 | 0.1109 | 0.2692 | 0.4884 | 0.7435 |
| pro_plus | 245 | 0.0367 | 0.0769 | 0.1111 | 0.7721 |
| ultra | 58 | 0.0517 | 0.0000 | 0.0000 | 0.7636 |

### Policy on the test set

n = 1466; contacted share 0.1576; expected value $837.48 (assumed playbook effects).

| Action | Rows |
|---|---:|
| `no_action` | 1218 |
| `limit_reset` | 113 |
| `cancel_flow_discount` | 82 |
| `pause_offer` | 35 |
| `holdout` | 17 |
| `personal_email` | 1 |

## Holdout and control design

- 10% of eligible subscribers are a fixed control group: a subscriber is in it when the first 8 hex digits of `sha256("holdout:<user_id>")` mod 100 fall below the holdout share (`serving/policy.py`). The same id always lands in the same group.
- Holdout rows are scored and logged with the action the policy would have taken (`would_have_sent`), but nothing is sent. Lift is the outcome gap between treated and holdout rows (`cli.outcomes`).
- The worked example `sub_santosh` is in the holdout: medium risk, `would_have_sent: limit_reset`.

## Lakehouse data (separate run)

The same pipeline also trains on the feature export of [local-data-lakehouse](https://github.com/santoshshinde2012/local-data-lakehouse) (synthetic events turned into point-in-time gold). That run writes only under `artifacts/lakehouse_run/`; the committed bundle above stays on the synthetic seed-42 data.

| Quantity | Value |
|---|---|
| Source | lakehouse gold N=7387 seed=42 (verified 2026-10-03) |
| n_train / n_test | 4431 / 1478 |
| Train lapse rate | 0.0740 |
| Best Optuna AUC (val) | 0.7335 |
| Calibrated test ROC AUC | 0.7261 |
| Threshold τ | 0.14 |

Source: [`results/lakehouse_e2e_summary.json`](../results/lakehouse_e2e_summary.json).

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

## Limitations

- Trained and evaluated on synthetic data only; there is no labelled real-world outcome.
- One seed (42) and one split. With about 140 lapses in test, AUC differences of a few points between ladder rows are within noise.
- Playbook effects and costs in `config.PLAYBOOKS` are assumptions, so the expected value is too.
- Thin segments: `pro_plus` (9 lapses in 245 test rows), `ultra` (3 lapses in 58 test rows). Treat their numbers as anecdotes.
- Drift checks fall back to the standardised mean difference: the committed `models/feature_stats.json` has no PSI bins.
- Latency is one machine's warm in-process timing, not a service SLA.

## Versioning

The served bundle is the committed `models/` directory (seed 42). It changes only with a retrain, recorded in the CHANGELOG; `make reproduce` retrains into `artifacts/repro/` and diffs against `models/metrics.json`.

| File | sha256 (first 12) |
|---|---|
| `models/calibrator.joblib` | `084e1bb4e112` |
| `models/churn_xgb.joblib` | `8ef0ee3164c2` |
| `models/feature_names.json` | `62be31e8c709` |
| `models/feature_stats.json` | `ef7040346431` |
| `models/metrics.json` | `1506885735aa` |

Pinned model libraries (`requirements.txt`): `scikit-learn==1.9.1`, `xgboost==3.4.1`, `lightgbm==4.7.0`, `catboost==1.2.10`, `optuna==5.0.0`.

## Related reading

- [architecture.md](architecture.md) — SOLID package map
- [data-dictionary.md](data/data-dictionary.md)
- [best-practices.md](guides/best-practices.md)
- [renewal-worked-examples.md](case-study/renewal-worked-examples.md)
- [algorithm-landscape.md](guides/algorithm-landscape.md) — what is on the ladder vs deferred
- [../results/benchmarks.md](../results/benchmarks.md)
- [../results/worked-examples.md](../results/worked-examples.md)
