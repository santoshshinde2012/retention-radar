# Benchmarks (seed 42)

What the reference run produced and how to read it. Source of truth:
[`../models/metrics.json`](../models/metrics.json) for metrics and
[`analysis.json`](analysis.json) (bootstrap, calibrator comparison, deciles, holdout sizes).
Model card: [`../docs/model-card.md`](../docs/model-card.md). Worked examples:
[`WORKED_EXAMPLES.md`](WORKED_EXAMPLES.md).

The committed bundle is trained on Linux x86-64 (the CI platform), where `make reproduce`
matches it exactly. On other CPUs (for example Apple Silicon) the tree libraries and the
Optuna search take slightly different paths, and a retrain differs from the third decimal.

## The table under the numbers

- 8,000 synthetic renewals. 280 were lost to failed cards (dunning) and 391 had a cancel
  already scheduled (cancel flow); both are routed out, leaving **7,329** T-7 rows for the model.
- Voluntary-lapse rate in the model table: **9.6%**.
- Stratified split 60/20/20: train **4,397** · validation **1,466** · test **1,466**
  (424 / 142 / 141 lapses).
- Optuna (20 trials) and the calibrator see validation only; test is read once.

## Table A: the ladder (test)

| Model | AUC-ROC | PR-AUC | F1 @ 0.5 | Notes |
|-------|---------|--------|----------|-------|
| Dummy (prior) | 0.500 | 0.096 | 0.000 | Accuracy 0.904 by predicting "renews" for everyone |
| **Logistic regression** | **0.781** | **0.281** | 0.333 | Best ranking on this run |
| CatBoost (untuned) | 0.765 | 0.276 | 0.330 | |
| Random Forest | 0.758 | 0.267 | 0.303 | |
| XGBoost (Optuna, raw) | 0.757 | 0.270 | 0.311 | The serving model before calibration |
| LightGBM (untuned) | 0.736 | 0.243 | 0.309 | |
| XGBoost (untuned) | 0.729 | 0.234 | 0.329 | |
| XGBoost (calibrated) | 0.757 | 0.270 | 0.000 | Calibration keeps the ranking; at 0.5 it flags nobody (max calibrated p = 0.45) |

"Untuned" means fixed hand-set parameters (CatBoost and LightGBM: 120 rounds, depth 4,
learning rate 0.08).

**Reading it.** Logistic regression beats the tuned booster by 0.024. In a paired bootstrap
of the test set (2,000 resamples) it comes out ahead 99.4% of the time; the 95% interval for
the difference is +0.004 to +0.045. The generator's lapse logit is mostly additive in its
causes, and there are only 424 training lapses for a tree to find interactions in.
Calibrated XGBoost still serves because the explanation and serving code were built around
it before the ladder came in; logistic regression would explain each subscriber just as
exactly (coefficient × standardised value). For a real deployment with this result, swap it
in: serving depends only on `predict_proba`.

## Table B: calibration and the operating point

| Quantity | Value |
|----------|-------|
| Brier, always predict the base rate | 0.087 |
| Brier raw (test) | 0.163 (inflated by `scale_pos_weight`) |
| Brier calibrated (test) | **0.079** |
| Calibration method | Platt (sigmoid), fit on validation |
| τ, best F1 on validation, frozen | **0.16** |
| F1 at τ: validation / test | 0.380 / 0.329 |
| At τ on test | 248 flagged (16.9%); precision 0.258; recall 0.454 |
| Top 10% of scores | 27.9% lapse (2.9× the base rate); holds 29.1% of all lapses |

Lapse rate by score decile (test, decile 1 = highest scores): 27.9% · 19.9% · 15.0% · 8.2% ·
8.8% · 6.9% · 3.4% · 2.1% · 2.7% · 1.4%.

Calibration by quintile of calibrated p (test):

| Quintile | Mean p | Observed lapse rate |
|----------|--------|---------------------|
| 1 | 0.031 | 0.020 |
| 2 | 0.045 | 0.027 |
| 3 | 0.068 | 0.079 |
| 4 | 0.110 | 0.116 |
| 5 | 0.224 | 0.239 |

**Why Platt, not isotonic.** With 142 lapses in validation, isotonic regression fit a
staircase: 37 distinct values on the test set, and 44 test subscribers at a calibrated P of
exactly 0.000. Platt scaling scored slightly better on test Brier (0.0794 vs 0.0799), kept a
distinct score for every subscriber, and never returns zero.

**Why the Brier gain is small.** A calibrated 0.079 against 0.087 for "always say 9.6%" is
about a 9% Brier skill. Most renewals are hard to call seven days out; the value is
concentrated at the top of the queue.

## Table C: the policy on the test set

Actions the renewal policy would take for the 1,466 test subscribers (τ = 0.16, 10% holdout,
playbook effects are **assumptions** in `config.PLAYBOOKS`):

| Action | Subscribers |
|--------|-------------|
| no_action | 1,218 |
| limit_reset | 113 |
| cancel_flow_discount | 82 |
| pause_offer | 35 |
| holdout | 17 |
| personal_email (Ultra) | 1 |

15.8% contacted. Summed expected value ≈ $837, which is only as true as the assumed effects.

**How big a holdout would it take?** Among test subscribers above τ, 25.8% lapse. With a 10%
holdout (a 9:1 split), detecting the limit reset's assumed effect (25.8% → 19.4%) at 95%
confidence and 80% power takes 394 held-out and 3,546 treated subscribers, **3,940** in all.
The discount's smaller assumed effect (25.8% → 22.7%) takes 1,727 held out and 15,543
treated, **17,270**. The policy routes 8.5% of all renewals to the limit reset and 5.8% to the
discount, so that is roughly **46,000** and **298,000** renewals of traffic. One day of
renewals, like the use-case pack, is nowhere near either.

## Table D: latency and search

| Quantity | Value |
|----------|-------|
| Warm single-row score + calibrate, p50 | ~2.7 ms on a GitHub Actions runner (host-dependent) |
| Optuna best trial | `max_depth=5`, `n_estimators=91`, `learning_rate≈0.025` |
| Optuna best validation AUC | 0.770 |

## Plots

![ROC](plots/roc_curve.png)
![Precision–Recall](plots/pr_curve.png)
![Calibration](plots/calibration_curve.png)
![Threshold vs F1](plots/threshold_f1.png)
![Confusion matrix @ 0.5](plots/confusion_matrix.png)

## Reproduce

```bash
make reproduce        # retrain into artifacts/repro/ and diff against models/metrics.json
```

## What not to claim

- That XGBoost beats logistic regression. It doesn't on this run.
- Any real effect size for a playbook. The effects are inputs, not results.
- Production lift. The generator encodes the mechanisms from the public record; it
  does not reproduce any company's data.
