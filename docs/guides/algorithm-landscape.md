# Algorithm landscape

Which models are on the ladder, which serves, and which were left out on purpose.
Numbers: [`models/metrics.json`](../../models/metrics.json) ·
[results/benchmarks.md](../../results/benchmarks.md).

The table is 7,329 T-7 renewals (seed 42) with 22 features after encoding `plan_tier`.

## On the ladder

| Model | Why it is there |
|-------|-----------------|
| Dummy (prior) | Shows what accuracy is worth at a 9.6% base rate |
| Logistic regression (scaled, class-balanced) | Linear peer; the generator's lapse logit is mostly additive |
| Random Forest | Bagged trees |
| XGBoost, default settings | Booster before tuning |
| XGBoost, Optuna-tuned on validation AUC | The serving model |
| LightGBM, default settings | Peer booster |
| CatBoost, default settings | Peer booster; strong on tabular benchmarks such as TabArena |

Logistic regression ranks best on this run (test AUC 0.781 against 0.757 for tuned
XGBoost). Calibrated XGBoost still serves because the explanation and serving code were built around it before the ladder came in; logistic regression would explain each subscriber just as exactly (coefficient × standardised value), and a calibrator works on either. For a real deployment with this result, swap it in: serving depends only on `predict_proba`.

## Calibration

Platt scaling (sigmoid), fit on validation. Isotonic regression was tried and rejected:
with 142 lapses in validation it produced a staircase of 37 distinct values,
tied most of the queue, and gave some subscribers a calibrated probability of exactly
0.000. Details: [results/benchmarks.md](../../results/benchmarks.md).

## Left out

| Family | Examples | Why not here |
|--------|----------|--------------|
| Uplift / treatment-effect models | T-, S-, X-learners, causal forests | The right next step for choosing who to contact, but they need randomised treatment data. The 10% holdout is there to produce it |
| Survival models | Cox PH, random survival forest | Would model time to lapse across several renewals; this repo scores one renewal at a time |
| Tabular foundation models | TabPFN, TabICL | Strong on small tables; licensing, GPU and serving story do not fit a CPU-only repo |
| Tabular deep learning | RealMLP, FT-Transformer, TabM | Heavy dependencies for little expected gain on this table |
| Ensembles of the ladder | soft voting | Hides which model family is doing the work |
| Conformal prediction | MAPIE | Useful for "not sure" flags; not needed for the current policy |
| Glass-box GAMs | EBM | A reasonable alternative to LogReg + SHAP; left out to keep the ladder short |

## References

- Erickson et al., *TabArena: A Living Benchmark for Machine Learning on Tabular Data*,
  NeurIPS 2025 Datasets and Benchmarks, [arXiv:2506.16791](https://arxiv.org/abs/2506.16791).
- Prokhorenkova et al., CatBoost, arXiv:1706.09516. Ke et al., LightGBM (NeurIPS 2017).
  Chen and Guestrin, XGBoost (KDD 2016).
- Ascarza, "Retention Futility: Targeting High-Risk Customers Might Be Ineffective",
  JMR 2018, on why a risk score is not an uplift score.

Related: [model-card.md](../model-card.md) · [best-practices.md](best-practices.md) ·
[architecture.md](../architecture.md)
