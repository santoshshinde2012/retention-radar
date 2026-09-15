# Algorithm landscape — what is on the ladder vs deferred

Teaching decision for Retention Radar (2025–26). **Cite:** [`../models/metrics.json`](../models/metrics.json) · **Benchmarks:** [`../results/BENCHMARKS.md`](../results/BENCHMARKS.md)

This study ships an **honest bake-off ladder** on a synthetic AI-platform churn table (seed **42**, N=5000, 22 numeric features after plan encoding). The **serving / Santosh / SHAP hero stays Optuna-tuned XGBoost + isotonic calibration**, even when a peer edges AUC.

## IN — published ladder (implement)

| Stage | Model | Role |
|-------|--------|------|
| Prior | Dummy (`strategy=prior`) | Accuracy theatre control |
| Linear | Logistic regression (scaled, balanced) | Strong additive-label peer |
| Bagging | Random Forest | Mid-tier ensemble peer |
| GBDT | XGBoost (default) | Tree baseline before Optuna |
| GBDT tuned | XGBoost (Optuna on val AUC) | Teaching vehicle |
| GBDT peer | LightGBM (sane defaults) | FOSS peer booster |
| GBDT peer | **CatBoost (sane defaults)** | Completes the **XGB / LightGBM / CatBoost** trilogy |
| Serve | **XGBoost + isotonic calibration** | Santosh packet, risk bands, SHAP |

**Why CatBoost now:** TabArena (NeurIPS 2025 Datasets & Benchmarks; arXiv:2506.16791) treats CatBoost as a first-class conventional tree and, in the conventional tuning regime, often ranks it at or near the top of classical GBDT peers. For a beginner CPU FOSS HITL packet path, a **default** CatBoost row is the missing honest peer — not a production crown.

**What we do *not* claim:** CatBoost (or any booster) “wins production churn.” On this synthetic, mostly additive label, LogReg and CatBoost can tie or edge Optuna XGB on test AUC; we still serve calibrated XGB for SHAP + calibration UX.

## DEFER — written rationale (do not implement here)

| Family | Examples | Why deferred for *this* teaching arc |
|--------|----------|--------------------------------------|
| Tabular foundation / ICL | TabPFN, TabPFN-v2/v3, TabICL | Strong on **small** tables on TabArena-class boards; licensing / GPU / serve story does not fit beginner **CPU FOSS HITL packet** path |
| Heavy tabular DL | RealMLP, FT-Transformer, TabM | Heavy deps and training story; not the DET / tree-ladder lesson |
| Survival | Cox PH, Random Survival Forest | Needs **time-to-event** label redesign; we keep a **binary snapshot** label |
| Causal / uplift | Meta-learners, T-/S-/X-learners | Different question (treatment effect), not P(churn) ranking |
| Soft-voting mega-ensemble | Average of all ladder models | Muddles ladder honesty; hides which inductive bias won |
| Conformal prediction | MAPIE | Valuable **HITL next study** (prediction sets); mention only — do not code unless trivial |
| Glass-box GAM | EBM (Explainable Boosting Machine) | Nice interpretability peer; defer to keep the published ladder readable |

## Fit to this use case

- **Label:** binary `churned` at a snapshot — not duration or treatment assignment.
- **Serve:** single-record JSON → 22 features → calibrated `p` → band → SHAP → HITL (`auto_action: none`).
- **Audience:** beginners on CPU with FOSS pins (`requirements.txt`).
- **Data foundation:** lakehouse gold exports feed the **same 22-feature contract**; algorithm choice lives in this repo, not in the lakehouse.

## Citations (landscape, not victory laps)

- Erickson et al., *TabArena: A Living Benchmark for Machine Learning on Tabular Data* (NeurIPS 2025 D&B; [arXiv:2506.16791](https://arxiv.org/abs/2506.16791)) — living Elo-style tabular board; CatBoost / LightGBM / XGBoost as conventional trees; neural and foundation models after ensembling.
- Prokhorenkova et al., CatBoost (arXiv:1706.09516); Ke et al., LightGBM; Chen & Guestrin, XGBoost — the GBDT trilogy peers on this ladder.
- Churn teaching practice: always publish Dummy + linear + tree peers before claiming booster success (see [`../results/BENCHMARKS.md`](../results/BENCHMARKS.md)).

## Related

- [`MODEL_CARD.md`](../MODEL_CARD.md) — metrics table from `models/metrics.json`
- [`BEST_PRACTICES.md`](BEST_PRACTICES.md) — dual-world synthetic vs lakehouse
- [`ARCHITECTURE.md`](../ARCHITECTURE.md) — train ≠ serve
