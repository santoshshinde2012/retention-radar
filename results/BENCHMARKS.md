# Benchmarks — seed 42 analysis

How we measure success, what the reference run produced, and how to read the ladder honestly.

**Cite:** [`../models/metrics.json`](../models/metrics.json) · **Card:** [`../docs/MODEL_CARD.md`](../docs/MODEL_CARD.md) · **Landscape:** [`../docs/ALGORITHM_LANDSCAPE.md`](../docs/ALGORITHM_LANDSCAPE.md) · **Santosh:** [`SANTOSH_ANALYSIS.md`](SANTOSH_ANALYSIS.md)

---

## What we measure

| Metric | Role |
|--------|------|
| AUC-ROC | Ranking quality across thresholds |
| PR-AUC / average precision | Ranking under imbalance (~19.1% train churn) |
| Precision / recall / F1 | At τ = 0.5 in ladder rows; best-F1 τ separate |
| Brier (raw vs calibrated) | Probability reliability |
| Warm latency | Single-row transform + `predict_proba` (+ calibrator) |

Splits: stratified n_train / val / test = **3000 / 1000 / 1000**, seed **42**. Optuna on **validation** only; test reported once.

---

## Table A — Honest ladder (test)

| Model | Test AUC-ROC | Test PR-AUC | Test F1 @ 0.5 | Notes |
|-------|--------------|-------------|----------------|-------|
| Dummy (prior) | **0.500** | **0.191** | **0.000** | Accuracy **0.809** by predicting non-churn |
| Logistic regression | **0.872** | **0.644** | **0.578** | Best AUC at full float (edges CatBoost) |
| Random Forest | **0.868** | **0.643** | **0.584** | Mid-tier ensemble peer |
| XGBoost default | **0.869** | **0.638** | **0.596** | Sane hyperparameters |
| XGBoost Optuna (raw) | **0.870** | **0.641** | **0.580** | Teaching / SHAP vehicle |
| LightGBM (default) | **0.865** | **0.633** | **0.584** | FOSS peer booster |
| CatBoost (default) | **0.872** | **0.636** | **0.583** | GBDT trilogy peer (ties LogReg at 3dp) |
| XGBoost calibrated | **0.866** | **0.612** | **0.601** | **Serving hero**; threshold-sensitive |

**Honest finding:** On this synthetic, mostly additive label, **LogReg and CatBoost land at published test AUC 0.872** (LogReg slightly ahead at full float: 0.8719 vs 0.8715). Optuna XGBoost **0.870**, RF **0.868**, LightGBM **0.865**. That does not invalidate the XGBoost teaching path (SHAP, nonlinear capacity, calibration demo) — it *does* forbid booster-only victory laps. **Serving hero stays calibrated XGB** even when CatBoost matches LogReg on rounded AUC.

---

## Table B — Calibration & threshold

| Quantity | Value |
|----------|-------|
| Brier raw (test) | **0.138** |
| Brier calibrated (test) | **0.106** |
| Calibration method | isotonic (fit on validation) |
| Best F1 threshold τ | **0.34** |
| Best F1 at that τ | **0.602** |
| Calibrated test AP | **0.612** |

---

## Table C — Latency (warm, calibrated path)

From `metrics.json` → `latency` (host-dependent; cite the JSON):

| Quantity | Value |
|----------|-------|
| p50 | **~2.01 ms** |
| p95 | **~2.15 ms** |
| mean | **~2.01 ms** |
| warmup / runs | 20 / 200 |

Fine for Streamlit; SHAP is separate and should stay on-demand.

---

## Table D — Optuna meta

| Item | Value |
|------|-------|
| Trials | 20 |
| Best val AUC | **0.913** |
| Example best depth | **3** |
| Train churn rate | **0.191** |

---

## Reference plots

![ROC](plots/roc_curve.png)

![PR](plots/pr_curve.png)

![Calibration](plots/calibration_curve.png)

![Threshold F1](plots/threshold_f1.png)

![Confusion](plots/confusion_matrix.png)

---

## How to reproduce

```bash
cd retention-radar
source .venv/bin/activate
export PYTHONPATH="$(pwd)"
CHURN_DATA_SOURCE=synthetic ./scripts/run_all.sh
python -m json.tool models/metrics.json | less
python -m src.infer --user santosh
make docs-results   # refresh results/plots/ from artifacts/
```

Same seed + same package versions should match within float noise. Changing the generator breaks bit-identical AUC — update the card and this narrative together.

**Dual world:** lakehouse gold E2E overwrites `models/` and is **not** the published ladder. See [`lakehouse-e2e-summary.json`](lakehouse-e2e-summary.json) and [docs/data-foundation-lakehouse.md](../docs/data-foundation-lakehouse.md). Restore: `git checkout -- models/ docs/MODEL_CARD.md docs/data-dictionary.md`.

---

## What not to claim

- Production ROI from synthetic lift  
- Fairness / DPIA from plan-tier slices alone  
- That XGBoost or CatBoost “won production churn” when LogReg/CatBoost tie or edge AUC  
- Mixing lakehouse Santosh scores with the seed-42 packet in one caption  
