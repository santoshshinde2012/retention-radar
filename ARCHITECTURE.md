# Architecture — Retention Radar (XGBoost AI Platform Churn)

**Model card:** [MODEL_CARD.md](MODEL_CARD.md) · **Practices:** [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md) · **Lakehouse:** [docs/data-foundation-lakehouse.md](docs/data-foundation-lakehouse.md)

## Purpose

End-to-end view of the fictional AI-platform **Retention Radar**: synthetic data through Streamlit single-record inference for **Santosh Shinde**. All components are FOSS (Python, pandas, scikit-learn, XGBoost, Optuna, SHAP, Streamlit).

The radar ranks quiet fade-out risk and hands a flight checklist to a human — it does not auto-act.

## End-to-end flow


```mermaid
flowchart TB
  subgraph offline ["Offline / Train"]
    A["Synthetic data generator"] --> B["Ingest + validate"]
    B --> C["Feature engineering<br/>22 columns"]
    C --> D["Train / val / test split"]
    D --> E["Honest ladder<br/>Dummy → LogReg → RF → XGB → LightGBM"]
    E --> F["Evaluate + calibrate + τ"]
    F --> G["Artifacts bundle"]
  end

  subgraph online ["Serve / Infer"]
    G --> H["Artifact loader"]
    H --> I["Single-record transform"]
    I --> J["predict_proba + calibrator"]
    J --> K["Risk band"]
    J --> L["SHAP explain"]
    K --> M["Streamlit / CLI packet"]
    L --> M
  end

  N["Hero: Santosh Shinde"] --> M
  M --> O["HITL: monitor / nurture / outreach"]
```

## Train vs serve boundary


```mermaid
flowchart LR
  subgraph train_side ["Train side — may write"]
    CFG["src/retention_radar/config.py"] --> TR["train / evaluate / calibrate"]
    TR --> ART["models/ + artifacts/"]
  end

  subgraph serve_side ["Serve side — read only"]
    ART --> INF["serving/infer.py / serving/packet.py"]
    INF --> UI["Streamlit app"]
    UI --> USER["Santosh or manual form"]
  end

  ART -.->|"never Optuna / never fit encoders"| serve_side
```

- **Train** may write models, metrics, plots, calibrators.
- **Serve** must not fit encoders, call Optuna, or regenerate labels.
- Shared: feature name list (22), threshold/band config, version metadata, [MODEL_CARD.md](MODEL_CARD.md).

## Module map

```mermaid
flowchart TB
  subgraph src_mods ["src/retention_radar"]
    CFG2["config.py"]
    GEN["data/generate.py"]
    ING["data/ingest.py"]
    FEA["features/transform.py"]
    TRN["training/"]
    EVA["evaluation/"]
    INF2["serving/infer.py"]
    SR["serving/packet.py"]
    POL["serving/policy.py"]
    EXP["serving/explain.py"]
  end
  GEN --> ING --> FEA --> TRN --> EVA
  TRN --> MOD["models/*.joblib + feature_names + metrics"]
  EVA --> MOD
  MOD --> INF2
  MOD --> SR
  EXP --> SR
  POL --> SR
  SR --> APP["app/streamlit_app.py"]
  INF2 --> APP
```

```text
retention-radar/
├── src/retention_radar/   # packages: data, features, training, evaluation, serving
├── src/*.py               # shims: python -m src.train, src.infer, …
├── app/streamlit_app.py
├── scripts/run_all.sh
├── data/raw/      # users.csv, santosh_shinde.json
├── models/        # churn_xgb.joblib, calibrator, feature_names.json, metrics.json
├── artifacts/     # plots + santosh_decision_packet.json
├── docs/          # dictionary, checklist, lakehouse notes, result charts
├── ARCHITECTURE.md / MODEL_CARD.md
├── data/external/ # lakehouse gold sync (gitignored)
└── scripts/       # run_all.sh + run_lakehouse_e2e.sh
```

**Contracts:** config in → artifacts out → UI reads artifacts only. Dictionary and `feature_names.json` must agree.

## Pipeline stages

| Stage | Responsibility | Typical outputs |
|-------|----------------|-----------------|
| Generate | Synthetic users + noisy `churned`; inject Santosh | `users.csv`, `santosh_shinde.json` |
| Ingest / validate | Schema, ranges, label rate, hero row | Clean table + fail-fast errors |
| Features | Select columns, encode `plan_tier` | `X`, `y` (22 model features) |
| Train / tune | Stratified splits, imbalance, baselines, Optuna on **val** | Best params, fitted model |
| Evaluate | AUC, PR, F1, Brier, τ, latency | `metrics.json`, plots |
| Infer | Load bundle, score one row | Probability + band + drivers |
| UI | Presets, forms, Decision tab | Streamlit HITL |
| Ops-lite | Drift, retrain, when not to ship | docs/BEST_PRACTICES.md |

## Key artifacts contract

| File | Consumer |
|------|----------|
| `models/churn_xgb.joblib` | `infer`, Streamlit |
| `models/calibrator.joblib` | calibrated `p` for bands |
| `models/feature_names.json` | Transform order (22) for Santosh’s vector |
| `models/metrics.json` | MODEL_CARD.md tables |
| `artifacts/santosh_decision_packet.json` | Case study + CI canary |
| Risk thresholds / bands | `infer.risk_band`, UI chips |
| `MODEL_CARD.md` | Humans; fill metrics after `run_all` |

## Runtime views

**Batch / offline:** `./scripts/run_all.sh` → compare metrics → update model card.  
**Interactive:** Streamlit → load Santosh → edit → score + SHAP.  
**Canary:** freeze Santosh JSON; diff `P(churn)` after retrain (reference **0.043 / 0.017** · monitor).

## Non-goals

- Live billing or CRM write-back  
- Multi-tenant auth  
- GPU serving  
- Paid feature stores or AutoML  

## SOLID map (packages → principles)

The layout is a **teaching** SOLID sketch, not a claim that every file is a textbook example. Callers depend on small Protocols so Dummy, LogReg, and XGBoost can swap at score time.

| Package / module | SRP | OCP | LSP | ISP | DIP |
|------------------|-----|-----|-----|-----|-----|
| `config.py` | One place for paths, seed, 22-column contract | Extend features via config, not scattered lists | — | Does not expose train/serve APIs | Downstream depends on config values, not ad-hoc paths |
| `data/generate.py` | Synthetic table + Santosh inject only | New generator knobs without touching serve | — | No scoring interface | Train scripts depend on CSV contract |
| `data/ingest.py` | Load + validate; fail loud on NaNs / bad plans | Extra checks can be added without changing transform | — | Validation is not mixed with Optuna | Train/eval depend on `load_users` |
| `features/transform.py` | Encode `plan_tier`, build X/y | New columns via `MODEL_FEATURE_COLUMNS` | `DefaultFeatureTransformer` honours `FeatureTransformer` | Transform-only API | Train/serve call the transformer, not pandas ad-hoc |
| `training/split.py` | Stratified split only | — | Same split helper for train/eval | No model API | Train/eval depend on split, not sklearn calls inline |
| `training/baselines.py` | Dummy + LogReg + RF + LightGBM peers | New baseline = new function; ladder stays | Sklearn/LGBM estimators stay substitutable via `predict_proba` | No SHAP/UI | Train depends on `run_baselines` |
| `training/train.py` | Fit default XGB + Optuna; write artifacts | Optuna search space can grow without serve changes | Best model still `predict_proba` | Does not own HITL copy | Writes files; UI never imports Optuna |
| `training/calibrate.py` | Fit/persist probability map | Method `isotonic`/`sigmoid` | `ProbabilityCalibrator` honours `Calibrator` | Transform-only | Scorer depends on Protocol, not sklearn class |
| `evaluation/metrics.py` | Metric dict helper | Extra keys without changing plots | — | No I/O | Train/eval share one helper |
| `evaluation/evaluate.py` | Holdout plots + τ sweep | New plots without retraining | — | Not a trainer | Reads artifacts |
| `evaluation/slices.py` | Educational `plan_tier` slices | New slice keys without claiming fairness | — | Slice report ≠ DPIA API | Evaluate calls slices |
| `evaluation/benchmark.py` | Latency only | — | — | No training | Reads serve path |
| **`serving/scoring.py`** | `CalibratedScorer`: raw → calibrated → display | New calibrator without UI changes | Dummy / LogReg / XGB all `predict_proba` | Tiny class: `raw_positive` + `score` | **Depends on `ProbabilisticClassifier` + `Calibrator` Protocols** (DIP) |
| `serving/infer.py` | Load bundle + one-row predict | — | Any Protocol-satisfying model | CLI wrapper stays thin | Uses scorer + transformer |
| `serving/explain.py` | Top drivers only | Swap SHAP vs gain without packet rewrite | — | Explain ≠ decide | Packet depends on explain helper |
| `serving/policy.py` | Risk band + HITL action; `auto_action: none` | New bands without retraining | `HitlDecisionPolicy` honours `DecisionPolicy` | Policy is not a model | Packet/UI depend on policy Protocol |
| `serving/packet.py` | Assemble validate → score → HITL JSON | Extra packet fields without trainer changes | — | Packet ≠ Streamlit | App depends on `build_decision_packet` |
| `serving/drift.py` | Lite distribution check | — | — | Not a trainer | CLI only |
| `app/streamlit_app.py` | Serve-only adapter | UI can change without retraining | — | Forms ≠ Optuna | Calls serving helpers only |
| `docs_gen.py` | Model card + dictionary from JSON/schema | New columns appear when config/schema grow | — | Docs ≠ train | Reads metrics + schema |

**`serving/scoring.py` in one sentence:** SRP = turn an estimator + optional calibrator into probability vectors; OCP = new models/calibrators without editing Streamlit; LSP = Dummy/LogReg/XGB are interchangeable via `predict_proba`; ISP = no fat “ModelService”; DIP = `CalibratedScorer` depends on Protocols in `protocols.py`, not on `XGBClassifier`.

Related: [src/retention_radar/protocols.py](src/retention_radar/protocols.py)

## Related docs

- [MODEL_CARD.md](MODEL_CARD.md) · [docs/data-dictionary.md](docs/data-dictionary.md)
- [docs/BEST_PRACTICES.md](docs/BEST_PRACTICES.md) · [docs/data-foundation-lakehouse.md](docs/data-foundation-lakehouse.md)
- [src/retention_radar/protocols.py](src/retention_radar/protocols.py)
