"""Analysis behind the published write-up: every number that is not a plain metric.

Writes ``<artifacts>/analysis.json`` (``make docs-results`` copies it to
``results/analysis.json``):

* ranking: paired bootstrap of logistic regression vs the tuned XGBoost (test AUC)
* calibration: Platt (used) vs isotonic (rejected) on the same validation fit
* the queue: lapse rate by score decile, calibration by quintile, top-10% capture,
  and the operating point at τ
* the holdout: lapse rate among eligible test subscribers, and how many renewals it
  takes to measure each playbook's *assumed* effect with a 10% holdout

Run after train + evaluate:
    python -m retention_radar.cli.analysis
"""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from retention_radar import config
from retention_radar.data.ingest import load_users
from retention_radar.features.transform import prepare_xy
from retention_radar.serving.policy import decide, risk_band
from retention_radar.training.baselines import train_logreg
from retention_radar.training.calibrate import fit_calibrator, load_calibrator
from retention_radar.training.split import stratified_train_val_test

Z_ALPHA = 1.959964  # two-sided 5%
Z_BETA = 0.841621  # 80% power


def holdout_sizes(p_control: float, relative_effect: float, holdout_pct: int) -> dict:
    """Sample sizes to detect ``p_control → p_control × (1 - effect)`` with a k:1 split.

    With a holdout share h, treated = (1-h)/h × holdout. Solves the two-proportion
    z-test for the holdout size, then scales to treated and total eligible.
    """
    p_t = p_control * (1 - relative_effect)
    ratio = (100 - holdout_pct) / holdout_pct  # treated per held-out subscriber
    var = p_control * (1 - p_control) + p_t * (1 - p_t) / ratio
    n_holdout = math.ceil((Z_ALPHA + Z_BETA) ** 2 * var / (p_control - p_t) ** 2)
    return {
        "p_control": round(p_control, 4),
        "p_treated": round(p_t, 4),
        "relative_effect": relative_effect,
        "holdout_pct": holdout_pct,
        "n_holdout": n_holdout,
        "n_treated": math.ceil(n_holdout * ratio),
        "n_eligible": math.ceil(n_holdout * 100 / holdout_pct),
    }


def _bootstrap_auc_diff(y, a, b, n: int = 2000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    d = np.asarray(diffs)
    return {
        "n_resamples": int(len(d)),
        "mean_diff": round(float(d.mean()), 4),
        "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
        "share_first_ahead": round(float((d > 0).mean()), 3),
    }


def main() -> dict:
    users = load_users()
    X, y = prepare_xy(users)
    X_tr, X_va, X_te, y_tr, y_va, y_te = stratified_train_val_test(X, y)
    y_va, y_te = np.asarray(y_va), np.asarray(y_te)
    bundle = joblib.load(config.MODEL_PATH)
    model, names = bundle["model"], bundle["feature_names"]
    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    tau = float(metrics["best_f1_threshold"])

    raw_va = model.predict_proba(X_va[names])[:, 1]
    raw_te = model.predict_proba(X_te[names])[:, 1]
    cal = load_calibrator(config.CALIBRATOR_PATH)
    p_te = cal.transform(raw_te)

    # Ranking: logistic regression vs the tuned booster, paired bootstrap on test.
    logreg = train_logreg(X_tr, y_tr)
    lr_te = logreg.predict_proba(X_te)[:, 1]
    coefs = dict(zip(X.columns, logreg.named_steps["clf"].coef_[0]))
    top_coefs = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:6]

    # Calibration: isotonic (rejected) vs Platt (used), both fit on validation.
    iso = fit_calibrator(y_va, raw_va, "isotonic").transform(raw_te)
    platt = fit_calibrator(y_va, raw_va, "sigmoid").transform(raw_te)

    # The queue.
    d = pd.DataFrame({"p": p_te, "y": y_te})
    d["decile"] = 10 - pd.qcut(d["p"].rank(method="first"), 10, labels=False)
    d["quintile"] = pd.qcut(d["p"].rank(method="first"), 5, labels=False) + 1
    deciles = d.groupby("decile").agg(n=("y", "size"), lapse_rate=("y", "mean"), mean_p=("p", "mean"))
    quintiles = d.groupby("quintile").agg(mean_p=("p", "mean"), lapse_rate=("y", "mean"))
    flagged = p_te >= tau

    # The holdout: what the policy sends on test, and the eligible lapse rate.
    records = users.loc[X_te.index].to_dict(orient="records")
    would = [decide(float(p), tau, risk_band(float(p)), r).get("would_have_sent") for p, r in zip(p_te, records)]
    eligible_rate = float(y_te[flagged].mean())
    share = {pb: sum(w == pb for w in would) / len(would) for pb in config.PLAYBOOKS}
    power = {}
    for pb in ("limit_reset", "cancel_flow_discount"):
        sizes = holdout_sizes(eligible_rate, config.PLAYBOOKS[pb]["effect"], config.HOLDOUT_PCT)
        sizes["share_of_renewals_routed_here"] = round(share[pb], 4)
        sizes["renewals_needed"] = math.ceil(sizes["n_eligible"] / share[pb]) if share[pb] else None
        power[pb] = sizes

    out = {
        "split": {
            "lapses": {
                "train": int(np.asarray(y_tr).sum()),
                "validation": int(y_va.sum()),
                "test": int(y_te.sum()),
            },
        },
        "ranking": {
            "logreg_test_auc": round(float(roc_auc_score(y_te, lr_te)), 4),
            "xgb_tuned_test_auc": round(float(roc_auc_score(y_te, raw_te)), 4),
            "bootstrap_logreg_minus_xgb": _bootstrap_auc_diff(y_te, lr_te, raw_te),
            "logreg_top_standardised_coefficients": {k: round(float(v), 3) for k, v in top_coefs},
        },
        "calibration": {
            "brier_base_rate_only": round(float(brier_score_loss(y_te, np.full(len(y_te), np.asarray(y_tr).mean()))), 4),
            "brier_raw": round(float(brier_score_loss(y_te, raw_te)), 4),
            "brier_platt": round(float(brier_score_loss(y_te, platt)), 4),
            "brier_isotonic": round(float(brier_score_loss(y_te, iso)), 4),
            "distinct_values_platt": int(len(np.unique(platt))),
            "distinct_values_isotonic": int(len(np.unique(iso))),
            "isotonic_zero_scores": int((iso == 0).sum()),
            "max_calibrated_p": round(float(p_te.max()), 4),
        },
        "queue": {
            "base_rate_test": round(float(y_te.mean()), 4),
            "top_10pct_lapse_rate": round(float(deciles.loc[1, "lapse_rate"]), 4),
            "top_10pct_share_of_lapses": round(float(d.loc[d.decile == 1, "y"].sum() / y_te.sum()), 4),
            "at_tau": {
                "tau": tau,
                "flagged_share": round(float(flagged.mean()), 4),
                "precision": round(float(y_te[flagged].mean()), 4),
                "recall": round(float(y_te[flagged].sum() / y_te.sum()), 4),
            },
            "lapse_rate_by_decile": {int(k): round(float(v), 4) for k, v in deciles["lapse_rate"].items()},
            "calibration_by_quintile": {
                int(k): {"mean_p": round(float(r.mean_p), 4), "lapse_rate": round(float(r.lapse_rate), 4)}
                for k, r in quintiles.iterrows()
            },
        },
        "holdout": {
            "eligible_lapse_rate_test": round(eligible_rate, 4),
            "note": "Sizes to detect each playbook's ASSUMED effect at 95% / 80% power with the configured holdout.",
            "power": power,
        },
    }
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.ARTIFACTS_DIR / "analysis.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {path}")
    return out


if __name__ == "__main__":
    main()
