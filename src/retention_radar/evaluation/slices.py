"""Educational segment metrics by plan_tier (NOT a protected-class fairness audit).

Reports precision / recall / AUC on the holdout test set sliced by ``plan_tier``.
This is a **product-segment diagnostic** for teaching — plan tier is a commercial
attribute, not a protected class. Do not treat these numbers as a formal fairness
or disparate-impact audit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


MIN_SAMPLES_FOR_AUC = 20
MIN_POSITIVES_FOR_AUC = 2


def slice_metrics_by_plan_tier(
    y_true,
    y_prob,
    plan_tiers,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute per-plan_tier test metrics.

    Returns a dict suitable for ``metrics.json["slice_metrics_by_plan_tier"]``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    tiers = pd.Series(list(plan_tiers)).astype(str).reset_index(drop=True)

    by_tier: dict[str, Any] = {}
    for tier in sorted(tiers.unique()):
        mask = (tiers == tier).to_numpy()
        n = int(mask.sum())
        yt = y_true[mask]
        yp = y_prob[mask]
        n_pos = int(yt.sum())
        n_neg = n - n_pos
        pred = (yp >= threshold).astype(int)
        block: dict[str, Any] = {
            "n": n,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "churn_rate": float(yt.mean()) if n else None,
            "precision": float(precision_score(yt, pred, zero_division=0)) if n else None,
            "recall": float(recall_score(yt, pred, zero_division=0)) if n else None,
            "roc_auc": None,
            "average_precision": None,
            "enough_samples_for_auc": False,
        }
        if n >= MIN_SAMPLES_FOR_AUC and n_pos >= MIN_POSITIVES_FOR_AUC and n_neg >= 1:
            try:
                block["roc_auc"] = float(roc_auc_score(yt, yp))
                block["average_precision"] = float(average_precision_score(yt, yp))
                block["enough_samples_for_auc"] = True
            except ValueError:
                pass
        by_tier[tier] = block

    return {
        "disclaimer": (
            "Educational segment diagnostics by plan_tier only. "
            "NOT a protected-class fairness audit / DPIA / disparate-impact study."
        ),
        "threshold": float(threshold),
        "min_samples_for_auc": MIN_SAMPLES_FOR_AUC,
        "by_plan_tier": by_tier,
    }


def print_slice_report(slice_block: dict) -> None:
    print("\nSlice metrics by plan_tier (educational — not a fairness audit):")
    print(f"  disclaimer: {slice_block.get('disclaimer')}")
    for tier, block in (slice_block.get("by_plan_tier") or {}).items():
        auc = block.get("roc_auc")
        auc_s = f"{auc:.3f}" if isinstance(auc, float) else "n/a"
        print(
            f"  {tier:12s} n={block['n']:4d}  churn={block['churn_rate']:.3f}  "
            f"P={block['precision']:.3f}  R={block['recall']:.3f}  AUC={auc_s}"
        )


def main() -> None:
    """Recompute plan_tier slices from the active users table + saved model."""
    import json

    import joblib

    from retention_radar import config
    from retention_radar.data.ingest import load_users
    from retention_radar.features.transform import prepare_xy
    from retention_radar.training.calibrate import load_calibrator
    from retention_radar.training.split import stratified_train_val_test

    if not config.MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {config.MODEL_PATH}. Run python -m retention_radar.cli.train first.")

    df = load_users()
    X, y = prepare_xy(df)
    _, _, X_test, _, _, y_test = stratified_train_val_test(X, y)
    payload = joblib.load(config.MODEL_PATH)
    feature_names = payload["feature_names"]
    model = payload["model"]
    X_test = X_test[feature_names]
    y_prob_raw = model.predict_proba(X_test)[:, 1]
    calibrator = load_calibrator(config.CALIBRATOR_PATH)
    y_prob = calibrator.transform(y_prob_raw) if calibrator is not None else y_prob_raw
    plan_tiers_test = df.loc[X_test.index, "plan_tier"]
    block = slice_metrics_by_plan_tier(y_test, y_prob, plan_tiers_test, threshold=0.5)
    print_slice_report(block)

    out: dict = {}
    if config.METRICS_PATH.exists():
        out = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    out["slice_metrics_by_plan_tier"] = block
    config.METRICS_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Updated {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
