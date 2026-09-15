"""Evaluate saved model: ROC, PR, calibration, threshold analysis."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.retention_radar import config
from src.retention_radar.data.ingest import load_users
from src.retention_radar.evaluation.slices import (
    print_slice_report,
    slice_metrics_by_plan_tier,
)
from src.retention_radar.features.transform import prepare_xy
from src.retention_radar.training.calibrate import load_calibrator
from src.retention_radar.training.split import stratified_train_val_test


def load_model(path: Path | None = None):
    payload = joblib.load(path or config.MODEL_PATH)
    return payload["model"], payload["feature_names"]


def threshold_sweep(y_true, y_prob, n_steps: int = 99) -> dict:
    thresholds = np.linspace(0.01, 0.99, n_steps)
    rows = []
    best_f1 = -1.0
    best_t = 0.5
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        p = float(precision_score(y_true, pred, zero_division=0))
        r = float(recall_score(y_true, pred, zero_division=0))
        f = float(f1_score(y_true, pred, zero_division=0))
        rows.append({"threshold": float(t), "precision": p, "recall": r, "f1": f})
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return {"rows": rows, "best_threshold": best_t, "best_f1": best_f1}


def main() -> None:
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_users()
    X, y = prepare_xy(df)
    _, _, X_test, _, _, y_test = stratified_train_val_test(X, y)

    model, feature_names = load_model()
    X_test = X_test[feature_names]
    y_prob_raw = model.predict_proba(X_test)[:, 1]

    calibrator = load_calibrator(config.CALIBRATOR_PATH)
    if calibrator is not None:
        y_prob = calibrator.transform(y_prob_raw)
        used_calibrated = True
    else:
        y_prob = y_prob_raw
        used_calibrated = False

    y_pred = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, y_prob))
    ap = float(average_precision_score(y_test, y_prob))
    brier_raw = float(brier_score_loss(y_test, y_prob_raw))
    brier_cal = float(brier_score_loss(y_test, y_prob))

    metrics = {
        "test_roc_auc": auc,
        "test_average_precision": ap,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "test_brier_raw": brier_raw,
        "test_brier_calibrated": brier_cal,
        "used_calibrated_probs": used_calibrated,
    }
    print("Test metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("\nClassification report (calibrated probs @ 0.5):")
    print(classification_report(y_test, y_pred, digits=3))

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"ROC Curve (AUC={auc:.3f})")
    roc_path = config.ARTIFACTS_DIR / "roc_curve.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    print(f"Saved {roc_path}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (threshold=0.5)")
    cm_path = config.ARTIFACTS_DIR / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Saved {cm_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title(f"Precision–Recall (AP={ap:.3f})")
    pr_path = config.ARTIFACTS_DIR / "pr_curve.png"
    fig.tight_layout()
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    print(f"Saved {pr_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, probs, style in [
        ("raw", y_prob_raw, "o-"),
        ("calibrated", y_prob, "s-"),
    ]:
        frac_pos, mean_pred = calibration_curve(
            y_test, probs, n_bins=10, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, style, label=label)
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability diagram")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    cal_path = config.ARTIFACTS_DIR / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(cal_path, dpi=120)
    plt.close(fig)
    print(f"Saved {cal_path}")

    sweep = threshold_sweep(y_test, y_prob)
    rows = sweep["rows"]
    ts = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ts, [r["precision"] for r in rows], label="precision")
    ax.plot(ts, [r["recall"] for r in rows], label="recall")
    ax.plot(ts, [r["f1"] for r in rows], label="f1", linewidth=2)
    ax.axvline(
        sweep["best_threshold"],
        color="gray",
        linestyle="--",
        label=f"best F1 @ {sweep['best_threshold']:.2f}",
    )
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold analysis (business action)")
    ax.legend(loc="best")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    thr_path = config.ARTIFACTS_DIR / "threshold_f1.png"
    fig.tight_layout()
    fig.savefig(thr_path, dpi=120)
    plt.close(fig)
    print(f"Saved {thr_path}")

    metrics["best_f1_threshold"] = sweep["best_threshold"]
    metrics["best_f1_at_threshold"] = sweep["best_f1"]

    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    metrics["pr_curve_points"] = int(len(prec))

    plan_tiers_test = df.loc[X_test.index, "plan_tier"]
    slice_block = slice_metrics_by_plan_tier(
        y_test, y_prob, plan_tiers_test, threshold=0.5
    )
    print_slice_report(slice_block)

    out = {}
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, encoding="utf-8") as f:
            out = json.load(f)
    out["evaluate_test"] = metrics
    out["brier_raw_test"] = brier_raw
    out["brier_calibrated_test"] = brier_cal
    out["average_precision_test"] = ap
    out["best_f1_threshold"] = sweep["best_threshold"]
    out["best_f1_at_threshold"] = sweep["best_f1"]
    out["slice_metrics_by_plan_tier"] = slice_block
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Updated {config.METRICS_PATH}")

    # Refresh model card *after* evaluate_test / τ land in metrics
    # (train writes the card earlier, before these keys exist).
    from src.retention_radar.docs_gen import write_model_card

    write_model_card(out)
    print(f"Refreshed model card → {config.MODEL_CARD_PATH}")


if __name__ == "__main__":
    main()
