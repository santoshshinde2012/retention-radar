"""Batch-score a gold feature CSV with the committed serve bundle.

Writes compact score rows (CSV and/or JSONL). ``auto_action`` is always ``none``.

Examples:
    python -m retention_radar.cli.batch_score \\
        --csv data/external/churn_user_features.csv \\
        --out artifacts/predictions/scores.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from retention_radar import config
from retention_radar.features.transform import row_to_feature_frame
from retention_radar.serving.policy import HitlDecisionPolicy, risk_band
from retention_radar.serving.scoring import CalibratedScorer
from retention_radar.training.calibrate import load_calibrator

SCORE_COLUMNS = [
    "user_id",
    "p_raw",
    "p_cal",
    "band",
    "hitl_action",
    "model_version",
    "scored_at",
    "auto_action",
]


def resolve_model_version(metrics: dict | None = None) -> str:
    """Stable teaching serve label (seed-42 calibrated XGB by default)."""
    if metrics and metrics.get("model_version"):
        return str(metrics["model_version"])
    seed = config.RANDOM_SEED
    if metrics and "random_seed" in metrics:
        seed = int(metrics["random_seed"])
    return f"seed{seed}-churn_xgb"


def load_metrics(path: Path | None = None) -> dict:
    p = path or config.METRICS_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _row_dict(row: pd.Series) -> dict[str, Any]:
    return {k: row[k] for k in row.index if pd.notna(row[k])}


def score_feature_frame(
    df: pd.DataFrame,
    model_bundle: dict,
    calibrator=None,
    metrics: dict | None = None,
    scored_at: str | None = None,
) -> pd.DataFrame:
    """Score every row; return a DataFrame with SCORE_COLUMNS (+ auto_action)."""
    metrics = metrics if metrics is not None else load_metrics()
    threshold = float(metrics.get("best_f1_threshold", 0.5))
    model_version = resolve_model_version(metrics)
    policy = HitlDecisionPolicy()
    scorer = CalibratedScorer(model_bundle["model"], calibrator)
    feature_names = model_bundle["feature_names"]
    when = scored_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        payload = _row_dict(row)
        user_id = str(payload.get("user_id", ""))
        X = row_to_feature_frame(payload)[feature_names]
        raw_arr, cal_arr, display_arr = scorer.score(X)
        p_raw = float(raw_arr[0])
        p_cal = float(cal_arr[0]) if cal_arr is not None else p_raw
        display = float(display_arr[0])
        band = risk_band(display)
        action = policy.decide(display, threshold, band)
        records.append(
            {
                "user_id": user_id,
                "p_raw": round(p_raw, 6),
                "p_cal": round(p_cal, 6),
                "band": band,
                "hitl_action": action["action"],
                "model_version": model_version,
                "scored_at": when,
                "auto_action": "none",
            }
        )
    return pd.DataFrame.from_records(records, columns=SCORE_COLUMNS)


def score_csv(
    csv_path: Path,
    out_csv: Path | None = None,
    out_jsonl: Path | None = None,
    model_path: Path | None = None,
    calibrator_path: Path | None = None,
    metrics_path: Path | None = None,
) -> pd.DataFrame:
    """Load CSV → score → optionally write CSV/JSONL under artifacts/predictions/."""
    model_path = model_path or config.MODEL_PATH
    calibrator_path = calibrator_path or config.CALIBRATOR_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    bundle = joblib.load(model_path)
    calibrator = load_calibrator(calibrator_path)
    metrics = load_metrics(metrics_path)
    df = pd.read_csv(csv_path)
    scores = score_feature_frame(df, bundle, calibrator=calibrator, metrics=metrics)

    if out_csv is None and out_jsonl is None:
        out_csv = config.ARTIFACTS_DIR / "predictions" / "scores.csv"

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_csv, index=False)

    if out_jsonl is not None:
        out_jsonl = Path(out_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as fout:
            for rec in scores.to_dict(orient="records"):
                fout.write(json.dumps(rec) + "\n")

    return scores


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Batch-score a gold feature CSV (HITL only; auto_action=none)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Feature CSV (default: data/external/churn_user_features.csv if present)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output scores CSV (default: artifacts/predictions/scores.csv)",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default=None,
        help="Optional JSONL output path",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--calibrator", type=str, default=None)
    args = parser.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else config.LAKEHOUSE_FEATURES_CSV
    if not csv_path.exists():
        raise SystemExit(
            f"Feature CSV not found: {csv_path}. "
            "Pass --csv path/to/features.csv or sync lakehouse gold."
        )

    out_csv = (
        Path(args.out)
        if args.out
        else (config.ARTIFACTS_DIR / "predictions" / "scores.csv")
    )
    out_jsonl = Path(args.jsonl) if args.jsonl else None
    model_path = Path(args.model) if args.model else config.MODEL_PATH
    calibrator_path = Path(args.calibrator) if args.calibrator else config.CALIBRATOR_PATH

    scores = score_csv(
        csv_path,
        out_csv=out_csv,
        out_jsonl=out_jsonl,
        model_path=model_path,
        calibrator_path=calibrator_path,
    )
    print(f"Scored {len(scores)} rows → {out_csv}")
    if out_jsonl:
        print(f"Also wrote JSONL → {out_jsonl}")
    print(scores["band"].value_counts().to_string())
    print(f"auto_action unique: {sorted(scores['auto_action'].unique().tolist())}")


if __name__ == "__main__":
    main()
