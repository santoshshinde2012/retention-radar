"""Batch-score a gold feature CSV into a ranked HITL review queue.

Every row is validated with the same contract as single-record serving
(``packet.validate_payload``). Valid rows are scored in one vectorised pass and
written as a queue sorted by calibrated risk (``rank`` 1 = review first); invalid
rows are never scored and go to a rejects file with the reasons, so one bad row
cannot sink (or silently pollute) a weekly job. ``auto_action`` is always ``none``.

Examples:
    python -m retention_radar.cli.batch_score \\
        --csv data/external/churn_user_features.csv \\
        --out artifacts/predictions/scores.csv
    python -m retention_radar.cli.batch_score \\
        --csv data/use_cases/weekly_batch.csv --scored-at 2026-09-28T09:00:00Z
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
from retention_radar.features.transform import encode_plan_tier
from retention_radar.serving.packet import validate_payload
from retention_radar.serving.policy import HitlDecisionPolicy, risk_band
from retention_radar.serving.scoring import CalibratedScorer
from retention_radar.training.calibrate import load_calibrator

SCORE_COLUMNS = [
    "rank",
    "user_id",
    "p_raw",
    "p_cal",
    "band",
    "hitl_action",
    "model_version",
    "scored_at",
    "auto_action",
]
REJECT_COLUMNS = ["row_number", "user_id", "errors"]


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


def _native(v: Any) -> Any:
    return v.item() if hasattr(v, "item") else v


def _row_dict(row: pd.Series) -> dict[str, Any]:
    """Row → plain-Python inference payload (the 24 contract fields only).

    Label / lake-metadata columns (``churned``, ``city``, …) are ignored so gold
    and training CSVs can be scored as-is; empty or NaN cells are dropped and then
    reported by validation as missing keys.
    """
    out = {}
    for k in config.INFERENCE_REQUIRED_KEYS:
        if k not in row.index:
            continue
        v = row[k]
        if pd.isna(v) or (isinstance(v, str) and not v.strip()):
            continue
        out[k] = _native(v)
    return out


def _coerce_numeric(payload: dict[str, Any]) -> dict[str, Any]:
    """CSV cells arrive as strings when a column has any text; parse numbers back."""
    out = dict(payload)
    for key in config.FEATURE_RANGES:
        v = out.get(key)
        if isinstance(v, str):
            try:
                out[key] = float(v)
            except ValueError:
                pass  # left as text → validate_payload reports "must be numeric"
    for key in ("user_id", "user_name", "plan_tier"):
        if key in out and not isinstance(out[key], str):
            out[key] = str(out[key])
    return out


def split_valid_payloads(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate raw records → (valid 24-field payloads, rejects with reasons)."""
    valid: list[dict[str, Any]] = []
    rejects: list[dict[str, Any]] = []
    for i, raw in enumerate(records, start=1):
        payload = _coerce_numeric(
            {
                k: raw[k]
                for k in config.INFERENCE_REQUIRED_KEYS
                if k in raw and raw[k] is not None and not (isinstance(raw[k], str) and not raw[k].strip())
            }
        )
        v = validate_payload(payload)
        if v["ok"]:
            valid.append(payload)
        else:
            rejects.append(
                {"row_number": i, "user_id": str(payload.get("user_id", "")), "errors": " | ".join(v["errors"])}
            )
    return valid, rejects


def split_valid_rows(df: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Validate every CSV row → (valid payloads, rejects frame with reasons)."""
    valid, rejects = split_valid_payloads([_row_dict(row) for _, row in df.iterrows()])
    return valid, pd.DataFrame.from_records(rejects, columns=REJECT_COLUMNS)


def score_payloads(
    payloads: list[dict[str, Any]],
    model_bundle: dict,
    calibrator=None,
    metrics: dict | None = None,
    scored_at: str | None = None,
) -> pd.DataFrame:
    """Score validated payloads in one pass → queue sorted by risk (rank 1 first)."""
    metrics = metrics if metrics is not None else load_metrics()
    when = scored_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not payloads:
        return pd.DataFrame(columns=SCORE_COLUMNS)

    threshold = float(metrics.get("best_f1_threshold", 0.5))
    model_version = resolve_model_version(metrics)
    policy = HitlDecisionPolicy()
    feature_names = model_bundle["feature_names"]

    frame = encode_plan_tier(pd.DataFrame(payloads))
    X = frame[feature_names].apply(pd.to_numeric, errors="raise")
    raw_arr, cal_arr, display_arr = CalibratedScorer(model_bundle["model"], calibrator).score(X)

    records: list[dict[str, Any]] = []
    for i, payload in enumerate(payloads):
        display = float(display_arr[i])
        band = risk_band(display)
        records.append(
            {
                "user_id": str(payload.get("user_id", "")),
                "p_raw": round(float(raw_arr[i]), 6),
                "p_cal": round(float(cal_arr[i]) if cal_arr is not None else display, 6),
                "band": band,
                "hitl_action": policy.decide(display, threshold, band)["action"],
                "model_version": model_version,
                "scored_at": when,
                "auto_action": "none",
            }
        )
    scores = pd.DataFrame.from_records(records)
    # Queue order: highest calibrated risk first. Isotonic calibration is a step
    # function (many users share a plateau), so raw P(churn) orders users inside a
    # plateau; user_id makes the order fully deterministic.
    scores = scores.sort_values(
        ["p_cal", "p_raw", "user_id"], ascending=[False, False, True], kind="mergesort"
    )
    scores.insert(0, "rank", range(1, len(scores) + 1))
    return scores.reset_index(drop=True)[SCORE_COLUMNS]


def score_feature_frame(
    df: pd.DataFrame,
    model_bundle: dict,
    calibrator=None,
    metrics: dict | None = None,
    scored_at: str | None = None,
) -> pd.DataFrame:
    """Validate + score a frame. Rejects are attached as ``result.attrs['rejected']``."""
    valid, rejects = split_valid_rows(df)
    scores = score_payloads(valid, model_bundle, calibrator, metrics, scored_at)
    scores.attrs["rejected"] = rejects
    return scores


def rejects_path_for(out_csv: Path) -> Path:
    return out_csv.with_name(f"{out_csv.stem}_rejected.csv")


def score_csv(
    csv_path: Path,
    out_csv: Path | None = None,
    out_jsonl: Path | None = None,
    model_path: Path | None = None,
    calibrator_path: Path | None = None,
    metrics_path: Path | None = None,
    scored_at: str | None = None,
) -> pd.DataFrame:
    """Load CSV → validate → score → write the queue (+ ``*_rejected.csv`` when needed)."""
    model_path = model_path or config.MODEL_PATH
    calibrator_path = calibrator_path or config.CALIBRATOR_PATH
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")

    bundle = joblib.load(model_path)
    calibrator = load_calibrator(calibrator_path)
    metrics = load_metrics(metrics_path)
    # dtype=str for identity columns keeps ids like "007" intact.
    df = pd.read_csv(csv_path, dtype={"user_id": str, "user_name": str, "plan_tier": str})
    scores = score_feature_frame(df, bundle, calibrator=calibrator, metrics=metrics, scored_at=scored_at)
    rejects = scores.attrs["rejected"]

    if out_csv is None and out_jsonl is None:
        out_csv = config.ARTIFACTS_DIR / "predictions" / "scores.csv"

    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out_csv, index=False)
        rej_path = rejects_path_for(out_csv)
        if len(rejects):
            rejects.to_csv(rej_path, index=False)
        elif rej_path.exists():
            rej_path.unlink()  # stale rejects from an earlier run would mislead

    if out_jsonl is not None:
        out_jsonl = Path(out_jsonl)
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(out_jsonl, "w", encoding="utf-8") as fout:
            for rec in scores.to_dict(orient="records"):
                fout.write(json.dumps({k: _native(v) for k, v in rec.items()}) + "\n")

    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch-score a feature CSV into a ranked HITL queue (auto_action=none)"
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
        help="Output queue CSV (default: artifacts/predictions/scores.csv); "
        "rejects go to <out>_rejected.csv",
    )
    parser.add_argument("--jsonl", type=str, default=None, help="Optional JSONL output path")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--calibrator", type=str, default=None)
    parser.add_argument(
        "--scored-at",
        type=str,
        default=None,
        help="Override the scored_at stamp (ISO-8601 UTC), e.g. for a reproducible demo",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any row was rejected by validation",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv) if args.csv else config.LAKEHOUSE_FEATURES_CSV
    if not csv_path.exists():
        raise SystemExit(
            f"Feature CSV not found: {csv_path}. "
            "Pass --csv path/to/features.csv or sync lakehouse gold."
        )

    out_csv = Path(args.out) if args.out else (config.ARTIFACTS_DIR / "predictions" / "scores.csv")
    out_jsonl = Path(args.jsonl) if args.jsonl else None
    model_path = Path(args.model) if args.model else config.MODEL_PATH
    calibrator_path = Path(args.calibrator) if args.calibrator else config.CALIBRATOR_PATH

    scores = score_csv(
        csv_path,
        out_csv=out_csv,
        out_jsonl=out_jsonl,
        model_path=model_path,
        calibrator_path=calibrator_path,
        scored_at=args.scored_at,
    )
    rejects = scores.attrs["rejected"]
    print(f"Scored {len(scores)} rows → {out_csv} (queue order: rank 1 = highest risk)")
    if out_jsonl:
        print(f"Also wrote JSONL → {out_jsonl}")
    if len(rejects):
        print(f"Rejected {len(rejects)} row(s) by validation → {rejects_path_for(out_csv)}")
        for rec in rejects.head(10).to_dict(orient="records"):
            print(f"  row {rec['row_number']} ({rec['user_id'] or 'no user_id'}): {rec['errors']}")
    if len(scores):
        print(scores["hitl_action"].value_counts().to_string())
        print(f"auto_action unique: {sorted(scores['auto_action'].unique().tolist())}")
    return 1 if (args.strict and len(rejects)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
