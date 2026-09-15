"""Lite feature-drift check vs train reference stats (FOSS demo).

Compares a new CSV (default: ``resolve_users_csv()``, so ``CHURN_DATA_SOURCE``
picks lakehouse gold or synthetic ``data/raw/users.csv``) to
``models/feature_stats.json`` using absolute mean z-scores per feature.

Exit codes
----------
Always exits **0** for this teaching demo unless ``--strict`` is passed
and severity is ``severe``. Gate CI / pipelines with ``--strict``.

Examples::

    python -m src.drift_check
    python -m src.drift_check --strict --z-threshold 3.0
    python -m src.drift_check --csv data/external/churn_user_features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.retention_radar import config
from src.retention_radar.data.ingest import resolve_users_csv
from src.retention_radar.features.transform import encode_plan_tier


def load_feature_stats(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_current_stats(csv_path: Path) -> dict[str, dict]:
    df = pd.read_csv(csv_path)
    encoded = encode_plan_tier(df)
    out: dict[str, dict] = {}
    for col in config.MODEL_FEATURE_COLUMNS:
        if col not in encoded.columns:
            continue
        series = pd.to_numeric(encoded[col], errors="coerce").dropna()
        out[col] = {
            "count": int(series.shape[0]),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }
    return out


def compare_stats(
    reference: dict,
    current: dict,
    z_threshold: float = 3.0,
) -> dict:
    """Per-feature abs mean z-score vs train reference (PSI-lite proxy)."""
    features = []
    n_flagged = 0
    max_abs_z = 0.0
    for col in config.MODEL_FEATURE_COLUMNS:
        if col not in reference or col not in current:
            continue
        ref = reference[col]
        cur = current[col]
        ref_mean = float(ref["mean"])
        ref_std = float(ref.get("std") or 0.0)
        cur_mean = float(cur["mean"])
        denom = ref_std if ref_std > 1e-9 else 1.0
        z = (cur_mean - ref_mean) / denom
        abs_z = abs(z)
        flagged = abs_z >= z_threshold
        if flagged:
            n_flagged += 1
        max_abs_z = max(max_abs_z, abs_z)
        features.append(
            {
                "feature": col,
                "ref_mean": ref_mean,
                "ref_std": ref_std,
                "cur_mean": cur_mean,
                "cur_std": float(cur["std"]),
                "abs_mean_z": round(abs_z, 4),
                "mean_delta": round(cur_mean - ref_mean, 6),
                "flagged": flagged,
            }
        )
    features.sort(key=lambda r: r["abs_mean_z"], reverse=True)
    severity = "ok"
    if n_flagged >= 5 or max_abs_z >= z_threshold * 2:
        severity = "severe"
    elif n_flagged >= 1:
        severity = "mild"
    return {
        "n_features_compared": len(features),
        "n_flagged": n_flagged,
        "max_abs_mean_z": round(max_abs_z, 4),
        "z_threshold": z_threshold,
        "severity": severity,
        "note": (
            "Lite abs-mean-z vs train feature_stats.json — not a production "
            "PSI / Evidently / Great Expectations monitor."
        ),
        "features": features,
    }


def print_report(report: dict) -> None:
    print(
        f"Drift check: severity={report['severity']}  "
        f"flagged={report['n_flagged']}/{report['n_features_compared']}  "
        f"max|z|={report['max_abs_mean_z']}  threshold={report['z_threshold']}"
    )
    print(report["note"])
    print("Top |z| features:")
    for row in report["features"][:8]:
        flag = " *" if row["flagged"] else ""
        print(
            f"  {row['feature']}: |z|={row['abs_mean_z']:.3f}  "
            f"ref_mean={row['ref_mean']:.4g}  cur_mean={row['cur_mean']:.4g}{flag}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lite feature drift check")
    parser.add_argument(
        "--csv", type=str, default=None, help="CSV to compare"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="Train reference feature_stats.json",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output drift report JSON",
    )
    parser.add_argument("--z-threshold", type=float, default=3.0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 when severity == severe (default: always exit 0)",
    )
    args = parser.parse_args(argv)

    stats_path = Path(args.stats) if args.stats else config.FEATURE_STATS_PATH
    csv_path = Path(args.csv) if args.csv else resolve_users_csv()
    if not stats_path.exists():
        print(
            f"Reference stats missing: {stats_path}. Run train first.",
            file=sys.stderr,
        )
        return 0 if not args.strict else 1
    if not csv_path.exists():
        print(f"CSV missing: {csv_path}", file=sys.stderr)
        return 0 if not args.strict else 1

    reference = load_feature_stats(stats_path)
    current = compute_current_stats(csv_path)
    report = compare_stats(reference, current, z_threshold=args.z_threshold)
    report["reference_path"] = str(stats_path)
    report["csv_path"] = str(csv_path)

    out_path = Path(args.out) if args.out else (config.ARTIFACTS_DIR / "drift_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_report(report)
    print(f"Wrote {out_path}")

    if args.strict and report["severity"] == "severe":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
