"""Feature-drift check vs train reference stats (FOSS demo).

Compares a new CSV (default: ``resolve_users_csv()``, so ``CHURN_DATA_SOURCE``
picks lakehouse gold or synthetic ``data/raw/users.csv``) to
``models/feature_stats.json``. Per feature it reports:

- **PSI** (population stability index) over training-decile bins stored in
  ``feature_stats.json["<col>"]["psi_bins"]``. Severity is driven by PSI with the
  usual rule of thumb: < 0.10 stable, 0.10–0.25 moderate shift, ≥ 0.25 major shift.
- **SMD** (standardised mean difference): ``|cur_mean - ref_mean| / ref_std``, i.e.
  how many training standard deviations the mean moved. An effect size, not a test.
- **mean z (SE)**: the same shift divided by the standard error
  ``ref_std / sqrt(n_current)``. With thousands of rows this flags tiny, harmless
  shifts, so it is reported for context and never drives severity.

Older ``feature_stats.json`` files without ``psi_bins`` fall back to SMD.

Exit codes
----------
Always exits **0** for this teaching demo unless ``--strict`` is passed
and severity is ``severe``. Gate CI / pipelines with ``--strict``.

Examples::

    python -m retention_radar.cli.drift_check
    python -m retention_radar.cli.drift_check --strict
    python -m retention_radar.cli.drift_check --csv data/external/churn_user_features.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from retention_radar import config
from retention_radar.data.ingest import resolve_users_csv
from retention_radar.features.transform import encode_plan_tier


def load_feature_stats(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_current_values(csv_path: Path) -> dict[str, list[float]]:
    """Numeric values per model feature from the current CSV (NaNs dropped)."""
    encoded = encode_plan_tier(pd.read_csv(csv_path))
    return {
        col: pd.to_numeric(encoded[col], errors="coerce").dropna().to_numpy()
        for col in config.MODEL_FEATURE_COLUMNS
        if col in encoded.columns
    }


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


PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
PSI_EPS = 1e-4


def population_stability_index(values, edges: list[float], ref_frac: list[float]) -> float:
    """PSI of ``values`` against training bins (``edges`` + exact training shares)."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    idx = np.searchsorted(np.asarray(edges, dtype=float), arr, side="right")
    cur = np.bincount(idx, minlength=len(edges) + 1) / arr.size
    ref = np.asarray(ref_frac, dtype=float)
    cur = np.clip(cur, PSI_EPS, None)
    ref = np.clip(ref, PSI_EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def compare_stats(
    reference: dict,
    current: dict,
    smd_threshold: float = 0.5,
    current_values: dict | None = None,
) -> dict:
    """Per-feature PSI (+ SMD and SE-based z for context) vs the train reference.

    ``current_values`` maps feature → array of current values; it is needed for PSI.
    Without it (or without ``psi_bins`` in the reference) severity falls back to SMD.
    """
    import math

    features = []
    use_psi = current_values is not None and all(
        "psi_bins" in reference.get(c, {}) for c in config.MODEL_FEATURE_COLUMNS if c in reference
    )
    n_flagged = 0
    n_major = 0
    max_psi = 0.0
    max_smd = 0.0
    for col in config.MODEL_FEATURE_COLUMNS:
        if col not in reference or col not in current:
            continue
        ref = reference[col]
        cur = current[col]
        ref_mean = float(ref["mean"])
        ref_std = float(ref.get("std") or 0.0)
        cur_mean = float(cur["mean"])
        n_cur = int(cur.get("count") or 0)
        delta = cur_mean - ref_mean
        denom = ref_std if ref_std > 1e-9 else 1.0
        smd = abs(delta) / denom
        se = denom / math.sqrt(n_cur) if n_cur > 0 else float("nan")
        mean_z_se = abs(delta) / se if n_cur > 0 else float("nan")
        row = {
            "feature": col,
            "ref_mean": ref_mean,
            "ref_std": ref_std,
            "cur_mean": cur_mean,
            "cur_std": float(cur["std"]),
            "n_current": n_cur,
            "mean_delta": round(delta, 6),
            "smd": round(smd, 4),
            "mean_z_se": round(mean_z_se, 2) if math.isfinite(mean_z_se) else None,
        }
        if use_psi:
            bins = ref["psi_bins"]
            psi = population_stability_index(current_values[col], bins["edges"], bins["ref_frac"])
            row["psi"] = round(psi, 4)
            flagged = psi >= PSI_MODERATE
            n_major += int(psi >= PSI_MAJOR)
            max_psi = max(max_psi, psi)
        else:
            flagged = smd >= smd_threshold
        row["flagged"] = bool(flagged)
        n_flagged += int(flagged)
        max_smd = max(max_smd, smd)
        features.append(row)

    key = "psi" if use_psi else "smd"
    features.sort(key=lambda r: r[key], reverse=True)
    severity = "ok"
    if use_psi:
        if n_major >= 1 or n_flagged >= 5:
            severity = "severe"
        elif n_flagged >= 1:
            severity = "mild"
        note = (
            f"PSI over training-decile bins (moderate ≥ {PSI_MODERATE}, major ≥ {PSI_MAJOR}); "
            "SMD and SE-based mean z are context only. A teaching monitor, not "
            "Evidently / Great Expectations."
        )
    else:
        if n_flagged >= 5 or max_smd >= smd_threshold * 2:
            severity = "severe"
        elif n_flagged >= 1:
            severity = "mild"
        note = (
            "No psi_bins in feature_stats.json (retrain to add them): severity from "
            f"standardised mean difference ≥ {smd_threshold}."
        )
    return {
        "method": "psi" if use_psi else "smd",
        "n_features_compared": len(features),
        "n_flagged": n_flagged,
        "n_major": n_major,
        "max_psi": round(max_psi, 4) if use_psi else None,
        "max_smd": round(max_smd, 4),
        "psi_thresholds": {"moderate": PSI_MODERATE, "major": PSI_MAJOR},
        "smd_threshold": smd_threshold,
        "severity": severity,
        "note": note,
        "features": features,
    }


def print_report(report: dict) -> None:
    head = (
        f"max PSI={report['max_psi']}" if report["method"] == "psi" else f"max SMD={report['max_smd']}"
    )
    print(
        f"Drift check ({report['method']}): severity={report['severity']}  "
        f"flagged={report['n_flagged']}/{report['n_features_compared']}  {head}"
    )
    print(report["note"])
    print("Top features:")
    for row in report["features"][:8]:
        flag = " *" if row["flagged"] else ""
        psi = f"PSI={row['psi']:.3f}  " if "psi" in row else ""
        print(
            f"  {row['feature']}: {psi}SMD={row['smd']:.3f}  "
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
    parser.add_argument(
        "--smd-threshold",
        "--z-threshold",  # deprecated alias: the old "z" was already a standardised mean difference
        dest="smd_threshold",
        type=float,
        default=0.5,
        help="SMD flag threshold, used only when feature_stats.json has no psi_bins",
    )
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
        print(
            f"CSV missing: {csv_path}. Generate it with "
            "`python -m retention_radar.cli.generate_data` (synthetic) or sync lakehouse "
            "gold, or pass --csv path/to/users.csv.",
            file=sys.stderr,
        )
        return 0 if not args.strict else 1

    reference = load_feature_stats(stats_path)
    current = compute_current_stats(csv_path)
    report = compare_stats(
        reference,
        current,
        smd_threshold=args.smd_threshold,
        current_values=load_current_values(csv_path),
    )
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
