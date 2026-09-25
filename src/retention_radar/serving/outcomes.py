"""Outcome write-back — close the predict → act → outcome loop.

Joins the HITL review log (``hitl_log``) to churn labels observed *after* the
review, then summarises observed churn per risk band and per action taken.
Read-only on models: this is a report for humans, not a retraining trigger.

Labels CSV contract: ``user_id``, ``churned`` (0/1) and optional ``observed_at``
(ISO date/time). When ``observed_at`` is present, only labels observed at or
after the review ``timestamp`` count, so a label can't leak backwards in time.
``data/raw/users.csv`` and lakehouse gold both satisfy the contract (no
``observed_at``), which is enough for the teaching demo.

Examples:
    python -m retention_radar.cli.hitl_outcomes \\
        --log artifacts/hitl_review_log.csv \\
        --labels data/raw/users.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from retention_radar import config
from retention_radar.serving.hitl_log import DEFAULT_LOG_PATH, HITL_LOG_COLUMNS

OUTCOME_COLUMNS = HITL_LOG_COLUMNS + ["churned", "observed_at"]
DEFAULT_OUTCOMES_CSV = config.ARTIFACTS_DIR / "hitl_outcomes.csv"


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def join_outcomes(log_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join review rows to labels; ``churned`` is NaN when not yet observed."""
    missing_log = [c for c in HITL_LOG_COLUMNS if c not in log_df.columns]
    if missing_log:
        raise ValueError(f"HITL log missing columns: {missing_log}")
    for col in ("user_id", "churned"):
        if col not in labels_df.columns:
            raise ValueError(f"Labels CSV missing column: {col}")

    log_df = log_df[HITL_LOG_COLUMNS].copy()
    log_df["user_id"] = log_df["user_id"].astype(str)
    labels = labels_df[["user_id", "churned"]].copy()
    labels["user_id"] = labels["user_id"].astype(str)
    labels["observed_at"] = (
        labels_df["observed_at"].fillna("").astype(str)
        if "observed_at" in labels_df.columns
        else ""
    )
    # One label per user: the latest observation wins.
    if "observed_at" in labels_df.columns:
        labels = labels.assign(_obs=_to_utc(labels["observed_at"])).sort_values("_obs")
        labels = labels.drop(columns="_obs")
    labels = labels.drop_duplicates("user_id", keep="last")

    joined = log_df.merge(labels, on="user_id", how="left")
    if "observed_at" in labels_df.columns:
        too_early = _to_utc(joined["observed_at"]) < _to_utc(joined["timestamp"])
    joined["churned"] = pd.to_numeric(joined["churned"], errors="coerce")
    if "observed_at" in labels_df.columns:
        joined.loc[too_early, "churned"] = float("nan")
        joined.loc[too_early, "observed_at"] = ""
    joined["observed_at"] = joined["observed_at"].fillna("")
    return joined[OUTCOME_COLUMNS]


def _group_summary(df: pd.DataFrame, key: str) -> list[dict[str, Any]]:
    rows = []
    for value, grp in df.groupby(key, dropna=False, sort=True):
        labelled = grp["churned"].dropna()
        rows.append(
            {
                key: "" if pd.isna(value) else str(value),
                "n_reviewed": int(len(grp)),
                "n_labelled": int(len(labelled)),
                "observed_churn_rate": (
                    round(float(labelled.mean()), 4) if len(labelled) else None
                ),
                "mean_p_cal": round(float(pd.to_numeric(grp["p_cal"]).mean()), 4),
            }
        )
    return rows


def summarize_outcomes(joined: pd.DataFrame) -> dict[str, Any]:
    """Observed churn vs calibrated P(churn) by band, and by action actually taken."""
    df = joined.copy()
    df["action_taken"] = df["action_taken"].fillna("").astype(str)
    blank = df["action_taken"].str.strip() == ""
    df.loc[blank, "action_taken"] = "(not recorded)"
    labelled = df["churned"].notna()
    agreed = df["action_taken"] == df["action_suggested"].fillna("").astype(str)
    return {
        "n_reviewed": int(len(df)),
        "n_labelled": int(labelled.sum()),
        "observed_churn_rate": (
            round(float(df.loc[labelled, "churned"].mean()), 4) if labelled.any() else None
        ),
        "reviewer_agreement_rate": round(float(agreed.mean()), 4) if len(df) else None,
        "by_band": _group_summary(df, "band"),
        "by_action_taken": _group_summary(df, "action_taken"),
        "auto_action": "none",
        "note": (
            "Descriptive only: action → churn differences are confounded by who "
            "got which action; this is not an uplift / causal estimate."
        ),
    }


def write_outcomes(
    log_path: Path,
    labels_path: Path,
    out_csv: Path | None = None,
    out_json: Path | None = None,
) -> dict[str, Any]:
    """Load log + labels → write joined CSV and summary JSON → return summary."""
    log_path, labels_path = Path(log_path), Path(labels_path)
    if not log_path.exists():
        raise FileNotFoundError(f"HITL review log not found: {log_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_path}")

    log_df = pd.read_csv(log_path, dtype={"user_id": str}, keep_default_na=False)
    labels_df = pd.read_csv(labels_path, dtype={"user_id": str})
    joined = join_outcomes(log_df, labels_df)
    summary = summarize_outcomes(joined)
    summary["log_path"] = str(log_path)
    summary["labels_path"] = str(labels_path)

    out_csv = Path(out_csv) if out_csv else DEFAULT_OUTCOMES_CSV
    out_json = Path(out_json) if out_json else out_csv.with_suffix(".json")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out_csv, index=False)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["out_csv"] = str(out_csv)
    summary["out_json"] = str(out_json)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Join the HITL review log to later churn labels (outcome write-back)"
    )
    parser.add_argument(
        "--log", type=str, default=None, help=f"HITL review log (default: {DEFAULT_LOG_PATH})"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Labels CSV with user_id, churned[, observed_at] (default: users.csv)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help=f"Joined CSV (default: {DEFAULT_OUTCOMES_CSV})"
    )
    parser.add_argument("--json", type=str, default=None, help="Summary JSON path")
    args = parser.parse_args(argv)

    from retention_radar.data.ingest import resolve_users_csv

    log_path = Path(args.log) if args.log else DEFAULT_LOG_PATH
    labels_path = Path(args.labels) if args.labels else resolve_users_csv()
    try:
        summary = write_outcomes(log_path, labels_path, args.out, args.json)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Joined {summary['n_reviewed']} review rows → {summary['out_csv']}")
    print(f"Summary → {summary['out_json']}")
    print(
        f"  labelled={summary['n_labelled']}  "
        f"observed_churn_rate={summary['observed_churn_rate']}  "
        f"reviewer_agreement={summary['reviewer_agreement_rate']}"
    )
    for row in summary["by_band"]:
        print(
            f"  band={row['band']:<7} n={row['n_reviewed']:<4} "
            f"churn={row['observed_churn_rate']}  mean_p_cal={row['mean_p_cal']}"
        )


if __name__ == "__main__":
    main()
