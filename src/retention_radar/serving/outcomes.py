"""Outcome write-back: close the score → act → outcome loop and measure lift.

Joins the action log (``action_log``) to renewal outcomes observed *after* the
action, then reports two things:

1. Observed lapse rate per risk band (is the calibrated score still honest?).
2. Lift per playbook: lapse rate in the holdout minus lapse rate among the
   subscribers the playbook was sent to, both drawn from the same eligible
   population. This is the number that says whether a playbook works. A churn
   score alone cannot tell you that.

Labels CSV contract: ``user_id``, ``churned`` (0/1) and optional ``observed_at``
(ISO date/time). When ``observed_at`` is present, only labels observed at or
after the action ``timestamp`` count, so a label can't leak backwards in time.

Examples:
    python -m retention_radar.cli.outcomes \\
        --log artifacts/action_log.csv \\
        --labels data/use_cases/renewal_outcomes.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from retention_radar import config
from retention_radar.serving.action_log import ACTION_LOG_COLUMNS, DEFAULT_LOG_PATH
from retention_radar.serving.policy import HOLDOUT, NO_ACTION

OUTCOME_COLUMNS = ACTION_LOG_COLUMNS + ["churned", "observed_at"]
DEFAULT_OUTCOMES_CSV = config.ARTIFACTS_DIR / "outcomes.csv"


def _to_utc(series: pd.Series) -> pd.Series:
    """Parse ISO-8601 (dates or datetimes, mixed forms); unparseable → NaT."""
    return pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")


def join_outcomes(log_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join action rows to labels; ``churned`` is NaN when not yet observed."""
    missing_log = [c for c in ACTION_LOG_COLUMNS if c not in log_df.columns]
    if missing_log:
        raise ValueError(f"action log missing columns: {missing_log}")
    for col in ("user_id", "churned"):
        if col not in labels_df.columns:
            raise ValueError(f"Labels CSV missing column: {col}")

    log_df = log_df[ACTION_LOG_COLUMNS].copy()
    log_df["user_id"] = log_df["user_id"].astype(str).str.strip()
    labels = labels_df[["user_id", "churned"]].copy()
    labels["user_id"] = labels["user_id"].astype(str).str.strip()
    labels["observed_at"] = (
        labels_df["observed_at"].fillna("").astype(str)
        if "observed_at" in labels_df.columns
        else ""
    )
    if "observed_at" in labels_df.columns:
        labels = labels.assign(_obs=_to_utc(labels["observed_at"])).sort_values(
            "_obs", na_position="first", kind="mergesort"
        )
        labels = labels.drop(columns="_obs")
    labels = labels.drop_duplicates("user_id", keep="last")

    joined = log_df.merge(labels, on="user_id", how="left")
    if "observed_at" in labels_df.columns:
        obs, act = _to_utc(joined["observed_at"]), _to_utc(joined["timestamp"])
        too_early = obs.isna() | act.isna() | (obs < act)
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
                "n": int(len(grp)),
                "n_labelled": int(len(labelled)),
                "observed_lapse_rate": (
                    round(float(labelled.mean()), 4) if len(labelled) else None
                ),
                "mean_p_cal": round(float(pd.to_numeric(grp["p_cal"]).mean()), 4),
            }
        )
    return rows


MIN_GROUP_FOR_VERDICT = 30
_Z = 1.96


def _wilson(k: int, n: int) -> tuple[float, float, float]:
    """(p, lower, upper) Wilson score interval. Stays sane at k = 0 or k = n."""
    p = k / n
    denom = 1 + _Z**2 / n
    centre = (p + _Z**2 / (2 * n)) / denom
    half = _Z * math.sqrt(p * (1 - p) / n + _Z**2 / (4 * n * n)) / denom
    return p, centre - half, centre + half


def _newcombe_diff(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p1 - p2 with Newcombe's hybrid-score 95% interval (method 10)."""
    p1, l1, u1 = _wilson(k1, n1)
    p2, l2, u2 = _wilson(k2, n2)
    d = p1 - p2
    lower = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    upper = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lower, upper


def lift_by_playbook(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Holdout lapse rate minus treated lapse rate, per suggested playbook.

    For each playbook, "would have been sent" defines the eligible group: the
    policy records ``would_have_sent`` for holdout rows, and treated rows are those
    whose ``action_taken`` equals the playbook. The 95% interval is Newcombe's
    (Wilson-based), because a normal approximation collapses to zero width when a
    small holdout has no lapses at all. No verdict is given unless both groups have
    at least ``MIN_GROUP_FOR_VERDICT`` subscribers.
    """
    labelled = df[df["churned"].notna()].copy()
    holdout = labelled[labelled["holdout"].astype(str).str.lower() == "true"].copy()
    holdout["_pb"] = holdout["would_have_sent"].astype(str)
    out = []
    for name in config.PLAYBOOKS:
        treated = labelled[labelled["action_taken"] == name]
        control = holdout[holdout["_pb"] == name]
        row: dict[str, Any] = {
            "playbook": name,
            "n_treated": int(len(treated)),
            "n_holdout": int(len(control)),
            "lapse_rate_treated": None,
            "lapse_rate_holdout": None,
            "lift_pp": None,
            "ci95_pp": None,
            "verdict": "no data",
        }
        if len(treated) and len(control):
            kt, nt = int(treated["churned"].sum()), len(treated)
            kc, nc = int(control["churned"].sum()), len(control)
            lift, lo, hi = _newcombe_diff(kc, nc, kt, nt)
            if min(nt, nc) < MIN_GROUP_FOR_VERDICT:
                verdict = f"too small (need {MIN_GROUP_FOR_VERDICT}+ per group)"
            elif lo > 0:
                verdict = "lift"
            elif hi < 0:
                verdict = "harm"
            else:
                verdict = "inconclusive"
            row.update(
                {
                    "lapse_rate_treated": round(kt / nt, 4),
                    "lapse_rate_holdout": round(kc / nc, 4),
                    "lift_pp": round(100 * lift, 2),
                    "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
                    "verdict": verdict,
                }
            )
        out.append(row)
    return out


def summarize_outcomes(joined: pd.DataFrame) -> dict[str, Any]:
    """Observed lapse vs calibrated p by band; lift vs holdout by playbook."""
    df = joined.copy()
    df["action_taken"] = df["action_taken"].fillna("").astype(str)
    blank = df["action_taken"].str.strip() == ""
    df.loc[blank, "action_taken"] = "(not recorded)"
    labelled = df["churned"].notna()
    matched = df["action_taken"] == df["action_suggested"].fillna("").astype(str)
    return {
        "n_logged": int(len(df)),
        "n_labelled": int(labelled.sum()),
        "observed_lapse_rate": (
            round(float(df.loc[labelled, "churned"].mean()), 4) if labelled.any() else None
        ),
        "action_matched_suggestion_rate": round(float(matched.mean()), 4) if len(df) else None,
        "by_band": _group_summary(df, "band"),
        "by_action_taken": _group_summary(df, "action_taken"),
        "lift_vs_holdout": lift_by_playbook(df),
        "auto_action": "none",
        "note": (
            "by_action_taken is descriptive (who got which action is not random). "
            "lift_vs_holdout compares like with like: subscribers the same playbook "
            f"would have gone to, split by the deterministic {config.HOLDOUT_PCT}% holdout."
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
        raise FileNotFoundError(f"Action log not found: {log_path}")
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
        description="Join the action log to renewal outcomes and estimate lift vs holdout"
    )
    parser.add_argument(
        "--log", type=str, default=None, help=f"Action log (default: {DEFAULT_LOG_PATH})"
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Labels CSV with user_id, churned[, observed_at] (default: renewals_t7.csv)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help=f"Joined CSV (default: {DEFAULT_OUTCOMES_CSV})"
    )
    parser.add_argument("--json", type=str, default=None, help="Summary JSON path")
    args = parser.parse_args(argv)

    from retention_radar.data.ingest import resolve_users_csv

    log_path = Path(args.log) if args.log else config.runtime_log_dir() / DEFAULT_LOG_PATH.name
    labels_path = Path(args.labels) if args.labels else resolve_users_csv()
    try:
        summary = write_outcomes(log_path, labels_path, args.out, args.json)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Joined {summary['n_logged']} action rows → {summary['out_csv']}")
    print(f"Summary → {summary['out_json']}")
    print(
        f"  labelled={summary['n_labelled']}  "
        f"observed_lapse_rate={summary['observed_lapse_rate']}"
    )
    for row in summary["by_band"]:
        print(
            f"  band={row['band']:<7} n={row['n']:<5} "
            f"lapse={row['observed_lapse_rate']}  mean_p_cal={row['mean_p_cal']}"
        )
    print("Lift vs holdout (percentage points, 95% CI):")
    for row in summary["lift_vs_holdout"]:
        if row["lift_pp"] is None:
            print(f"  {row['playbook']:<22} not enough data (treated={row['n_treated']}, holdout={row['n_holdout']})")
            continue
        print(
            f"  {row['playbook']:<22} treated={row['n_treated']:<5} holdout={row['n_holdout']:<4} "
            f"lift={row['lift_pp']:+.1f}pp CI={row['ci95_pp']} ({row['verdict']})"
        )


# Re-exported so callers can tell policy outcomes apart without importing policy.
__all__ = ["HOLDOUT", "NO_ACTION", "join_outcomes", "summarize_outcomes", "write_outcomes", "main"]


if __name__ == "__main__":
    main()
