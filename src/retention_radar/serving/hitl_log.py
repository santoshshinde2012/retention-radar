"""HITL review log — append reviewer decisions after scoring.

This is the start of a predict → act → outcome loop. Outcome write-back
(linking retention actions to later churn labels) is deferred.

Examples:
    python -m retention_radar.cli.hitl_log \\
        --from-packet artifacts/santosh_decision_packet.json \\
        --reviewer santosh --action-taken monitor --notes "looks fine"
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retention_radar import config

HITL_LOG_COLUMNS = [
    "user_id",
    "p_cal",
    "band",
    "action_suggested",
    "reviewer",
    "action_taken",
    "notes",
    "timestamp",
]

TEMPLATE_CSV = config.PROJECT_ROOT / "configs" / "templates" / "hitl_review_log.csv"
SCHEMA_PATH = config.PROJECT_ROOT / "configs" / "hitl_review_log.schema.json"
DEFAULT_LOG_PATH = config.ARTIFACTS_DIR / "hitl_review_log.csv"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_from_score(
    *,
    user_id: str,
    p_cal: float,
    band: str,
    action_suggested: str,
    reviewer: str = "",
    action_taken: str = "",
    notes: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one HITL log row from a scored / packet-like record."""
    return {
        "user_id": user_id,
        "p_cal": f"{float(p_cal):.6f}",
        "band": band,
        "action_suggested": action_suggested,
        "reviewer": reviewer,
        "action_taken": action_taken,
        "notes": notes,
        "timestamp": timestamp or _utc_now(),
    }


def row_from_packet(
    packet: dict[str, Any],
    *,
    reviewer: str = "",
    action_taken: str = "",
    notes: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Extract HITL log fields from a decision packet."""
    scoring = packet.get("scoring") or {}
    hitl = packet.get("hitl") or {}
    p_cal = scoring.get("churn_probability_calibrated")
    if p_cal is None:
        p_cal = scoring.get("churn_probability", 0.0)
    return row_from_score(
        user_id=str(packet.get("user_id") or ""),
        p_cal=float(p_cal),
        band=str(scoring.get("risk_band") or ""),
        action_suggested=str(hitl.get("action") or ""),
        reviewer=reviewer,
        action_taken=action_taken,
        notes=notes,
        timestamp=timestamp,
    )


def row_from_score_record(
    record: dict[str, Any],
    *,
    reviewer: str = "",
    action_taken: str = "",
    notes: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Extract HITL log fields from a batch_score / API score row."""
    return row_from_score(
        user_id=str(record.get("user_id") or ""),
        p_cal=float(record.get("p_cal", record.get("churn_probability_calibrated", 0.0))),
        band=str(record.get("band") or record.get("risk_band") or ""),
        action_suggested=str(
            record.get("hitl_action")
            or (record.get("hitl") or {}).get("action")
            or record.get("action_suggested")
            or ""
        ),
        reviewer=reviewer,
        action_taken=action_taken,
        notes=notes,
        timestamp=timestamp,
    )


def ensure_log_header(path: Path) -> None:
    """Create the log file with the template header if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HITL_LOG_COLUMNS)
        writer.writeheader()


def append_hitl_row(path: Path, row: dict[str, Any]) -> Path:
    """Append one validated row to the HITL review log CSV."""
    path = Path(path)
    ensure_log_header(path)
    missing = [c for c in HITL_LOG_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"HITL log row missing columns: {missing}")
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HITL_LOG_COLUMNS)
        writer.writerow({c: row[c] for c in HITL_LOG_COLUMNS})
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Append a HITL review-log row from a packet or score JSON"
    )
    parser.add_argument(
        "--from-packet",
        type=str,
        default=None,
        help="Path to decision packet JSON",
    )
    parser.add_argument(
        "--from-score",
        type=str,
        default=None,
        help="Path to a single score JSON (batch/API shaped)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help=f"Log CSV path (default: {DEFAULT_LOG_PATH})",
    )
    parser.add_argument("--reviewer", type=str, default="")
    parser.add_argument("--action-taken", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args(argv)

    if not args.from_packet and not args.from_score:
        raise SystemExit("Provide --from-packet or --from-score")

    log_path = Path(args.log) if args.log else DEFAULT_LOG_PATH

    if args.from_packet:
        with open(args.from_packet, encoding="utf-8") as f:
            packet = json.load(f)
        row = row_from_packet(
            packet,
            reviewer=args.reviewer,
            action_taken=args.action_taken,
            notes=args.notes,
        )
    else:
        with open(args.from_score, encoding="utf-8") as f:
            record = json.load(f)
        row = row_from_score_record(
            record,
            reviewer=args.reviewer,
            action_taken=args.action_taken,
            notes=args.notes,
        )

    append_hitl_row(log_path, row)
    print(f"Appended HITL review → {log_path}")
    print(
        f"  user={row['user_id']} band={row['band']} "
        f"suggested={row['action_suggested']} taken={row['action_taken']!r}"
    )


if __name__ == "__main__":
    main()
