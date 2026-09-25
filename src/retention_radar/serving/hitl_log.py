"""HITL review log — append reviewer decisions after scoring.

This is the act step of a predict → act → outcome loop. Outcome write-back
(joining these rows to later churn labels) lives in ``serving/outcomes.py``
(``python -m retention_radar.cli.hitl_outcomes``).

Examples:
    python -m retention_radar.cli.hitl_log \\
        --from-packet artifacts/santosh_decision_packet.json \\
        --reviewer santosh --action-taken monitor --notes "looks fine"
    # Bulk: import a reviewer-decisions CSV (user_id, reviewer, action_taken, notes,
    # timestamp) against a batch_score queue — e.g. an export from the CRM task list.
    python -m retention_radar.cli.hitl_log \\
        --from-scores artifacts/use_cases/queue.csv \\
        --decisions data/use_cases/review_decisions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retention_radar import config

try:  # POSIX advisory lock so concurrent writers (threads or processes) never interleave
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

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
    if not packet.get("scoring"):
        raise ValueError(
            "packet was held by validation (no score); fix the input data and "
            "re-score before logging a review"
        )
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
    """Extract HITL log fields from a batch_score / API score row.

    Refuses records that carry no score (held by validation, a queue reject, an
    API 422 body): a review must always point at a real model output.
    """
    hitl = record.get("hitl") if isinstance(record.get("hitl"), dict) else {}
    p_cal = record.get("p_cal", record.get("churn_probability_calibrated"))
    band = str(record.get("band") or record.get("risk_band") or "")
    if (
        ("scoring" in record and not record["scoring"])
        or hitl.get("blocked_by_validation")
        or p_cal in (None, "")
        or band not in {"low", "medium", "high"}
    ):
        raise ValueError(
            "record carries no score (held by validation or not a score row); "
            "fix the input and re-score before logging a review"
        )
    return row_from_score(
        user_id=str(record.get("user_id") or "").strip(),
        p_cal=float(p_cal),
        band=band,
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


_LOG_LOCK = threading.Lock()


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append-only write: header once, never truncate; safe across threads/processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, open(path, "a", encoding="utf-8", newline="") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=HITL_LOG_COLUMNS)
        f.seek(0, 2)
        if f.tell() == 0:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row[c] for c in HITL_LOG_COLUMNS})


def ensure_log_header(path: Path) -> None:
    """Create the log file with the template header if missing (never truncates)."""
    _append_rows(Path(path), [])


def append_hitl_row(path: Path, row: dict[str, Any]) -> Path:
    """Append one validated row to the HITL review log CSV."""
    path = Path(path)
    missing = [c for c in HITL_LOG_COLUMNS if c not in row]
    if missing:
        raise ValueError(f"HITL log row missing columns: {missing}")
    _append_rows(path, [row])
    return path


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(row.get(c, "")).strip() for c in ("user_id", "reviewer", "action_taken", "timestamp"))


DECISION_COLUMNS = ["user_id", "reviewer", "action_taken", "notes", "timestamp"]


def rows_from_decisions(
    scores: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    default_timestamp: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Join reviewer decisions to scored queue rows → (log rows, problems).

    Score fields (p_cal, band, suggested action) always come from the queue, never
    from the decisions file, so a reviewer export cannot rewrite what the model said.
    """
    by_user: dict[str, dict[str, Any]] = {}
    dupes: set[str] = set()
    for r in scores:
        uid = str(r["user_id"]).strip()
        if uid in by_user:
            dupes.add(uid)
        by_user[uid] = r
    rows, problems = [], []
    for i, d in enumerate(decisions, start=1):
        missing = [c for c in ("user_id", "reviewer", "action_taken") if not str(d.get(c) or "").strip()]
        if missing:
            problems.append(f"decision row {i}: missing {missing}")
            continue
        uid = str(d["user_id"]).strip()
        rec = by_user.get(uid)
        if rec is None:
            problems.append(f"decision row {i}: user_id {uid!r} is not in the scored queue")
            continue
        if uid in dupes:
            problems.append(f"decision row {i}: user_id {uid!r} appears more than once in the queue")
            continue
        rows.append(
            row_from_score_record(
                rec,
                reviewer=str(d["reviewer"]).strip(),
                action_taken=str(d["action_taken"]).strip(),
                notes=str(d.get("notes") or "").strip(),
                timestamp=str(d.get("timestamp") or "").strip() or default_timestamp,
            )
        )
    return rows, problems


def _read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
        "--from-scores",
        type=str,
        default=None,
        help="batch_score queue CSV; use with --decisions for a bulk import",
    )
    parser.add_argument(
        "--decisions",
        type=str,
        default=None,
        help="Reviewer decisions CSV (user_id, reviewer, action_taken, notes, timestamp)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Review time (ISO-8601 UTC); default now",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log CSV path (default: <RETENTION_RADAR_LOG_DIR or artifacts>/hitl_review_log.csv)",
    )
    parser.add_argument("--reviewer", type=str, default="")
    parser.add_argument("--action-taken", type=str, default="")
    parser.add_argument("--notes", type=str, default="")
    args = parser.parse_args(argv)

    log_path = Path(args.log) if args.log else config.runtime_log_dir() / DEFAULT_LOG_PATH.name

    if args.from_scores or args.decisions:
        if not (args.from_scores and args.decisions):
            raise SystemExit("Bulk import needs both --from-scores and --decisions")
        rows, problems = rows_from_decisions(
            _read_csv_dicts(Path(args.from_scores)),
            _read_csv_dicts(Path(args.decisions)),
            default_timestamp=args.timestamp,
        )
        if problems:  # all-or-nothing: fix the file and rerun, nothing half-imported
            for msg in problems:
                print(f"  problem: {msg}")
            raise SystemExit(f"hitl_log: {len(problems)} problem(s); nothing appended to {log_path}")
        # Idempotent: a rerun does not append reviews that are already in the log.
        existing = {_row_key(r) for r in _read_csv_dicts(log_path)} if log_path.exists() else set()
        new_rows = [r for r in rows if _row_key(r) not in existing]
        _append_rows(log_path, new_rows)
        print(f"Appended {len(new_rows)} HITL review row(s) → {log_path}")
        if len(new_rows) < len(rows):
            print(f"  skipped {len(rows) - len(new_rows)} already-logged review(s)")
        agreed = sum(r["action_taken"] == r["action_suggested"] for r in rows)
        print(f"  reviewer agreed with the suggested action on {agreed}/{len(rows)}")
        return

    if not args.from_packet and not args.from_score:
        raise SystemExit("Provide --from-packet, --from-score, or --from-scores + --decisions")

    src = args.from_packet or args.from_score
    try:
        with open(src, encoding="utf-8") as f:
            obj = json.load(f)
        build = row_from_packet if args.from_packet else row_from_score_record
        row = build(
            obj,
            reviewer=args.reviewer,
            action_taken=args.action_taken,
            notes=args.notes,
            timestamp=args.timestamp,
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"hitl_log: {src} is not a single JSON object ({exc}); --from-packet / --from-score "
            "take one packet or score JSON, not the JSONL from single_record --dir"
        ) from exc
    except (ValueError, OSError) as exc:
        raise SystemExit(f"hitl_log: {exc}") from exc

    append_hitl_row(log_path, row)
    print(f"Appended HITL review → {log_path}")
    print(
        f"  user={row['user_id']} band={row['band']} "
        f"suggested={row['action_suggested']} taken={row['action_taken']!r}"
    )


if __name__ == "__main__":
    main()
