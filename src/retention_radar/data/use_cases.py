"""Use-case data pack for the serving layer (``data/use_cases/``).

Builds a small, reproducible set of **real seed-42 records** that walk the whole
service. Every scenario row except the Santosh hero comes from the test split
(never used for training, early stopping, calibration or τ); Santosh is the injected
hero and sits in the validation split. The pack covers: one named scenario per HITL path (monitor / nurture in both bands /
outreach / escalate), records that must be *held* by validation, a weekly batch to
turn into a review queue, reviewer decisions, and churn labels observed 30 days
later for the outcome report.

Everything is synthetic (seed 42, no real PII). Scenario rows are picked from the
**test split** (never trained on) by explicit filters, and each scenario's expected
band / action is what the committed ``models/`` bundle returns — ``--check``
re-scores the pack and fails if the bundle or policy stops agreeing.

Examples:
    python -m retention_radar.cli.generate_data          # data/raw/users.csv (seed 42)
    python -m retention_radar.cli.build_use_cases        # rebuild data/use_cases/
    python -m retention_radar.cli.build_use_cases --check
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from retention_radar import config
from retention_radar.data.generate import santosh_profile
from retention_radar.data.ingest import load_users
from retention_radar.features.transform import prepare_xy
from retention_radar.serving.batch_score import load_metrics, score_feature_frame
from retention_radar.serving.infer import predict_user
from retention_radar.serving.policy import hitl_action, risk_band
from retention_radar.training.calibrate import load_calibrator
from retention_radar.training.split import stratified_train_val_test

USE_CASE_DIR = config.PROJECT_ROOT / "data" / "use_cases"

# Fixed demo calendar so the outcome join (labels observed after review) is stable.
SCORED_AT = "2026-09-28T09:00:00Z"
REVIEWED_AT = "2026-09-29T15:00:00Z"
LABELS_OBSERVED_AT = "2026-10-28T00:00:00Z"
REVIEWER = "cs-demo (simulated)"
WEEKLY_SAMPLE = 200

OUTREACH = "retention outreach (human review)"
NURTURE = "nurture / check-in"


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    story: str
    band: str
    action: str
    select: Callable[[pd.DataFrame], pd.Series] | None  # None → Santosh hero
    reviewer_action: str | None = None  # override of the suggested action
    reviewer_note: str = "accepted suggested action"
    # ``story`` / ``reviewer_note`` are str.format templates over the selected record,
    # so every number quoted in them is true of that record by construction.


SCENARIOS: list[Scenario] = [
    Scenario(
        "steady_power_user",
        "Steady power user (Santosh)",
        "On the {plan_tier} plan {days_since_signup:.0f} days, adoption {feature_adoption_score:.2f}, "
        "renewing in {days_until_renewal:.0f} days. {failed_requests_rate:.0%} failed requests and "
        "{support_tickets_last_90d:.0f} tickets in 90 days: friction the model does not worry about.",
        "low",
        "monitor",
        None,
        reviewer_note="healthy; no action",
    ),
    Scenario(
        "dip_but_healthy",
        "Usage dip, still healthy",
        "Session(s) this week: {sessions_last_7d:.0f}, against {sessions_last_30d:.0f} in 30 days "
        "(engagement_trend {engagement_trend:.2f}), but adoption {feature_adoption_score:.2f}, NPS "
        "{nps_score:.1f} and no failed payments. One quiet week is not churn risk; the queue stays calm.",
        "low",
        "monitor",
        lambda t: (t.engagement_trend < 0.6)
        & (t.payment_failures_last_90d == 0)
        & (t.feature_adoption_score >= 0.5),
        reviewer_note="seasonal dip; no action",
    ),
    Scenario(
        "new_trial_friction",
        "New free trial hitting friction",
        "Free plan, {days_since_signup:.0f} days old, already {support_tickets_last_90d:.0f} support "
        "tickets. Low band, but above half of τ: a light check-in, not a sales call.",
        "low",
        NURTURE,
        lambda t: (t.days_since_signup <= 90)
        & (t.plan_tier == "free")
        & (t.support_tickets_last_90d >= 2),
        reviewer_note="offered an onboarding session",
    ),
    Scenario(
        "borderline_friction",
        "Borderline: friction just under τ",
        "{support_tickets_last_90d:.0f} tickets and {failed_requests_rate:.0%} failed requests: medium "
        "band but still below the best-F1 threshold. The policy says nurture; the reviewer "
        "upgrades to outreach because the tickets are about one bug. This is the human override "
        "the log exists for.",
        "medium",
        NURTURE,
        lambda t: (t.support_tickets_last_90d >= 3) | (t.failed_requests_rate >= 0.25),
        reviewer_action=OUTREACH,
        reviewer_note="{support_tickets_last_90d:.0f} tickets on the same bug; call instead of email",
    ),
    Scenario(
        "payment_friction",
        "Payment failures plus support load",
        "{payment_failures_last_90d:.0f} failed payments and {support_tickets_last_90d:.0f} support "
        "tickets in 90 days on the {plan_tier} plan. Above τ, so a human owns the outreach; nothing "
        "is sent automatically.",
        "medium",
        OUTREACH,
        lambda t: (t.payment_failures_last_90d >= 2) & (t.support_tickets_last_90d >= 3),
        reviewer_note="billing fixed on call; outreach done",
    ),
    Scenario(
        "gone_dark",
        "Gone dark",
        "Nothing in the last 30 days (no sessions, API calls, tokens or agent runs); last seen "
        "{last_active_days_ago:.0f} days ago, with {support_tickets_last_90d:.0f} tickets and "
        "{payment_failures_last_90d:.0f} failed payments in 90 days. High band: escalate to the "
        "retention owner today.",
        "high",
        "escalate",
        lambda t: (t.last_active_days_ago >= 30)
        & (t.sessions_last_7d == 0)
        & (t.sessions_last_30d == 0)
        & (t.api_calls_last_30d == 0)
        & (t.tokens_consumed_last_30d == 0)
        & (t.agent_runs_last_30d == 0),
        reviewer_note="escalated to account owner",
    ),
    Scenario(
        "enterprise_renewal_risk",
        "Enterprise renewal at risk",
        "Enterprise account renewing in {days_until_renewal:.0f} days with seat utilisation "
        "{seat_utilization:.0%}, last active {last_active_days_ago:.0f} days ago. High band; the "
        "reviewer books an executive-sponsor call.",
        "high",
        "escalate",
        lambda t: (t.plan_tier == "enterprise")
        & (t.days_until_renewal <= 30)
        & (t.seat_utilization < 0.4),
        reviewer_note="exec sponsor call booked before renewal",
    ),
]

# (id, what is wrong, field → bad value or None to drop, substring the error must contain)
INVALID_CASES: list[tuple[str, str, dict[str, Any], str]] = [
    ("unknown_plan_tier", "plan_tier outside the enum", {"plan_tier": "platinum"}, "plan_tier"),
    ("missing_nps_score", "nps_score missing (nulls fail loud)", {"nps_score": None}, "nps_score"),
    (
        "rate_out_of_range",
        "failed_requests_rate above 1.0",
        {"failed_requests_rate": 1.4},
        "failed_requests_rate",
    ),
    (
        "non_numeric_sessions",
        "sessions_last_30d sent as text",
        {"sessions_last_30d": "ten"},
        "sessions_last_30d",
    ),
]


def _record(row: pd.Series | dict) -> dict[str, Any]:
    """24-field inference payload with plain-Python values (no label)."""
    src = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    out = {}
    for k in config.INFERENCE_REQUIRED_KEYS:
        v = src[k]
        out[k] = v.item() if hasattr(v, "item") else v
    return out


def _load_bundle():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
    return joblib.load(config.MODEL_PATH), load_calibrator(config.CALIBRATOR_PATH), load_metrics()


def _score_one(record: dict, bundle, calibrator, metrics) -> dict[str, Any]:
    res = predict_user(record, bundle, calibrator=calibrator, top_k=3)
    p = float(res["churn_probability"])
    band = risk_band(p)
    action = hitl_action(p, float(metrics.get("best_f1_threshold", 0.5)), band)["action"]
    return {
        "p_raw": round(float(res["churn_probability_raw"]), 6),
        "p_cal": round(p, 6),
        "band": band,
        "hitl_action": action,
        "top_drivers": [
            {"feature": f, "contribution": round(float(c), 3)} for f, c in res["top_features"]
        ],
    }


def build_use_cases(out_dir: Path = USE_CASE_DIR) -> dict[str, Any]:
    """Select scenario rows, score them with the committed bundle, write the pack."""
    users = load_users()
    X, y = prepare_xy(users)
    _, _, X_test, _, _, _ = stratified_train_val_test(X, y)
    holdout = users.loc[X_test.index].copy()
    bundle, calibrator, metrics = _load_bundle()

    scored = score_feature_frame(holdout, bundle, calibrator, metrics, scored_at=SCORED_AT)
    table = holdout.merge(scored[["user_id", "p_cal", "band", "hitl_action"]], on="user_id")

    personas, used = [], set()
    for sc in SCENARIOS:
        if sc.select is None:
            record = {k: v for k, v in santosh_profile().items() if k != "churned"}
            source = {"dataset": "data/raw/santosh_shinde.json", "split": "hero (injected)"}
        else:
            cand = table[
                sc.select(table)
                & (table.band == sc.band)
                & (table.hitl_action == sc.action)
                & ~table.user_id.isin(used)
            ]
            if cand.empty:
                raise RuntimeError(f"no seed-42 holdout row fits scenario {sc.id!r}")
            # Most typical candidate: closest to the median risk; user_id breaks ties.
            median = cand.p_cal.median()
            cand = cand.assign(_d=(cand.p_cal - median).abs()).sort_values(["_d", "user_id"])
            row = cand.iloc[0]
            record = _record(row)
            source = {
                "dataset": "seed-42 synthetic users.csv",
                "split": "test (holdout)",
                "candidates_matching_filter": int(len(cand)),
            }
        used.add(record["user_id"])
        result = _score_one(record, bundle, calibrator, metrics)
        if (result["band"], result["hitl_action"]) != (sc.band, sc.action):
            raise RuntimeError(f"{sc.id}: bundle returned {result}, expected {sc.band}/{sc.action}")
        personas.append(
            {
                "id": sc.id,
                "title": sc.title,
                "story": sc.story.format(**record),
                "source": source,
                "expected": {k: result[k] for k in ("p_raw", "p_cal", "band", "hitl_action")},
                "top_drivers": result["top_drivers"],
                "reviewer": {
                    "action_taken": sc.reviewer_action or sc.action,
                    "notes": sc.reviewer_note.format(**record),
                },
                "record": record,
            }
        )

    base = personas[[p["id"] for p in personas].index("payment_friction")]["record"]
    invalid = []
    for iid, problem, patch, needle in INVALID_CASES:
        rec = copy.deepcopy(base)
        rec["user_id"] = f"invalid_{iid}"
        rec["user_name"] = f"Invalid record ({iid})"
        for k, v in patch.items():
            rec[k] = v
        invalid.append({"id": iid, "problem": problem, "error_contains": needle, "record": rec})

    # Weekly batch: a seeded sample of the holdout + every persona + the invalid rows.
    persona_ids = {p["record"]["user_id"] for p in personas}
    sample = holdout[~holdout.user_id.isin(persona_ids)].sample(
        n=WEEKLY_SAMPLE, random_state=config.RANDOM_SEED
    )
    batch_rows = [_record(r) for _, r in sample.iterrows()] + [p["record"] for p in personas]
    batch = pd.DataFrame(batch_rows, columns=config.INFERENCE_REQUIRED_KEYS)
    batch = batch.sort_values("user_id").reset_index(drop=True)
    bad = pd.DataFrame([i["record"] for i in invalid], columns=config.INFERENCE_REQUIRED_KEYS)
    weekly = pd.concat([batch, bad], ignore_index=True)

    # Reviewer decisions: the rows a CS reviewer actually works (everything not on
    # "monitor") plus every persona. Simulated reviewer: accepts the suggestion
    # unless a persona scenario records an override.
    queue = score_feature_frame(batch, bundle, calibrator, metrics, scored_at=SCORED_AT)
    persona_by_uid = {p["record"]["user_id"]: p for p in personas}
    decisions = []
    for rec in queue.to_dict(orient="records"):
        uid = rec["user_id"]
        persona = persona_by_uid.get(uid)
        if persona is None and rec["hitl_action"] == "monitor":
            continue
        decisions.append(
            {
                "user_id": uid,
                "reviewer": REVIEWER,
                "action_taken": persona["reviewer"]["action_taken"] if persona else rec["hitl_action"],
                "notes": persona["reviewer"]["notes"] if persona else "accepted suggested action",
                "timestamp": REVIEWED_AT,
            }
        )

    # Day-30 labels for everyone in the (valid) batch, from the generator's ground truth.
    truth = users.set_index("user_id")[config.TARGET_COLUMN]
    labels = pd.DataFrame(
        {
            "user_id": batch.user_id,
            "churned": [int(truth[u]) for u in batch.user_id],
            "observed_at": LABELS_OBSERVED_AT,
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("records", "invalid"):
        (out_dir / sub).mkdir(exist_ok=True)
        for old in (out_dir / sub).glob("*.json"):
            old.unlink()
    for p in personas:
        _write_json(out_dir / "records" / f"{p['id']}.json", p["record"])
    for i in invalid:
        _write_json(out_dir / "invalid" / f"{i['id']}.json", i["record"])
    manifest = {
        "about": "Seed-42 synthetic use cases for the serving layer; see README.md.",
        "model_version": f"seed{config.RANDOM_SEED}-churn_xgb",
        "best_f1_threshold": metrics.get("best_f1_threshold"),
        "calendar": {
            "scored_at": SCORED_AT,
            "reviewed_at": REVIEWED_AT,
            "labels_observed_at": LABELS_OBSERVED_AT,
        },
        "personas": personas,
        "invalid_records": invalid,
        "weekly_batch": {
            "file": "weekly_batch.csv",
            "rows": int(len(weekly)),
            "valid_rows": int(len(batch)),
            "invalid_rows": int(len(bad)),
            "expected_actions": queue["hitl_action"].value_counts().sort_index().to_dict(),
        },
    }
    _write_json(out_dir / "personas.json", manifest)
    weekly.to_csv(out_dir / "weekly_batch.csv", index=False)
    pd.DataFrame(decisions).to_csv(out_dir / "review_decisions.csv", index=False)
    labels.to_csv(out_dir / "labels_day30.csv", index=False)
    (out_dir / "README.md").write_text(_readme(manifest, len(decisions)), encoding="utf-8")
    return manifest


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or (USE_CASE_DIR / "personas.json")
    return json.loads(p.read_text(encoding="utf-8"))


def check_use_cases(out_dir: Path = USE_CASE_DIR) -> list[str]:
    """Re-verify every committed file of the pack against the current bundle.

    Checks: scenario records (manifest and ``records/*.json``) still score to the
    recorded p_raw / p_cal / band / action / top drivers; invalid records (manifest
    and ``invalid/*.json``) are still rejected for the recorded reason; the weekly
    batch still yields the recorded queue and rejects; decisions cover exactly the
    non-monitor queue plus every scenario, overriding only where a scenario says so;
    labels cover exactly the valid batch users.
    """
    from retention_radar.serving.packet import normalize_record, validate_payload

    manifest = load_manifest(out_dir / "personas.json")
    bundle, calibrator, metrics = _load_bundle()
    problems = []

    def _read_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return f"<unreadable: {exc}>"

    for p in manifest["personas"]:
        on_disk = _read_json(out_dir / "records" / f"{p['id']}.json")
        if on_disk != p["record"]:
            problems.append(f"records/{p['id']}.json differs from personas.json")
        got = _score_one(p["record"], bundle, calibrator, metrics)
        exp = p["expected"]
        drivers = [d["feature"] for d in p["top_drivers"]]
        if (
            (got["band"], got["hitl_action"]) != (exp["band"], exp["hitl_action"])
            or abs(got["p_cal"] - exp["p_cal"]) > 1e-6
            or abs(got["p_raw"] - exp["p_raw"]) > 1e-6
            or [d["feature"] for d in got["top_drivers"]] != drivers
        ):
            problems.append(f"{p['id']}: expected {exp} drivers {drivers}, got {got}")
    for i in manifest["invalid_records"]:
        if _read_json(out_dir / "invalid" / f"{i['id']}.json") != i["record"]:
            problems.append(f"invalid/{i['id']}.json differs from personas.json")
        v = validate_payload(normalize_record(i["record"])[0])
        if v["ok"] or not any(i["error_contains"] in e for e in v["errors"]):
            problems.append(f"invalid/{i['id']}: validation did not reject it as expected: {v}")

    batch = pd.read_csv(
        out_dir / "weekly_batch.csv", dtype={"user_id": str, "user_name": str, "plan_tier": str}
    )
    scores = score_feature_frame(batch, bundle, calibrator, metrics, scored_at=SCORED_AT)
    wb = manifest["weekly_batch"]
    got_actions = scores["hitl_action"].value_counts().sort_index().to_dict()
    if got_actions != wb["expected_actions"] or len(scores.attrs["rejected"]) != wb["invalid_rows"]:
        problems.append(
            f"weekly_batch: expected {wb['expected_actions']} + {wb['invalid_rows']} rejects, "
            f"got {got_actions} + {len(scores.attrs['rejected'])} rejects"
        )

    suggested = dict(zip(scores["user_id"], scores["hitl_action"]))
    personas = {p["record"]["user_id"]: p for p in manifest["personas"]}
    decisions = pd.read_csv(out_dir / "review_decisions.csv", dtype=str).fillna("")
    expect_ids = {u for u, a in suggested.items() if a != "monitor"} | set(personas)
    if set(decisions["user_id"]) != expect_ids or decisions["user_id"].duplicated().any():
        problems.append("review_decisions.csv does not cover exactly the non-monitor queue + scenarios")
    for d in decisions.to_dict(orient="records"):
        persona = personas.get(d["user_id"])
        want = persona["reviewer"]["action_taken"] if persona else suggested.get(d["user_id"])
        if d["action_taken"] != want:
            problems.append(f"review_decisions: {d['user_id']} took {d['action_taken']!r}, expected {want!r}")

    labels = pd.read_csv(out_dir / "labels_day30.csv", dtype={"user_id": str})
    if set(labels["user_id"]) != set(suggested) or labels["user_id"].duplicated().any():
        problems.append("labels_day30.csv does not cover exactly the valid batch users")
    if not set(labels["churned"].unique()) <= {0, 1}:
        problems.append("labels_day30.csv churned must be 0/1")
    return problems


def _readme(manifest: dict[str, Any], n_decisions: int) -> str:
    tau = manifest["best_f1_threshold"]
    lines = [
        "# Use-case data pack (seed 42)",
        "",
        "_Generated by `python -m retention_radar.cli.build_use_cases` — do not edit by hand._",
        "",
        "Real rows from the seed-42 synthetic data, one per HITL path, plus records the "
        "service must refuse. Every scenario except Santosh is a **test-split** row (never "
        "used for training, early stopping, calibration or τ); Santosh is the injected hero "
        "and sits in the **validation** split, which fits the calibrator and chooses τ. "
        "Expected results are "
        f"what the committed `models/` bundle returns (τ = {tau}); "
        "`python -m retention_radar.cli.build_use_cases --check` re-verifies them. "
        "Synthetic data only, no real PII.",
        "",
        "## Scenarios",
        "",
        "| Scenario | Record | Raw → calibrated | Band | Suggested action | Reviewer did |",
        "|----------|--------|------------------|------|------------------|--------------|",
    ]
    for p in manifest["personas"]:
        e, r = p["expected"], p["reviewer"]
        lines.append(
            f"| **{p['title']}** | [`records/{p['id']}.json`](records/{p['id']}.json) "
            f"(`{p['record']['user_id']}`) | {e['p_raw']:.3f} → {e['p_cal']:.3f} | "
            f"{e['band']} | {e['hitl_action']} | {r['action_taken']} |"
        )
    lines += ["", "Stories and top drivers:", ""]
    for p in manifest["personas"]:
        drivers = ", ".join(f"`{d['feature']}` {d['contribution']:+.2f}" for d in p["top_drivers"])
        lines.append(f"- **{p['title']}** — {p['story']} Top drivers (SHAP, log-odds): {drivers}.")
    lines += [
        "",
        "## Records the service must hold (never scored or queued)",
        "",
        "| File | Problem |",
        "|------|---------|",
    ]
    for i in manifest["invalid_records"]:
        lines.append(f"| [`invalid/{i['id']}.json`](invalid/{i['id']}.json) | {i['problem']} |")
    wb = manifest["weekly_batch"]
    acts = ", ".join(f"{k}: {v}" for k, v in wb["expected_actions"].items())
    cal = manifest["calendar"]
    lines += [
        "",
        "## Files",
        "",
        "| File | Use |",
        "|------|-----|",
        f"| `weekly_batch.csv` | {wb['rows']} rows = {wb['valid_rows'] - 1} test-split users "
        f"(a seeded sample + every scenario) + Santosh + {wb['invalid_rows']} invalid rows. "
        f"Expected queue: {acts}. No label column, like production input. |",
        f"| `review_decisions.csv` | {n_decisions} reviewer decisions (everything not on "
        "*monitor*, plus every scenario) at "
        f"`{cal['reviewed_at']}`. Simulated reviewer: accepts the suggestion except the "
        "scenario overrides above. |",
        f"| `labels_day30.csv` | Churn observed at `{cal['labels_observed_at']}` for every "
        "valid batch user (the generator's ground truth, framed as a later outcome). |",
        "| `personas.json` | Machine-readable manifest: stories, expected outputs, drivers, records. |",
        "",
        "## Walk the whole service",
        "",
        "```bash",
        "make use-cases            # queue → packets → reviews → day-30 outcome report",
        "# or step by step (fresh output dir; the review import skips already-logged rows):",
        "rm -rf artifacts/use_cases && mkdir -p artifacts/use_cases",
        "python -m retention_radar.cli.batch_score --csv data/use_cases/weekly_batch.csv \\",
        f"  --out artifacts/use_cases/queue.csv --scored-at {cal['scored_at']}",
        "python -m retention_radar.cli.single_record --dir data/use_cases/records \\",
        "  --out artifacts/use_cases/packets.jsonl",
        "python -m retention_radar.cli.hitl_log --from-scores artifacts/use_cases/queue.csv \\",
        "  --decisions data/use_cases/review_decisions.csv --log artifacts/use_cases/hitl_review_log.csv",
        "python -m retention_radar.cli.hitl_outcomes --log artifacts/use_cases/hitl_review_log.csv \\",
        "  --labels data/use_cases/labels_day30.csv --out artifacts/use_cases/hitl_outcomes.csv",
        "curl -s localhost:8000/v1/churn/score -H 'content-type: application/json' \\",
        "  -d @data/use_cases/records/gone_dark.json",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build / check the data/use_cases pack")
    parser.add_argument("--out-dir", type=str, default=str(USE_CASE_DIR))
    parser.add_argument(
        "--check",
        action="store_true",
        help="Re-score the committed pack with the current bundle; exit 1 on mismatch",
    )
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.check:
        problems = check_use_cases(out_dir)
        for msg in problems:
            print(f"MISMATCH {msg}")
        if not problems:
            m = load_manifest(out_dir / "personas.json")
            print(
                f"Use cases OK: {len(m['personas'])} scenarios, {len(m['invalid_records'])} held "
                f"records, weekly batch {m['weekly_batch']['expected_actions']}"
            )
        return 1 if problems else 0
    manifest = build_use_cases(out_dir)
    for p in manifest["personas"]:
        e = p["expected"]
        print(f"  {p['id']:<24} {p['record']['user_id']:<16} p_cal={e['p_cal']:.3f} "
              f"{e['band']:<6} {e['hitl_action']}")
    print(f"Wrote use-case pack → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
