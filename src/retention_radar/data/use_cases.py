"""Use-case data pack for the serving layer (``data/use_cases/``).

A small, reproducible slice of the renewal business, built from the seed-42
cohort and the committed ``models/`` bundle:

* Scenarios: the two worked examples (Santosh, Arjun) plus one real test-split
  subscriber per path through the policy (limit reset, pause offer, cancel-flow
  discount on a first renewal, overage shock, quiet-but-fine, holdout, the one
  Ultra subscriber who gets a person-written email).
* Records the service must hold (never scored or queued), including a Teams
  seat, which belongs to a different model.
* A daily T-7 batch to turn into an action queue, the messaging tool's send
  export for that queue, and renewal outcomes observed afterwards.

Renewal outcomes are the generator's ground truth, with one explicit
simulation: for subscribers who were actually sent a playbook, a would-be
lapse is flipped to a renewal with probability equal to that playbook's
*assumed* effect. That gives the outcome report something to find, and shows
how wide the confidence interval is at this size. Nothing here is a measured
effect.

Examples:
    python -m retention_radar.cli.generate_data
    python -m retention_radar.cli.build_use_cases
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
import numpy as np
import pandas as pd

from retention_radar import config
from retention_radar.data.generate import HERO_PROFILES
from retention_radar.data.ingest import load_users
from retention_radar.features.transform import prepare_xy
from retention_radar.serving.batch_score import load_metrics, score_feature_frame
from retention_radar.serving.infer import predict_user
from retention_radar.serving.policy import HOLDOUT, NO_ACTION, decide, risk_band
from retention_radar.training.calibrate import load_calibrator
from retention_radar.training.split import stratified_train_val_test

USE_CASE_DIR = config.PROJECT_ROOT / "data" / "use_cases"

# Fixed demo calendar so the outcome join (observed after the action) is stable.
SCORED_AT = "2026-09-28T06:00:00Z"
ACTED_AT = "2026-09-28T09:00:00Z"
OUTCOMES_OBSERVED_AT = "2026-10-05T00:00:00Z"  # the renewal date (T-7 + 7 days)
LIFECYCLE_TOOL = "lifecycle-tool"
DAILY_SAMPLE = 400


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    story: str
    band: str | None  # None → accept whatever band the bundle returns
    action: str
    select: Callable[[pd.DataFrame], pd.Series] | None  # None → worked example
    hero: str | None = None
    executed_as: str | None = None  # what the send export records (override)
    note: str = "sent as suggested"
    # ``story`` and ``note`` are str.format templates over the record, so every
    # number quoted in them is true of that record by construction.


SCENARIOS: list[Scenario] = [
    Scenario(
        "santosh_capped_pro",
        "Santosh: capped on Pro, first renewal since the cut",
        "Pro, {renewals_completed:.0f} renewals paid, and this is his first renewal since the "
        "weekly cap was cut. {limit_hits_14d:.0f} cap hits in 14 days, {cheap_model_share_28d:.0%} "
        "of requests on the cheaper model, {active_days_7d:.0f} active days this week against "
        "{active_days_28d:.0f} in 28. The policy picks the limit reset over a discount: his "
        "problem is the cap, not the price.",
        "medium",
        "limit_reset",
        None,
        hero="santosh",
    ),
    Scenario(
        "arjun_steady_pro_plus",
        "Arjun: steady Pro+, deliberately left alone",
        "Pro+ for {renewals_completed:.0f} months, {allowance_used_pct:.0%} of the allowance used, "
        "no cap hits, accept rate steady. Scoring him is useful; contacting him is not.",
        "low",
        NO_ACTION,
        None,
        hero="arjun",
        note="nothing sent",
    ),
    Scenario(
        "first_renewal_tourist",
        "First renewal, already fading",
        "Renewal number one: {active_days_7d:.0f} active days this week, {active_days_28d:.0f} in 28, "
        "{cheap_model_share_28d:.0%} cheap-model share. First renewals are the cliff; the only "
        "playbook worth arming is the cancel-flow discount, paid only if they click cancel.",
        None,
        "cancel_flow_discount",
        lambda t: (t.renewals_completed == 0) & (t.limit_hits_14d < 2),
    ),
    Scenario(
        "side_project_lull",
        "Side-project user between projects",
        "{weekend_usage_ratio:.0%} of activity on weekends and {active_days_7d:.0f} active day(s) "
        "this week. This is the subscriber a pause offer exists for.",
        None,
        "pause_offer",
        lambda t: t.weekend_usage_ratio >= 0.5,
    ),
    Scenario(
        "overage_shock",
        "Switched overage off after a bill",
        "Paid ${overage_usd_28d:.2f} of overage in 28 days, then switched it off; "
        "{limit_hits_14d:.0f} cap hits since. The send export shows the messaging tool "
        "suppressed the email because the subscriber opted out of marketing mail.",
        None,
        "limit_reset",
        lambda t: (t.overage_toggled_off == 1),
        executed_as="suppressed (marketing opt-out)",
        note="email suppressed by opt-out list; logged so it is not counted as treated",
    ),
    Scenario(
        "quiet_but_fine",
        "Quiet week, long tenure",
        "{active_days_7d:.0f} active day(s) this week against {active_days_28d:.0f} in 28, but "
        "{renewals_completed:.0f} renewals paid and no cap trouble. A dip is not a lapse.",
        "low",
        NO_ACTION,
        lambda t: (t.engagement_trend < 0.6) & (t.renewals_completed >= 8),
        note="nothing sent",
    ),
    Scenario(
        "holdout_example",
        "Eligible, but in the holdout",
        "P(lapse) is above τ and a playbook would pay back, but this subscriber hashes into the "
        "{holdout_pct}% holdout. Nothing is sent; the row is what the playbook is measured against.",
        None,
        HOLDOUT,
        lambda t: t.p_cal > 0,
        note="holdout: nothing sent",
    ),
    Scenario(
        "ultra_personal_email",
        "Ultra subscriber: the one place a person writes",
        "Ultra (${price:.0f}/month) above τ. At this price, fifteen minutes of a person's time "
        "has a positive expected value, so the playbook is a personal email, not a template.",
        None,
        "personal_email",
        lambda t: t.plan_tier == "ultra",
        note="written and sent by a person",
    ),
]

# (id, what is wrong, field → bad value or None to drop, substring the error must contain)
INVALID_CASES: list[tuple[str, str, dict[str, Any], str]] = [
    (
        "teams_seat",
        "plan_tier 'teams': seat contraction is a different model",
        {"plan_tier": "teams"},
        "plan_tier",
    ),
    (
        "accept_rate_missing",
        "suggestion_accept_rate_28d missing (inline suggestions turned off): nulls fail loud",
        {"suggestion_accept_rate_28d": None},
        "suggestion_accept_rate_28d",
    ),
    (
        "active_days_inconsistent",
        "active_days_7d larger than active_days_28d",
        {"active_days_7d": 7, "active_days_28d": 3},
        "active_days_7d",
    ),
    (
        "limit_hits_as_text",
        "limit_hits_14d sent as text",
        {"limit_hits_14d": "three"},
        "limit_hits_14d",
    ),
]


def _record(row: pd.Series | dict) -> dict[str, Any]:
    """24-field scoring payload with plain-Python values (no label)."""
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
    d = decide(p, float(metrics.get("best_f1_threshold", 0.5)), band, record)
    return {
        "p_raw": round(float(res["churn_probability_raw"]), 6),
        "p_cal": round(p, 6),
        "band": band,
        "action": d["action"],
        "holdout": bool(d["holdout"]),
        "expected_value_usd": d["expected_value_usd"],
        "top_drivers": [
            {"feature": f, "contribution": round(float(c), 3)} for f, c in res["top_features"]
        ],
    }


def _story_fields(record: dict) -> dict[str, Any]:
    return {
        **record,
        "holdout_pct": config.HOLDOUT_PCT,
        "price": config.PLAN_PRICE_USD[record["plan_tier"]],
    }


def _executed(action: str, override: str | None) -> tuple[str, str]:
    """(executed_by, action_taken) for the simulated send export."""
    if override:
        return LIFECYCLE_TOOL, override
    if action in (NO_ACTION, HOLDOUT):
        return "none", action
    if action == "personal_email":
        return "a person (simulated)", action
    return LIFECYCLE_TOOL, action


def build_use_cases(out_dir: Path = USE_CASE_DIR) -> dict[str, Any]:
    """Select scenario rows, score them with the committed bundle, write the pack."""
    users = load_users()
    X, y = prepare_xy(users)
    _, _, X_test, _, _, _ = stratified_train_val_test(X, y)
    test_rows = users.loc[X_test.index].copy()
    bundle, calibrator, metrics = _load_bundle()

    scored = score_feature_frame(test_rows, bundle, calibrator, metrics, scored_at=SCORED_AT)
    table = test_rows.merge(scored[["user_id", "p_cal", "band", "action"]], on="user_id")

    personas, used = [], set()
    for sc in SCENARIOS:
        if sc.select is None:
            record = HERO_PROFILES[sc.hero]()
            source = {
                "dataset": f"data/raw/subscribers/{config.HEROES[sc.hero]}",
                "split": "worked example (scoring-time record, not in training)",
            }
        else:
            mask = sc.select(table) & (table.action == sc.action) & ~table.user_id.isin(used)
            if sc.band is not None:
                mask &= table.band == sc.band
            cand = table[mask]
            if cand.empty:
                raise RuntimeError(f"no seed-42 test-split row fits scenario {sc.id!r}")
            median = cand.p_cal.median()
            cand = cand.assign(_d=(cand.p_cal - median).abs()).sort_values(["_d", "user_id"])
            record = _record(cand.iloc[0])
            source = {
                "dataset": "seed-42 renewals_t7.csv",
                "split": "test (holdout)",
                "candidates_matching_filter": int(len(cand)),
            }
        used.add(record["user_id"])
        result = _score_one(record, bundle, calibrator, metrics)
        if result["action"] != sc.action or (sc.band and result["band"] != sc.band):
            raise RuntimeError(f"{sc.id}: bundle returned {result}, expected {sc.band}/{sc.action}")
        executed_by, action_taken = _executed(sc.action, sc.executed_as)
        fields = _story_fields(record)
        personas.append(
            {
                "id": sc.id,
                "title": sc.title,
                "story": sc.story.format(**fields),
                "source": source,
                "expected": {
                    k: result[k]
                    for k in ("p_raw", "p_cal", "band", "action", "holdout", "expected_value_usd")
                },
                "top_drivers": result["top_drivers"],
                "executed": {
                    "executed_by": executed_by,
                    "action_taken": action_taken,
                    "notes": sc.note.format(**fields),
                },
                "record": record,
            }
        )

    base = personas[[p["id"] for p in personas].index("first_renewal_tourist")]["record"]
    invalid = []
    for iid, problem, patch, needle in INVALID_CASES:
        rec = copy.deepcopy(base)
        rec["user_id"] = f"invalid_{iid}"
        rec["user_name"] = f"Invalid record ({iid})"
        for k, v in patch.items():
            rec[k] = v
        invalid.append({"id": iid, "problem": problem, "error_contains": needle, "record": rec})

    # Daily T-7 batch: a seeded sample of the test split + every scenario + invalid rows.
    persona_ids = {p["record"]["user_id"] for p in personas}
    sample = test_rows[~test_rows.user_id.isin(persona_ids)].sample(
        n=min(DAILY_SAMPLE, len(test_rows) - len(persona_ids)), random_state=config.RANDOM_SEED
    )
    batch_rows = [_record(r) for _, r in sample.iterrows()] + [p["record"] for p in personas]
    batch = pd.DataFrame(batch_rows, columns=config.INFERENCE_REQUIRED_KEYS)
    batch = batch.sort_values("user_id").reset_index(drop=True)
    bad = pd.DataFrame([i["record"] for i in invalid], columns=config.INFERENCE_REQUIRED_KEYS)
    daily = pd.concat([batch, bad], ignore_index=True)

    # Send export: one row per queue row the business touched or deliberately did not.
    queue = score_feature_frame(batch, bundle, calibrator, metrics, scored_at=SCORED_AT)
    persona_by_uid = {p["record"]["user_id"]: p for p in personas}
    sends = []
    for rec in queue.to_dict(orient="records"):
        uid = rec["user_id"]
        persona = persona_by_uid.get(uid)
        if persona is None and rec["action"] == NO_ACTION:
            continue
        if persona:
            ex = persona["executed"]
        else:
            by, taken = _executed(rec["action"], None)
            ex = {
                "executed_by": by,
                "action_taken": taken,
                "notes": "holdout: nothing sent" if taken == HOLDOUT else "sent as suggested",
            }
        sends.append({"user_id": uid, **ex, "timestamp": ACTED_AT})

    # Renewal outcomes: generator ground truth + the simulated playbook effect.
    truth = users.set_index("user_id")[config.TARGET_COLUMN]
    hero_truth = {"sub_santosh": 1, "sub_arjun": 0}  # worked examples: illustrative outcomes
    taken = {s["user_id"]: s["action_taken"] for s in sends}
    rng = np.random.default_rng(config.RANDOM_SEED)
    outcomes = []
    for uid in batch.user_id:
        lapsed = int(hero_truth[uid]) if uid in hero_truth else int(truth[uid])
        pb = config.PLAYBOOKS.get(taken.get(uid, ""))
        if lapsed and pb is not None and rng.random() < float(pb["effect"]):
            lapsed = 0
        outcomes.append({"user_id": uid, "churned": lapsed, "observed_at": OUTCOMES_OBSERVED_AT})
    labels = pd.DataFrame(outcomes)

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
        "about": "Seed-42 synthetic renewal use cases for the serving layer; see README.md.",
        "model_version": f"seed{config.RANDOM_SEED}-churn_xgb",
        "best_f1_threshold": metrics.get("best_f1_threshold"),
        "holdout_pct": config.HOLDOUT_PCT,
        "calendar": {
            "scored_at": SCORED_AT,
            "acted_at": ACTED_AT,
            "outcomes_observed_at": OUTCOMES_OBSERVED_AT,
        },
        "personas": personas,
        "invalid_records": invalid,
        "daily_batch": {
            "file": "daily_t7_batch.csv",
            "rows": int(len(daily)),
            "valid_rows": int(len(batch)),
            "invalid_rows": int(len(bad)),
            "expected_actions": queue["action"].value_counts().sort_index().to_dict(),
        },
    }
    _write_json(out_dir / "personas.json", manifest)
    daily.to_csv(out_dir / "daily_t7_batch.csv", index=False)
    pd.DataFrame(sends).to_csv(out_dir / "actions_taken.csv", index=False)
    labels.to_csv(out_dir / "renewal_outcomes.csv", index=False)
    (out_dir / "README.md").write_text(_readme(manifest, len(sends)), encoding="utf-8")
    return manifest


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    p = path or (USE_CASE_DIR / "personas.json")
    return json.loads(p.read_text(encoding="utf-8"))


def check_use_cases(out_dir: Path = USE_CASE_DIR) -> list[str]:
    """Re-verify every committed file of the pack against the current bundle."""
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
        if _read_json(out_dir / "records" / f"{p['id']}.json") != p["record"]:
            problems.append(f"records/{p['id']}.json differs from personas.json")
        got = _score_one(p["record"], bundle, calibrator, metrics)
        exp = p["expected"]
        drivers = [d["feature"] for d in p["top_drivers"]]
        if (
            (got["band"], got["action"]) != (exp["band"], exp["action"])
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
        out_dir / "daily_t7_batch.csv", dtype={"user_id": str, "user_name": str, "plan_tier": str}
    )
    scores = score_feature_frame(batch, bundle, calibrator, metrics, scored_at=SCORED_AT)
    db = manifest["daily_batch"]
    got_actions = scores["action"].value_counts().sort_index().to_dict()
    if got_actions != db["expected_actions"] or len(scores.attrs["rejected"]) != db["invalid_rows"]:
        problems.append(
            f"daily_batch: expected {db['expected_actions']} + {db['invalid_rows']} rejects, "
            f"got {got_actions} + {len(scores.attrs['rejected'])} rejects"
        )

    suggested = dict(zip(scores["user_id"], scores["action"]))
    personas = {p["record"]["user_id"]: p for p in manifest["personas"]}
    sends = pd.read_csv(out_dir / "actions_taken.csv", dtype=str).fillna("")
    expect_ids = {u for u, a in suggested.items() if a != NO_ACTION} | set(personas)
    if set(sends["user_id"]) != expect_ids or sends["user_id"].duplicated().any():
        problems.append("actions_taken.csv does not cover exactly the non-no_action queue + scenarios")
    for d in sends.to_dict(orient="records"):
        persona = personas.get(d["user_id"])
        want = persona["executed"]["action_taken"] if persona else suggested.get(d["user_id"])
        if d["action_taken"] != want:
            problems.append(f"actions_taken: {d['user_id']} took {d['action_taken']!r}, expected {want!r}")

    labels = pd.read_csv(out_dir / "renewal_outcomes.csv", dtype={"user_id": str})
    if set(labels["user_id"]) != set(suggested) or labels["user_id"].duplicated().any():
        problems.append("renewal_outcomes.csv does not cover exactly the valid batch subscribers")
    if not set(labels["churned"].unique()) <= {0, 1}:
        problems.append("renewal_outcomes.csv churned must be 0/1")
    return problems


def _readme(manifest: dict[str, Any], n_sends: int) -> str:
    tau = manifest["best_f1_threshold"]
    lines = [
        "# Use-case data pack (seed 42)",
        "",
        "_Generated by `python -m retention_radar.cli.build_use_cases`. Do not edit by hand._",
        "",
        "One day of the renewal business for a monthly AI coding assistant plan: the "
        "subscribers whose renewal is seven days out, what the policy suggests for each, "
        "what the messaging tool actually sent, and whether each subscriber renewed. "
        "Scenario rows are real **test-split** rows (never used for training, calibration "
        "or τ); Santosh and Arjun are worked examples scored at T-7. Expected results are what "
        f"the committed `models/` bundle returns (τ = {tau}); "
        "`python -m retention_radar.cli.build_use_cases --check` re-verifies them. "
        "Synthetic data only, no real PII.",
        "",
        "## Scenarios",
        "",
        "| Scenario | Record | Raw → calibrated | Band | Action | Actually done |",
        "|----------|--------|------------------|------|--------|---------------|",
    ]
    for p in manifest["personas"]:
        e, x = p["expected"], p["executed"]
        lines.append(
            f"| **{p['title']}** | [`records/{p['id']}.json`](records/{p['id']}.json) "
            f"(`{p['record']['user_id']}`) | {e['p_raw']:.3f} → {e['p_cal']:.3f} | "
            f"{e['band']} | {e['action']} | {x['action_taken']} |"
        )
    lines += ["", "Stories and top drivers:", ""]
    for p in manifest["personas"]:
        drivers = ", ".join(f"`{d['feature']}` {d['contribution']:+.2f}" for d in p["top_drivers"])
        lines.append(f"- **{p['title']}**: {p['story']} Top drivers (SHAP, log-odds): {drivers}.")
    lines += [
        "",
        "## Records the service must hold (never scored or queued)",
        "",
        "| File | Problem |",
        "|------|---------|",
    ]
    for i in manifest["invalid_records"]:
        lines.append(f"| [`invalid/{i['id']}.json`](invalid/{i['id']}.json) | {i['problem']} |")
    db = manifest["daily_batch"]
    acts = ", ".join(f"{k}: {v}" for k, v in db["expected_actions"].items())
    cal = manifest["calendar"]
    lines += [
        "",
        "## Files",
        "",
        "| File | Use |",
        "|------|-----|",
        f"| `daily_t7_batch.csv` | {db['rows']} rows = {db['valid_rows']} subscribers renewing in "
        f"seven days (a seeded test-split sample + every scenario) + {db['invalid_rows']} invalid rows. "
        f"Expected queue: {acts}. No label column, like production input. |",
        f"| `actions_taken.csv` | {n_sends} rows from the messaging tool's send export at "
        f"`{cal['acted_at']}`: every queue row that was not `no_action`, plus every scenario. "
        "Includes holdout rows (nothing sent) and one suppression. |",
        f"| `renewal_outcomes.csv` | Renewal outcome observed at `{cal['outcomes_observed_at']}` for "
        "every valid batch subscriber. Generator ground truth, except that a would-be lapse among "
        "subscribers actually sent a playbook is flipped to a renewal with probability equal to the "
        "playbook's **assumed** effect. Simulated, not measured. |",
        "| `personas.json` | Machine-readable manifest: stories, expected outputs, drivers, records. |",
        "",
        "## Walk the whole service",
        "",
        "```bash",
        "make use-cases            # queue → packets → action log → outcomes + lift vs holdout",
        "# or step by step (fresh output dir; the action import skips already-logged rows):",
        "rm -rf artifacts/use_cases && mkdir -p artifacts/use_cases",
        "python -m retention_radar.cli.batch_score --csv data/use_cases/daily_t7_batch.csv \\",
        f"  --out artifacts/use_cases/queue.csv --scored-at {cal['scored_at']}",
        "python -m retention_radar.cli.single_record --dir data/use_cases/records \\",
        "  --out artifacts/use_cases/packets.jsonl",
        "python -m retention_radar.cli.action_log --from-scores artifacts/use_cases/queue.csv \\",
        "  --decisions data/use_cases/actions_taken.csv --log artifacts/use_cases/action_log.csv",
        "python -m retention_radar.cli.outcomes --log artifacts/use_cases/action_log.csv \\",
        "  --labels data/use_cases/renewal_outcomes.csv --out artifacts/use_cases/outcomes.csv",
        "curl -s localhost:8000/v1/churn/score -H 'content-type: application/json' \\",
        "  -d @data/use_cases/records/santosh_capped_pro.json",
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
                f"records, daily batch {m['daily_batch']['expected_actions']}"
            )
        return 1 if problems else 0
    manifest = build_use_cases(out_dir)
    for p in manifest["personas"]:
        e = p["expected"]
        print(
            f"  {p['id']:<24} {p['record']['user_id']:<11} p_cal={e['p_cal']:.3f} "
            f"{e['band']:<6} {e['action']}"
        )
    print(f"Wrote use-case pack → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
