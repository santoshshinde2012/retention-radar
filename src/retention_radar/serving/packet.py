"""Single-record decision packet: validate → score → explain → cohort → HITL.

Examples:
    python -m retention_radar.cli.single_record --user santosh \\
        --out artifacts/santosh_decision_packet.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from retention_radar import config
from retention_radar.data.ingest import resolve_users_csv
from retention_radar.features.transform import row_to_feature_frame
from retention_radar.serving.infer import predict_user, resolve_user_json
from retention_radar.serving.policy import HitlDecisionPolicy
from retention_radar.training.calibrate import load_calibrator


def load_metrics(path: Path | None = None) -> dict:
    p = path or config.METRICS_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_feature_stats(path: Path | None = None) -> dict:
    p = path or config.FEATURE_STATS_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def validate_payload(payload: dict) -> dict[str, Any]:
    """Validate inference payload against required keys, ranges, plan_tier enum.

    Returns a validation block (never raises for soft range issues — collects errors).
    """
    errors: list[str] = []
    warnings: list[str] = []

    missing = [
        k
        for k in config.INFERENCE_REQUIRED_KEYS
        if k not in payload or payload[k] is None
    ]
    if missing:
        errors.append(f"missing required keys (nulls fail loud): {missing}")

    plan = payload.get("plan_tier")
    if plan is not None and plan not in config.PLAN_TIER_ORDER:
        errors.append(f"plan_tier must be one of {config.PLAN_TIER_ORDER}, got {plan!r}")

    for key, (lo, hi) in config.FEATURE_RANGES.items():
        if key not in payload:
            continue
        try:
            val = float(payload[key])
        except (TypeError, ValueError):
            errors.append(f"{key} must be numeric, got {payload[key]!r}")
            continue
        if val < lo or val > hi:
            errors.append(f"{key}={val} outside allowed range [{lo}, {hi}]")

    if (
        "sessions_last_7d" in payload
        and "sessions_last_30d" in payload
        and "engagement_trend" in payload
        and not missing
    ):
        s7 = float(payload["sessions_last_7d"])
        s30 = float(payload["sessions_last_30d"])
        expected = s7 / max(1.0, s30 / 4.0)
        actual = float(payload["engagement_trend"])
        if abs(expected - actual) > 0.25:
            warnings.append(
                f"engagement_trend={actual:.4f} diverges from "
                f"sessions formula ≈{expected:.4f} (tolerance 0.25)"
            )

    schema_ok = None
    if config.USER_RECORD_SCHEMA_PATH.exists() and not missing:
        try:
            import jsonschema  # type: ignore

            with open(config.USER_RECORD_SCHEMA_PATH, encoding="utf-8") as f:
                schema = json.load(f)
            jsonschema.validate(instance=payload, schema=schema)
            schema_ok = True
        except ImportError:
            schema_ok = None
        except Exception as exc:  # noqa: BLE001
            schema_ok = False
            errors.append(f"jsonschema: {exc}")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "schema_validated": schema_ok,
        "required_keys_present": len(missing) == 0,
    }


def flag_outliers(payload: dict, feature_stats: dict) -> list[dict]:
    """Flag features outside training p01–p99."""
    flags = []
    if not feature_stats:
        return flags
    try:
        X = row_to_feature_frame(payload)
    except Exception:
        return flags
    for col in config.MODEL_FEATURE_COLUMNS:
        if col not in feature_stats or col not in X.columns:
            continue
        val = float(X.iloc[0][col])
        st = feature_stats[col]
        lo, hi = st["p01"], st["p99"]
        if val < lo or val > hi:
            flags.append(
                {
                    "feature": col,
                    "value": val,
                    "p01": lo,
                    "p99": hi,
                    "side": "low" if val < lo else "high",
                }
            )
    return flags


def cohort_percentiles(
    payload: dict,
    users_csv: Path | None = None,
    features: list[str] | None = None,
) -> dict[str, dict]:
    """Percentile rank of each key feature vs population in users.csv."""
    features = features or config.COHORT_COMPARE_FEATURES
    csv_path = users_csv or resolve_users_csv()
    out: dict[str, dict] = {}
    if not csv_path.exists():
        return out
    df = pd.read_csv(csv_path)
    for feat in features:
        if feat not in payload or feat not in df.columns:
            continue
        val = float(payload[feat])
        col = df[feat].astype(float)
        pct = float((col <= val).mean() * 100.0)
        out[feat] = {
            "value": val,
            "percentile": round(pct, 2),
            "population_median": float(col.median()),
            "population_mean": float(col.mean()),
        }
    return out


def build_decision_packet(
    payload: dict,
    model_bundle: dict | None = None,
    calibrator=None,
    metrics: dict | None = None,
    feature_stats: dict | None = None,
    policy: HitlDecisionPolicy | None = None,
) -> dict[str, Any]:
    """Full validate → score → explain → cohort → HITL decision packet."""
    metrics = metrics if metrics is not None else load_metrics()
    feature_stats = feature_stats if feature_stats is not None else load_feature_stats()
    policy = policy or HitlDecisionPolicy()

    validation = validate_payload(payload)
    if model_bundle is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
        model_bundle = joblib.load(config.MODEL_PATH)
    if calibrator is None:
        calibrator = load_calibrator(config.CALIBRATOR_PATH)

    score = predict_user(payload, model_bundle, calibrator=calibrator)
    display = float(score["churn_probability"])
    band = score["risk_band"]
    threshold = float(metrics.get("best_f1_threshold", 0.5))
    action = policy.decide(display, threshold, band)
    outliers = flag_outliers(payload, feature_stats)
    cohort = cohort_percentiles(payload)

    packet = {
        "user_id": payload.get("user_id"),
        "user_name": payload.get("user_name"),
        "validation": validation,
        "payload": payload,
        "scoring": {
            "churn_probability_raw": score["churn_probability_raw"],
            "churn_probability_calibrated": score["churn_probability_calibrated"],
            "churn_probability": display,
            "risk_band": band,
            "best_f1_threshold": threshold,
            "half_threshold": 0.5 * threshold,
        },
        "explanation": {
            "top_features": [
                {"feature": n, "contribution": float(v)} for n, v in score["top_features"]
            ],
        },
        "outliers": outliers,
        "cohort_compare": cohort,
        "hitl": action,
        "meta": {
            "model_path": str(config.MODEL_PATH),
            "calibrator_path": str(config.CALIBRATOR_PATH),
            "feature_stats_path": str(config.FEATURE_STATS_PATH),
            "metrics_path": str(config.METRICS_PATH),
            "feature_count": len(config.MODEL_FEATURE_COLUMNS),
            "data_source": config.CHURN_DATA_SOURCE,
            "users_csv": str(resolve_users_csv()),
        },
    }
    return packet


def print_human_summary(packet: dict) -> None:
    s = packet["scoring"]
    h = packet["hitl"]
    v = packet["validation"]
    print(f"User: {packet.get('user_name')} ({packet.get('user_id')})")
    print(
        f"Validation: {'OK' if v.get('ok') else 'FAILED'}  "
        f"errors={len(v.get('errors') or [])}"
    )
    raw = s.get("churn_probability_raw")
    cal = s.get("churn_probability_calibrated")
    print(
        f"P(churn) raw={raw:.4f}  calibrated={cal:.4f}  "
        f"band={s.get('risk_band')}  threshold={s.get('best_f1_threshold'):.4f}"
    )
    print(f"HITL action: {h.get('action')}  (auto={h.get('auto_action')})")
    print(f"Rationale: {h.get('rationale')}")
    print("Top drivers:")
    for item in packet["explanation"]["top_features"][:5]:
        print(f"  {item['feature']}: {item['contribution']:+.4f}")
    if packet["outliers"]:
        print(f"Outliers vs train p01–p99: {len(packet['outliers'])}")
        for o in packet["outliers"][:5]:
            print(f"  {o['feature']}={o['value']} ({o['side']})")
    print("Cohort percentiles (vs active users table):")
    for feat, block in list(packet["cohort_compare"].items())[:6]:
        print(f"  {feat}: pctl={block['percentile']:.1f}  value={block['value']}")


def batch_score_dir(
    dir_path: Path,
    out_path: Path,
    model_bundle: dict,
    calibrator,
) -> int:
    """Score every ``*.json`` in ``dir_path``; write one packet per line to JSONL."""
    files = sorted(
        p for p in dir_path.iterdir() if p.suffix.lower() == ".json" and p.is_file()
    )
    if not files:
        raise SystemExit(f"No JSON files found in {dir_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for fp in files:
            with open(fp, encoding="utf-8") as f:
                payload = json.load(f)
            packet = build_decision_packet(
                payload, model_bundle=model_bundle, calibrator=calibrator
            )
            packet["source_file"] = str(fp)
            fout.write(json.dumps(packet) + "\n")
            n_ok += 1
            action = packet.get("hitl", {}).get("action")
            print(
                f"  [{n_ok}/{len(files)}] {fp.name}: "
                f"P={packet['scoring']['churn_probability']:.4f} "
                f"band={packet['scoring']['risk_band']} action={action}"
            )
    print(f"Wrote {n_ok} packets → {out_path}")
    return n_ok


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build single-record / batch decision packets"
    )
    parser.add_argument("--user", type=str, default=None, help="Shortcut: santosh")
    parser.add_argument("--json", type=str, default=None, help="Path to user JSON")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory of user JSON files → batch JSONL output",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path (JSON for single; JSONL for --dir)",
    )
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args(argv)

    model_path = Path(args.model) if args.model else config.MODEL_PATH
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}. Run train first.")

    bundle = joblib.load(model_path)
    calibrator = load_calibrator(config.CALIBRATOR_PATH)

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            raise SystemExit(f"Not a directory: {dir_path}")
        out_path = (
            Path(args.out)
            if args.out
            else (config.ARTIFACTS_DIR / "batch_decision_packets.jsonl")
        )
        batch_score_dir(dir_path, out_path, bundle, calibrator)
        return

    json_path = resolve_user_json(args.user, args.json)
    if not json_path.exists():
        raise SystemExit(f"User JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    packet = build_decision_packet(payload, model_bundle=bundle, calibrator=calibrator)

    out_path = (
        Path(args.out)
        if args.out
        else (config.ARTIFACTS_DIR / "santosh_decision_packet.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)
    print_human_summary(packet)
    print(f"\nWrote decision packet → {out_path}")


if __name__ == "__main__":
    main()
