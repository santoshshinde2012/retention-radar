"""Score a user JSON with the trained churn model.

Examples:
    python -m src.infer --user santosh
    python -m src.infer --json data/raw/santosh_shinde.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.retention_radar import config
from src.retention_radar.data.ingest import resolve_santosh_json
from src.retention_radar.features.transform import row_to_feature_frame
from src.retention_radar.serving.explain import top_contributing_features
from src.retention_radar.serving.policy import risk_band
from src.retention_radar.serving.scoring import CalibratedScorer
from src.retention_radar.training.calibrate import load_calibrator


def load_payload(path: Path) -> dict:
    """Load the joblib model bundle (estimator + feature names)."""
    return joblib.load(path)


def resolve_user_json(user: str | None, json_path: str | None) -> Path:
    """Resolve ``--user santosh`` or ``--json path`` to a payload file."""
    if json_path:
        return Path(json_path)
    if user and user.lower() in {"santosh", "santosh_shinde", "santosh-shinde"}:
        return resolve_santosh_json()
    raise SystemExit("Provide --user santosh or --json path/to/user.json")


def predict_user(
    user_dict: dict, model_bundle: dict, calibrator=None, top_k: int = 5
) -> dict:
    """Score one user through the transformer + classifier + optional calibrator."""
    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    X = row_to_feature_frame(user_dict)[feature_names]
    scorer = CalibratedScorer(model, calibrator)
    raw_arr, cal_arr, display_arr = scorer.score(X)
    raw_proba = float(raw_arr[0])
    cal_proba = float(cal_arr[0]) if cal_arr is not None else None
    display = float(display_arr[0])
    top = top_contributing_features(model, X, feature_names, top_k=top_k)
    return {
        "user_id": user_dict.get("user_id"),
        "user_name": user_dict.get("user_name"),
        "churn_probability": display,
        "churn_probability_raw": raw_proba,
        "churn_probability_calibrated": cal_proba,
        "risk_band": risk_band(display),
        "top_features": top,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Infer churn probability for a user")
    parser.add_argument("--user", type=str, default=None, help="Shortcut: santosh")
    parser.add_argument("--json", type=str, default=None, help="Path to user JSON")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to joblib model"
    )
    args = parser.parse_args(argv)

    json_path = resolve_user_json(args.user, args.json)
    if not json_path.exists():
        raise SystemExit(f"User JSON not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        user_dict = json.load(f)

    model_path = Path(args.model) if args.model else config.MODEL_PATH
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}. Run python -m src.train first."
        )

    bundle = load_payload(model_path)
    calibrator = load_calibrator(config.CALIBRATOR_PATH)
    result = predict_user(user_dict, bundle, calibrator=calibrator)

    print(f"User: {result['user_name']} ({result['user_id']})")
    raw = result["churn_probability_raw"]
    cal = result["churn_probability_calibrated"]
    if cal is not None:
        print(
            f"P(churn) raw={raw:.4f}  calibrated={cal:.4f}  "
            f"risk={result['risk_band']}"
        )
    else:
        print(f"P(churn) = {raw:.4f}  risk={result['risk_band']}")
    print("Top contributing features:")
    for name, score in result["top_features"]:
        print(f"  {name}: {score:+.4f}")


if __name__ == "__main__":
    main()
