"""Explainability helpers: XGBoost gain importance and optional SHAP."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def xgb_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Return gain-based feature importance sorted descending."""
    booster = model.get_booster()
    score_map = booster.get_score(importance_type="gain")
    rows = []
    for i, name in enumerate(feature_names):
        key_candidates = [name, f"f{i}"]
        gain = 0.0
        for k in key_candidates:
            if k in score_map:
                gain = float(score_map[k])
                break
        rows.append({"feature": name, "gain": gain})
    df = pd.DataFrame(rows).sort_values("gain", ascending=False).reset_index(drop=True)
    return df


def top_contributing_features(
    model,
    X_row: pd.DataFrame,
    feature_names: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Approximate local contribution via SHAP if available, else importance × value.

    Returns list of (feature_name, contribution_score) sorted by |score|.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_row)
        if isinstance(sv, list):
            values = np.asarray(sv[1] if len(sv) == 2 else sv[0])[0]
        else:
            values = np.asarray(sv)[0]
        pairs = list(zip(feature_names, values.tolist()))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return [(n, float(v)) for n, v in pairs[:top_k]]
    except Exception:
        imp = xgb_feature_importance(model, feature_names)
        gains = dict(zip(imp["feature"], imp["gain"]))
        row = X_row.iloc[0]
        pairs = []
        for name in feature_names:
            g = float(gains.get(name, 0.0))
            risk_up = {
                "failed_requests_rate",
                "support_tickets_last_90d",
                "payment_failures_last_90d",
                "last_active_days_ago",
            }
            sign = 1.0 if name in risk_up else -1.0
            score = sign * g * (1.0 + abs(float(row[name])) * 0.01)
            pairs.append((name, score))
        pairs.sort(key=lambda t: abs(t[1]), reverse=True)
        return pairs[:top_k]


def shap_summary_values(model, X: pd.DataFrame) -> Any:
    """Return SHAP values matrix for a batch (or None if shap unavailable)."""
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        return explainer.shap_values(X)
    except Exception:
        return None


def main() -> None:
    """Write gain-based XGBoost importance from the saved bundle."""
    import joblib

    from retention_radar import config

    if not config.MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {config.MODEL_PATH}. Run python -m retention_radar.cli.train first.")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(config.MODEL_PATH)
    df = xgb_feature_importance(bundle["model"], bundle["feature_names"])
    out = config.ARTIFACTS_DIR / "xgb_feature_importance.csv"
    df.to_csv(out, index=False)
    print(df.head(12).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
