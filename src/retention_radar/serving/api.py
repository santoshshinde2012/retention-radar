"""Thin local FastAPI serve path (teaching only — no auth).

Preferred route: ``POST /v1/churn/score``
Conceptual alias documented as ``POST /v1/churn:score`` (colon awkward in FastAPI).

Run locally:
    uvicorn retention_radar.serving.api:app --reload --app-dir src
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from retention_radar import config
from retention_radar.serving.batch_score import load_metrics, resolve_model_version
from retention_radar.serving.explain import top_contributing_features
from retention_radar.serving.infer import predict_user
from retention_radar.serving.policy import HitlDecisionPolicy
from retention_radar.training.calibrate import load_calibrator

app = FastAPI(
    title="Retention Radar (teaching)",
    description=(
        "Thin local score API. Teaching-only — **no auth**. "
        "Preferred path: ``POST /v1/churn/score``. "
        "Conceptual GCP-style name: ``POST /v1/churn:score`` (same contract)."
    ),
    version="0.1.0",
)


class ChurnScoreRequest(BaseModel):
    """22-field inference contract (Santosh-shaped record OK)."""

    model_config = ConfigDict(extra="allow")

    user_id: str
    user_name: str = ""
    days_since_signup: float
    sessions_last_7d: float
    sessions_last_30d: float
    avg_session_minutes: float
    models_used_count: float
    api_calls_last_30d: float
    tokens_consumed_last_30d: float
    tools_used_count: float
    failed_requests_rate: float
    support_tickets_last_90d: float
    plan_tier: str
    payment_failures_last_90d: float
    feature_adoption_score: float
    nps_score: float
    last_active_days_ago: float
    weekend_usage_ratio: float
    engagement_trend: float
    spend_usd_last_30d: float
    days_until_renewal: float
    agent_runs_last_30d: float
    ide_plugin_sessions_last_30d: float
    seat_utilization: float


class ChurnScoreResponse(BaseModel):
    user_id: str | None = None
    p_raw: float
    p_cal: float
    band: str
    hitl_action: str
    model_version: str
    auto_action: str = Field(default="none")
    shap_top: list[dict[str, Any]] | None = None


@lru_cache(maxsize=1)
def _load_serve_bundle() -> tuple[dict, Any, dict]:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
    bundle = joblib.load(config.MODEL_PATH)
    calibrator = load_calibrator(config.CALIBRATOR_PATH)
    metrics = load_metrics()
    return bundle, calibrator, metrics


def _append_prediction_log(record: dict[str, Any]) -> None:
    log_dir = config.ARTIFACTS_DIR / "prediction_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = log_dir / f"scores_{day}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def score_payload(
    payload: dict[str, Any],
    *,
    include_shap: bool = False,
    log_prediction: bool = True,
) -> dict[str, Any]:
    """Score one user dict; always sets ``auto_action=none``."""
    try:
        bundle, calibrator, metrics = _load_serve_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        result = predict_user(payload, bundle, calibrator=calibrator)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"score failed: {exc}") from exc

    threshold = float(metrics.get("best_f1_threshold", 0.5))
    p_raw = float(result["churn_probability_raw"])
    p_cal = float(
        result["churn_probability_calibrated"]
        if result["churn_probability_calibrated"] is not None
        else result["churn_probability"]
    )
    band = result["risk_band"]
    hitl = HitlDecisionPolicy().decide(p_cal, threshold, band)
    model_version = resolve_model_version(metrics)
    scored_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    out: dict[str, Any] = {
        "user_id": result.get("user_id"),
        "p_raw": round(p_raw, 6),
        "p_cal": round(p_cal, 6),
        "band": band,
        "hitl_action": hitl["action"],
        "model_version": model_version,
        "auto_action": "none",
        "shap_top": None,
    }

    if include_shap:
        out["shap_top"] = [
            {"feature": n, "contribution": float(v)} for n, v in result["top_features"]
        ]

    if log_prediction:
        _append_prediction_log({**out, "scored_at": scored_at})

    return out


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/churn/score", response_model=ChurnScoreResponse)
def churn_score(
    body: ChurnScoreRequest,
    shap: bool = Query(False, description="Include top SHAP / driver features"),
    log: bool = Query(True, description="Append JSONL under artifacts/prediction_log/"),
) -> dict[str, Any]:
    """Score one user. Alias concept: ``POST /v1/churn:score`` (same body)."""
    payload = body.model_dump()
    return score_payload(payload, include_shap=shap, log_prediction=log)


def main() -> None:
    """``python -m retention_radar.serving.api`` → uvicorn on :8000."""
    import uvicorn

    uvicorn.run(
        "retention_radar.serving.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
