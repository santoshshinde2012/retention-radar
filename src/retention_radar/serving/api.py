"""Thin local FastAPI serve path (teaching only — no auth).

Routes (all HITL: ``auto_action`` is always ``none``):

- ``POST /v1/churn/score``   one 24-field record → score, band, suggested action,
  rationale (``?shap=true`` adds top drivers). Invalid records → 422 with the
  hold reason; they are never scored. Conceptual alias: ``POST /v1/churn:score``.
- ``POST /v1/churn/batch``   ``{"records": [...]}`` → queue sorted by risk + rejects.
- ``POST /v1/churn/reviews`` a human decision on a user the service scored; score
  fields come from the service's own prediction log, never from the client.
- ``GET  /healthz``

Run locally:
    uvicorn retention_radar.serving.api:app --reload --app-dir src
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from retention_radar import config
from retention_radar.serving.batch_score import (
    load_metrics,
    resolve_model_version,
    score_payloads,
    split_valid_payloads,
)
from retention_radar.serving.hitl_log import (
    DEFAULT_LOG_PATH,
    append_hitl_row,
    row_from_score_record,
)
from retention_radar.serving.infer import predict_user
from retention_radar.serving.packet import normalize_record, validate_payload
from retention_radar.serving.policy import HitlDecisionPolicy, validation_hold
from retention_radar.training.calibrate import load_calibrator

app = FastAPI(
    title="Retention Radar (teaching)",
    description=(
        "Thin local churn-ranking service. Teaching-only — **no auth**. "
        "Scores go to a human (`auto_action: none`). "
        "Preferred path: ``POST /v1/churn/score``. "
        "Conceptual GCP-style name: ``POST /v1/churn:score`` (same contract)."
    ),
    version="0.2.0",
)


class ChurnScoreRequest(BaseModel):
    """24-field inference contract — OpenAPI documentation only.

    The route validates the raw JSON object with the shared ``normalize_record`` +
    ``validate_payload`` contract (the same one batch / CLI / UI use), so every
    surface accepts and refuses exactly the same records. Extra fields are ignored.
    """

    model_config = ConfigDict(extra="allow")

    user_id: str = Field(..., min_length=1)
    user_name: str = Field(..., min_length=1)
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
    user_id: Optional[str] = None
    p_raw: float
    p_cal: float
    band: str
    hitl_action: str
    rationale: str
    best_f1_threshold: float
    model_version: str
    scored_at: str
    auto_action: str = Field(default="none")
    validation_warnings: List[str] = Field(default_factory=list)
    shap_top: Optional[List[Dict[str, Any]]] = None


class BatchRequest(BaseModel):
    records: List[Any] = Field(
        ..., description="24-field records; invalid or non-object items are rejected, not scored"
    )


class BatchResponse(BaseModel):
    model_version: str
    scored_at: str
    auto_action: str = "none"
    queue: List[Dict[str, Any]]
    rejected: List[Dict[str, Any]]


class ReviewRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    user_id: str = Field(..., min_length=1)
    reviewer: str = Field(..., min_length=1)
    action_taken: str = Field(..., min_length=1)
    notes: str = ""


@lru_cache(maxsize=1)
def _load_serve_bundle() -> tuple[dict, Any, dict]:
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
    bundle = joblib.load(config.MODEL_PATH)
    calibrator = load_calibrator(config.CALIBRATOR_PATH)
    metrics = load_metrics()
    return bundle, calibrator, metrics


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prediction_log_dir() -> Path:
    return config.runtime_log_dir() / "prediction_log"


def _append_prediction_log(record: dict[str, Any]) -> None:
    log_dir = _prediction_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = log_dir / f"scores_{day}.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _latest_logged_score(user_id: str) -> dict[str, Any] | None:
    """Most recent score this service logged for ``user_id`` (None if never scored)."""
    log_dir = _prediction_log_dir()
    if not log_dir.exists():
        return None
    for path in sorted(log_dir.glob("scores_*.jsonl"), reverse=True):
        latest = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a truncated line (crash / full disk) must not block reviews
                if isinstance(rec, dict) and rec.get("user_id") == user_id:
                    latest = rec
        if latest is not None:
            return latest
    return None


def _hold_422(errors: list[str]) -> HTTPException:
    return HTTPException(status_code=422, detail={"errors": errors, "hitl": validation_hold(errors)})


@app.exception_handler(RequestValidationError)
async def _request_validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """On the scoring route, body-level 422s (not JSON, not an object) carry the hold block too."""
    if request.url.path != "/v1/churn/score":
        return await request_validation_exception_handler(request, exc)
    errors = [
        f"{'.'.join(str(p) for p in e.get('loc', [])[1:]) or 'body'}: {e.get('msg')}"
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": {"errors": errors, "hitl": validation_hold(errors)}})


def score_payload(
    payload: dict[str, Any],
    *,
    include_shap: bool = False,
    log_prediction: bool = True,
) -> dict[str, Any]:
    """Validate → score one user; invalid input raises 422 and is never scored."""
    try:
        bundle, calibrator, metrics = _load_serve_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    record, notes = normalize_record(payload)
    validation = validate_payload(record)
    if not validation["ok"]:
        raise _hold_422(validation["errors"])

    result = predict_user(record, bundle, calibrator=calibrator, explain=include_shap)
    threshold = float(metrics.get("best_f1_threshold", 0.5))
    p_raw = float(result["churn_probability_raw"])
    p_cal = float(
        result["churn_probability_calibrated"]
        if result["churn_probability_calibrated"] is not None
        else result["churn_probability"]
    )
    band = result["risk_band"]
    hitl = HitlDecisionPolicy().decide(p_cal, threshold, band)

    out: dict[str, Any] = {
        "user_id": result.get("user_id"),
        "p_raw": round(p_raw, 6),
        "p_cal": round(p_cal, 6),
        "band": band,
        "hitl_action": hitl["action"],
        "rationale": hitl["rationale"],
        "best_f1_threshold": threshold,
        "model_version": resolve_model_version(metrics),
        "scored_at": _now(),
        "auto_action": "none",
        "validation_warnings": notes + validation["warnings"],
        "shap_top": None,
    }
    if include_shap:
        out["shap_top"] = [
            {"feature": n, "contribution": float(v)} for n, v in result["top_features"]
        ]
    if log_prediction:
        _append_prediction_log({k: v for k, v in out.items() if k != "shap_top"})
    return out


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/v1/churn/score",
    response_model=ChurnScoreResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": ChurnScoreRequest.model_json_schema()}},
        }
    },
)
def churn_score(
    body: Dict[str, Any] = Body(..., description="24-field record (see ChurnScoreRequest)"),
    shap: bool = Query(False, description="Include top SHAP / driver features"),
    log: bool = Query(True, description="Append JSONL under <log dir>/prediction_log/"),
) -> dict[str, Any]:
    """Score one user. Alias concept: ``POST /v1/churn:score`` (same body)."""
    return score_payload(body, include_shap=shap, log_prediction=log)


@app.post("/v1/churn/batch", response_model=BatchResponse)
def churn_batch(
    body: BatchRequest,
    log: bool = Query(True, description="Append each scored row to the prediction log"),
) -> dict[str, Any]:
    """Score many users → queue sorted by calibrated risk (rank 1 first) + rejects."""
    try:
        bundle, calibrator, metrics = _load_serve_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if len(body.records) > 10_000:
        raise HTTPException(status_code=413, detail="max 10000 records per request; use cli.batch_score")
    valid, rejected = split_valid_payloads(body.records)
    when = _now()
    queue = score_payloads(valid, bundle, calibrator, metrics, scored_at=when)
    rows = [{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()} for r in queue.to_dict(orient="records")]
    if log:
        for r in rows:
            _append_prediction_log({k: v for k, v in r.items() if k != "rank"})
    return {
        "model_version": resolve_model_version(metrics),
        "scored_at": when,
        "auto_action": "none",
        "queue": rows,
        "rejected": rejected,
    }


@app.post("/v1/churn/reviews")
def churn_review(body: ReviewRequest) -> dict[str, Any]:
    """Log a human decision for a user this service already scored (HITL review log)."""
    scored = _latest_logged_score(body.user_id)
    if scored is None:
        raise HTTPException(
            status_code=404,
            detail=f"user_id {body.user_id!r} has no logged score; score it first (log=true)",
        )
    row = row_from_score_record(
        scored,
        reviewer=body.reviewer.strip(),
        action_taken=body.action_taken.strip(),
        notes=body.notes.strip(),
    )
    path = append_hitl_row(config.runtime_log_dir() / DEFAULT_LOG_PATH.name, row)
    return {"logged": row, "log_path": str(path), "auto_action": "none"}


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
