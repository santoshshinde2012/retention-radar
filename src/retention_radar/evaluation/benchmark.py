"""Single-record inference latency benchmark (CPU).

Run:
    python -m retention_radar.cli.benchmark
"""

from __future__ import annotations

import json
import time

import joblib
import numpy as np

from retention_radar import config
from retention_radar.data.ingest import resolve_santosh_json
from retention_radar.features.transform import row_to_feature_frame
from retention_radar.training.calibrate import load_calibrator


def _percentile(arr: np.ndarray, q: float) -> float:
    return float(np.percentile(arr, q))


def benchmark_latency(
    model,
    feature_names: list[str],
    X_row,
    calibrator=None,
    warmup: int | None = None,
    runs: int | None = None,
) -> dict:
    """Time predict_proba (+ optional calibrate) for one row."""
    warmup = config.LATENCY_WARMUP if warmup is None else warmup
    runs = config.LATENCY_RUNS if runs is None else runs

    def once():
        proba = model.predict_proba(X_row)[:, 1]
        if calibrator is not None:
            calibrator.transform(proba)
        return proba

    for _ in range(warmup):
        once()

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        once()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times_ms, dtype=float)
    return {
        "n_warmup": warmup,
        "n_runs": runs,
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_p50": _percentile(arr, 50),
        "latency_ms_p95": _percentile(arr, 95),
        "latency_ms_p99": _percentile(arr, 99),
        "latency_ms_min": float(arr.min()),
        "latency_ms_max": float(arr.max()),
        "calibrated": calibrator is not None,
    }


def main() -> None:
    if not config.MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {config.MODEL_PATH}. Run train first.")
    santosh_path = resolve_santosh_json()
    if not santosh_path.exists():
        raise SystemExit(f"Santosh JSON not found: {santosh_path}")

    bundle = joblib.load(config.MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    calibrator = load_calibrator(config.CALIBRATOR_PATH)

    with open(santosh_path, encoding="utf-8") as f:
        user = json.load(f)
    X = row_to_feature_frame(user)[feature_names]

    print(
        f"Benchmarking single-record latency "
        f"(warmup={config.LATENCY_WARMUP}, runs={config.LATENCY_RUNS})..."
    )
    result = benchmark_latency(model, feature_names, X, calibrator=calibrator)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    out = {}
    if config.METRICS_PATH.exists():
        with open(config.METRICS_PATH, encoding="utf-8") as f:
            out = json.load(f)
    out["latency"] = result
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Updated {config.METRICS_PATH}")

    from retention_radar.docs_gen import write_model_card

    write_model_card(out)


if __name__ == "__main__":
    main()
