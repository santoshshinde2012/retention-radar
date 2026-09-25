"""Check that a fresh retrain reproduces the committed seed-42 bundle.

``make reproduce`` retrains into an isolated ``RETENTION_RADAR_ARTIFACT_DIR`` and
then runs this check, which diffs the new ``metrics.json`` (and Santosh packet)
against the committed ``models/metrics.json`` that every published number cites.
Latency is machine-dependent and is reported, never compared.

Examples:
    python -m retention_radar.cli.check_reproduction --artifact-dir artifacts/repro
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from retention_radar import config

# Versions that produced the committed bundle (see requirements.txt / requirements.lock).
PINNED = {
    "xgboost": "3.4.1",
    "scikit-learn": "1.9.1",
    "optuna": "5.0.0",
    "lightgbm": "4.7.0",
    "catboost": "1.2.10",
}
SKIP_PREFIXES = ("latency",)
PACKET_KEYS = ("churn_probability_raw", "churn_probability_calibrated", "risk_band", "best_f1_threshold")


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _equal(a: Any, b: Any, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_equal(x, y, tol) for x, y in zip(a, b))
    return a == b


def compare_metrics(
    reference: dict[str, Any], candidate: dict[str, Any], tol: float = 1e-9
) -> list[tuple[str, Any, Any]]:
    """Return ``(key, reference, candidate)`` for every non-latency mismatch."""
    ref, cand = flatten(reference), flatten(candidate)
    diffs = []
    for key in sorted(set(ref) | set(cand)):
        if key.startswith(SKIP_PREFIXES):
            continue
        a, b = ref.get(key, "<missing>"), cand.get(key, "<missing>")
        if not _equal(a, b, tol):
            diffs.append((key, a, b))
    return diffs


def compare_packets(
    reference: dict[str, Any], candidate: dict[str, Any], tol: float = 1e-9
) -> list[tuple[str, Any, Any]]:
    ref_s, cand_s = reference.get("scoring") or {}, candidate.get("scoring") or {}
    diffs = [
        (f"scoring.{k}", ref_s.get(k), cand_s.get(k))
        for k in PACKET_KEYS
        if not _equal(ref_s.get(k), cand_s.get(k), tol)
    ]
    ref_a = (reference.get("hitl") or {}).get("action")
    cand_a = (candidate.get("hitl") or {}).get("action")
    if ref_a != cand_a:
        diffs.append(("hitl.action", ref_a, cand_a))
    return diffs


def installed_versions() -> dict[str, str]:
    out = {"python": ".".join(map(str, sys.version_info[:3]))}
    for dist in PINNED:
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[dist] = "not installed"
    return out


def _fmt(v: Any) -> str:
    return f"{v:.6g}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diff a fresh retrain against the committed seed-42 bundle"
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        help="RETENTION_RADAR_ARTIFACT_DIR used for the retrain (holds metrics.json)",
    )
    parser.add_argument(
        "--reference",
        default=str(config.SEED_MODELS_DIR / "metrics.json"),
        help="Committed metrics (default: models/metrics.json)",
    )
    parser.add_argument(
        "--reference-packet",
        default=str(config.PROJECT_ROOT / "results" / "santosh_decision_packet.sample.json"),
    )
    parser.add_argument("--tol", type=float, default=1e-9)
    args = parser.parse_args(argv)

    art = Path(args.artifact_dir)
    cand_path = art / "metrics.json"
    if not cand_path.exists():
        print(f"missing {cand_path} — run the retrain first (make reproduce)", file=sys.stderr)
        return 2

    versions = installed_versions()
    print("Environment: " + " · ".join(f"{k} {v}" for k, v in versions.items()))
    drift = [k for k, v in PINNED.items() if versions.get(k) != v]
    if sys.version_info < (3, 12):
        drift.insert(0, "python<3.12")

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    candidate = json.loads(cand_path.read_text(encoding="utf-8"))
    diffs = compare_metrics(reference, candidate, args.tol)

    packet_path = art / "santosh_decision_packet.json"
    ref_packet_path = Path(args.reference_packet)
    if packet_path.exists() and ref_packet_path.exists():
        diffs += compare_packets(
            json.loads(ref_packet_path.read_text(encoding="utf-8")),
            json.loads(packet_path.read_text(encoding="utf-8")),
            args.tol,
        )

    lat = candidate.get("latency") or {}
    if lat:
        print(
            f"Latency (machine-dependent, not compared): p50 {lat.get('latency_ms_p50', 0):.2f} ms "
            f"(committed {(reference.get('latency') or {}).get('latency_ms_p50', 0):.2f} ms)"
        )

    if not diffs:
        n = len([k for k in flatten(reference) if not k.startswith(SKIP_PREFIXES)])
        print(f"REPRODUCED: {n} committed metric values match exactly (tol={args.tol:g}).")
        return 0

    print(f"NOT REPRODUCED: {len(diffs)} value(s) differ from the committed bundle.")
    for key, a, b in diffs[:40]:
        print(f"  {key}: committed {_fmt(a)} -> retrain {_fmt(b)}")
    if len(diffs) > 40:
        print(f"  … {len(diffs) - 40} more")
    if drift:
        print(
            "Likely cause: environment differs from the pinned training stack: "
            + ", ".join(drift)
            + ". Use Python 3.12+ and `pip install -r requirements.txt`."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
