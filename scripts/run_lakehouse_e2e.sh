#!/usr/bin/env bash
# Full use-case E2E: lakehouse sample → gold export → sync → Retention Radar train/infer.
# Usage:
#   ./scripts/run_lakehouse_e2e.sh [/path/to/local-data-lakehouse]
#
# Isolation: sets RETENTION_RADAR_ARTIFACT_DIR=artifacts/lakehouse_run so train /
# evaluate / docs_gen / packets NEVER overwrite committed seed-42 models/ or
# published docs/MODEL_CARD.md / data-dictionary.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAKE="${1:-${LAKEHOUSE_ROOT:-}}"
if [[ -z "${LAKE}" ]]; then
  for cand in "$ROOT/../local-data-lakehouse" /workspace/local-data-lakehouse; do
    if [[ -d "$cand/scripts" ]]; then LAKE="$cand"; break; fi
  done
fi
if [[ -z "${LAKE}" || ! -d "$LAKE" ]]; then
  echo "Pass lakehouse root: $0 /path/to/local-data-lakehouse" >&2
  exit 2
fi

LAKE_ART="$ROOT/artifacts/lakehouse_run"
mkdir -p "$LAKE_ART"
export RETENTION_RADAR_ARTIFACT_DIR="$LAKE_ART"

cd "$LAKE"
echo "==> [lakehouse] churn-sample (N_USERS=${N_USERS:-5000})"
N_USERS="${N_USERS:-5000}" CHURN_SEED="${CHURN_SEED:-42}" python3 scripts/generate_churn_sample.py
echo "==> [lakehouse] churn-gold-local"
python3 scripts/build_churn_gold_local.py

cd "$ROOT"
echo "==> [radar] sync exports"
./scripts/sync_lakehouse_exports.sh "$LAKE/data/export"

echo "==> [radar] full pipeline on lakehouse gold → ${RETENTION_RADAR_ARTIFACT_DIR}"
echo "    (committed models/ + docs/MODEL_CARD.md left untouched)"
CHURN_DATA_SOURCE=lakehouse ./scripts/run_all.sh

# Dual-world cite: refresh committed summary from this isolated run (best-effort).
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python3 - <<'PY'
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

art = Path("artifacts/lakehouse_run")
metrics_path = art / "metrics.json"
packet_path = art / "santosh_decision_packet.json"
summary_path = Path("results/lakehouse-e2e-summary.json")

metrics = {}
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

packet = {}
if packet_path.exists():
    packet = json.loads(packet_path.read_text(encoding="utf-8"))

scoring = packet.get("scoring") or {}
hitl = packet.get("hitl") or {}
payload = packet.get("payload") or {}

n_total = None
if metrics.get("n_train") is not None:
    n_total = int(metrics.get("n_train") or 0) + int(metrics.get("n_val") or 0) + int(
        metrics.get("n_test") or 0
    )
source = f"lakehouse gold N={n_total} seed=42" if n_total else "lakehouse gold seed=42"
tuned_val = metrics.get("tuned_val") or {}
cal_test = metrics.get("calibrated_test") or {}
eval_test = metrics.get("evaluate_test") or {}

summary = {
    "verified_at": date.today().isoformat(),
    "source": source,
    "n_train": metrics.get("n_train"),
    "n_test": metrics.get("n_test"),
    "churn_rate_train": metrics.get("churn_rate_train"),
    "best_optuna_auc_val": metrics.get("best_optuna_auc") or tuned_val.get("roc_auc"),
    "calibrated_test_roc_auc": cal_test.get("roc_auc") or eval_test.get("test_roc_auc"),
    "best_f1_threshold": eval_test.get("best_f1_threshold") or scoring.get("best_f1_threshold"),
    "santosh": {
        "user_id": packet.get("user_id") or payload.get("user_id"),
        "name": packet.get("user_name") or payload.get("user_name"),
        "p_churn_raw": scoring.get("churn_probability_raw"),
        "p_churn_calibrated": scoring.get("churn_probability_calibrated")
            or scoring.get("churn_probability"),
        "risk": scoring.get("risk_band"),
        "hitl": hitl.get("action"),
    },
    "artifact_dir": str(art),
    "note": (
        "Lakehouse E2E writes only under artifacts/lakehouse_run/ via "
        "RETENTION_RADAR_ARTIFACT_DIR. Committed models/ + docs/MODEL_CARD.md "
        "are the published synthetic seed-42 ladder and must not be replaced."
    ),
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"==> wrote {summary_path}")
print(f"==> lakehouse artifacts → {art}/")
PY

echo "==> lakehouse E2E complete (seed-42 models/ untouched)"
