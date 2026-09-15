#!/usr/bin/env bash
# End-to-end Retention Radar pipeline.
# Default: synthetic generator (CI-safe).
# Lakehouse: CHURN_DATA_SOURCE=lakehouse|auto with data/external exports present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

SOURCE="${CHURN_DATA_SOURCE:-synthetic}"
export CHURN_DATA_SOURCE="$SOURCE"

echo "==> data source: CHURN_DATA_SOURCE=${CHURN_DATA_SOURCE}"

if [[ "${CHURN_DATA_SOURCE}" == "lakehouse" ]]; then
  if [[ ! -f data/external/churn_user_features.csv ]]; then
    echo "Missing data/external/churn_user_features.csv" >&2
    echo "Sync lakehouse exports first: ./scripts/sync_lakehouse_exports.sh /path/to/export" >&2
    exit 1
  fi
  echo "==> using lakehouse gold export (skip generate_data)"
elif [[ "${CHURN_DATA_SOURCE}" == "auto" && -f data/external/churn_user_features.csv ]]; then
  echo "==> auto: lakehouse export present — skip generate_data"
else
  echo "==> generate_data (synthetic)"
  python -m src.generate_data
fi

echo "==> ingest check"
python -m src.ingest

echo "==> train"
python -m src.train

echo "==> evaluate"
python -m src.evaluate

echo "==> slice_metrics (plan_tier segments)"
python -m src.slice_metrics

echo "==> explain (XGB gain importance)"
python -m src.explain

echo "==> benchmark"
python -m src.benchmark

echo "==> infer santosh"
python -m src.infer --user santosh

echo "==> drift_check (non-fatal)"
python -m src.drift_check || echo "drift_check skipped/failed (non-fatal)"

echo "==> single_record santosh decision packet"
python -m src.single_record --user santosh --out artifacts/santosh_decision_packet.json

echo "==> done (Santosh decision packet written)"
