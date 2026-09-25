#!/usr/bin/env bash
# End-to-end Retention Radar pipeline.
# Default: synthetic generator (CI-safe).
# Lakehouse: CHURN_DATA_SOURCE=lakehouse|auto with data/external exports present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

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
  python -m retention_radar.cli.generate_data
fi

echo "==> ingest check"
python -m retention_radar.cli.ingest

echo "==> train"
python -m retention_radar.cli.train

echo "==> evaluate"
python -m retention_radar.cli.evaluate

echo "==> slice_metrics (plan_tier segments)"
python -m retention_radar.cli.slice_metrics

echo "==> explain (XGB gain importance)"
python -m retention_radar.cli.explain

echo "==> benchmark"
python -m retention_radar.cli.benchmark

echo "==> infer santosh"
python -m retention_radar.cli.infer --user santosh

echo "==> drift_check (non-fatal)"
python -m retention_radar.cli.drift_check || echo "drift_check skipped/failed (non-fatal)"

echo "==> single_record santosh decision packet"
# No --out: default resolves to config.ARTIFACTS_DIR, so RETENTION_RADAR_ARTIFACT_DIR
# (lakehouse E2E) keeps its packet under artifacts/lakehouse_run/ instead of
# overwriting the synthetic artifacts/santosh_decision_packet.json.
python -m retention_radar.cli.single_record --user santosh

echo "==> done (Santosh decision packet written)"
