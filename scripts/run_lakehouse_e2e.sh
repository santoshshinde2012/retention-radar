#!/usr/bin/env bash
# Full use-case E2E: lakehouse sample → gold export → sync → Retention Radar train/infer.
# Usage:
#   ./scripts/run_lakehouse_e2e.sh [/path/to/local-data-lakehouse]
#
# Note: this overwrites local models/metrics.json with the lakehouse train.
# Published seed-42 ladder stays synthetic — archive lake metrics under
# artifacts/lakehouse_run/ and restore models from git before publishing docs.
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

cd "$LAKE"
echo "==> [lakehouse] churn-sample (N_USERS=${N_USERS:-5000})"
N_USERS="${N_USERS:-5000}" CHURN_SEED="${CHURN_SEED:-42}" python3 scripts/generate_churn_sample.py
echo "==> [lakehouse] churn-gold-local"
python3 scripts/build_churn_gold_local.py

cd "$ROOT"
echo "==> [radar] sync exports"
./scripts/sync_lakehouse_exports.sh "$LAKE/data/export"

echo "==> [radar] full pipeline on lakehouse gold"
CHURN_DATA_SOURCE=lakehouse ./scripts/run_all.sh

mkdir -p "$ROOT/artifacts/lakehouse_run"
cp -f "$ROOT/models/metrics.json" "$ROOT/artifacts/lakehouse_run/metrics.json"
cp -f "$ROOT/models/feature_stats.json" "$ROOT/artifacts/lakehouse_run/feature_stats.json" 2>/dev/null || true
cp -f "$ROOT/artifacts/santosh_decision_packet.json" "$ROOT/artifacts/lakehouse_run/santosh_decision_packet.json" 2>/dev/null || true
echo "==> archived lakehouse run → artifacts/lakehouse_run/"
echo "    Tip: git checkout -- models/ MODEL_CARD.md docs/data-dictionary.md"
echo "         restores the published synthetic seed-42 ladder."

echo "==> lakehouse E2E complete"
