#!/usr/bin/env bash
# Walk the whole service on the seed-42 use-case pack (data/use_cases/):
#   weekly batch → ranked review queue (+ rejects) → decision packets per scenario
#   → invalid records held → reviewer decisions imported → day-30 outcome report.
# Outputs: artifacts/use_cases/ (gitignored). Committed bundle only; nothing retrains.
#
#   make use-cases        # or: ./scripts/run_use_cases.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CHURN_DATA_SOURCE="${CHURN_DATA_SOURCE:-synthetic}"
PY="${PYTHON:-python}"
UC="$ROOT/data/use_cases"
OUT="${USE_CASE_OUT:-$ROOT/artifacts/use_cases}"
SCORED_AT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["calendar"]["scored_at"])' "$UC/personas.json")"
rm -rf "$OUT"
mkdir -p "$OUT"

echo "==> [1/6] pack still matches the committed bundle"
"$PY" -m retention_radar.cli.build_use_cases --check

echo
echo "==> [2/6] weekly batch → ranked review queue (invalid rows rejected, not scored)"
"$PY" -m retention_radar.cli.batch_score --csv "$UC/weekly_batch.csv" \
  --out "$OUT/queue.csv" --jsonl "$OUT/queue.jsonl" --scored-at "$SCORED_AT"
echo "Top of the queue:"
head -6 "$OUT/queue.csv" | cut -d, -f1-6

echo
echo "==> [3/6] one decision packet per scenario (validate → score → explain → HITL)"
"$PY" -m retention_radar.cli.single_record --dir "$UC/records" --out "$OUT/packets.jsonl"

echo
echo "==> [4/6] invalid records are held for data fixes"
"$PY" -m retention_radar.cli.single_record --dir "$UC/invalid" --out "$OUT/held_packets.jsonl"

echo
echo "==> [5/6] reviewer decisions → HITL review log (score fields come from the queue)"
"$PY" -m retention_radar.cli.hitl_log --from-scores "$OUT/queue.csv" \
  --decisions "$UC/review_decisions.csv" --log "$OUT/hitl_review_log.csv"

echo
echo "==> [6/6] day-30 labels → outcome report"
"$PY" -m retention_radar.cli.hitl_outcomes --log "$OUT/hitl_review_log.csv" \
  --labels "$UC/labels_day30.csv" --out "$OUT/hitl_outcomes.csv"

echo
echo "==> use-case walk complete → $OUT"
