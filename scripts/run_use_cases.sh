#!/usr/bin/env bash
# Walk one day of the renewal business on the seed-42 use-case pack (data/use_cases/):
#   daily T-7 batch → ranked action queue (+ rejects) → decision packet per scenario
#   → invalid records held → send export imported → renewal outcomes + lift vs holdout.
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
echo "==> [2/6] daily T-7 batch → ranked action queue (invalid rows rejected, not scored)"
"$PY" -m retention_radar.cli.batch_score --csv "$UC/daily_t7_batch.csv" \
  --out "$OUT/queue.csv" --jsonl "$OUT/queue.jsonl" --scored-at "$SCORED_AT"
echo "Top of the queue:"
head -6 "$OUT/queue.csv" | cut -d, -f1-8

echo
echo "==> [3/6] one decision packet per scenario (validate → score → explain → action)"
"$PY" -m retention_radar.cli.single_record --dir "$UC/records" --out "$OUT/packets.jsonl"

echo
echo "==> [4/6] invalid records are held for data fixes"
"$PY" -m retention_radar.cli.single_record --dir "$UC/invalid" --out "$OUT/held_packets.jsonl"

echo
echo "==> [5/6] send export → action log (score + holdout fields come from the queue)"
"$PY" -m retention_radar.cli.action_log --from-scores "$OUT/queue.csv" \
  --decisions "$UC/actions_taken.csv" --log "$OUT/action_log.csv"

echo
echo "==> [6/6] renewal outcomes → calibration by band + lift vs holdout"
"$PY" -m retention_radar.cli.outcomes --log "$OUT/action_log.csv" \
  --labels "$UC/renewal_outcomes.csv" --out "$OUT/outcomes.csv"

echo
echo "==> use-case walk complete → $OUT"
