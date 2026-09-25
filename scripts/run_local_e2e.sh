#!/usr/bin/env bash
# Everything, locally, end to end — and prove it without touching committed files.
#
#   make e2e-local                    # or: ./scripts/run_local_e2e.sh
#   LAKEHOUSE_ROOT=/path/to/local-data-lakehouse make e2e-local
#
# Steps: env check → lint → seed-42 canary → reproduce the published ladder in an
# isolated dir (diff vs models/metrics.json) → full pytest (incl. headless Streamlit
# + API) → every serve surface on the committed bundle (infer, packet, batch,
# HITL log, outcomes, drift, live API, live Streamlit) → optional lakehouse E2E
# (diff vs results/lakehouse-e2e-summary.json) → git isolation check.
# Outputs land in artifacts/local_e2e/ (gitignored).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CHURN_DATA_SOURCE=synthetic
unset RETENTION_RADAR_ARTIFACT_DIR
PY="${PYTHON:-python}"
if [[ "$PY" == */* ]]; then PY="$(cd "$(dirname "$PY")" && pwd)/$(basename "$PY")"; fi
export PYTHON="$PY"
OUT="$ROOT/artifacts/local_e2e"
API_PORT="${API_PORT:-8765}"
UI_PORT="${UI_PORT:-8599}"
rm -rf "$OUT"
mkdir -p "$OUT"

GUARDED=(models docs results data/raw data/use_cases)
tree_state() { # content fingerprint of the committed-output paths (tracked diffs + untracked files)
  git -C "$ROOT" diff --no-ext-diff -- "${GUARDED[@]}"
  git -C "$ROOT" ls-files -o --exclude-standard -- "${GUARDED[@]}" | sort | while read -r f; do
    sha256sum "$f"
  done
}
HAS_GIT=0
if git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
  HAS_GIT=1
  BEFORE="$(tree_state | sha256sum)"
fi

PASSED=()
SKIPPED=()
step() { echo; echo "==> [$1] $2"; }
ok() { PASSED+=("$1"); }

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

wait_http() { # url, seconds
  local url="$1" n="${2:-60}"
  for _ in $(seq 1 "$n"); do
    if "$PY" -c "import sys,urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2)" "$url" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

step 1 "environment"
"$PY" - <<'PY'
import sys
from importlib import metadata
if sys.version_info < (3, 12):
    sys.exit(f"Python {sys.version.split()[0]} found; Python 3.12+ is required "
             "(the committed bundle was trained with XGBoost 3.4.1). "
             "Create the venv with python3.12: make setup PYTHON=python3.12")
pins = {"xgboost": "3.4.1", "scikit-learn": "1.9.1", "optuna": "5.0.0",
        "lightgbm": "4.7.0", "catboost": "1.2.10"}
bad = {k: metadata.version(k) for k in pins if metadata.version(k) != pins[k]}
print("python", sys.version.split()[0], "·", " · ".join(f"{k} {metadata.version(k)}" for k in pins))
if bad:
    sys.exit(f"Pinned libs differ from requirements.txt: {bad}. Run: pip install -r requirements.txt")
PY
ok env

step 2 "lint (ruff)"
if "$PY" -m ruff --version >/dev/null 2>&1; then
  "$PY" -m ruff check src/ tests/ app/
  ok lint
else
  echo "ruff not installed — skipped (pip install ruff)"
  SKIPPED+=(lint)
fi

step 3 "seed-42 canary on the committed bundle"
"$PY" -m pytest -q -p no:warnings tests/test_seed42_canary.py tests/test_artifact_dir_isolation.py
ok canary

step 4 "reproduce the published ladder (isolated retrain, N_USERS=5000, 20 trials)"
RETENTION_RADAR_ARTIFACT_DIR="$OUT/repro" N_USERS=5000 N_OPTUNA_TRIALS=20 \
  ./scripts/run_all.sh > "$OUT/reproduce.log" 2>&1 || { tail -40 "$OUT/reproduce.log"; exit 1; }
"$PY" -m retention_radar.cli.check_reproduction --artifact-dir "$OUT/repro"
ok reproduce

step 5 "full test suite (pipeline smoke, API, headless Streamlit, contracts)"
"$PY" -m pytest -q -p no:warnings
ok pytest

step 6 "use cases through every CLI surface (queue → packets → held → reviews → outcomes)"
"$PY" -m retention_radar.cli.infer --user santosh
USE_CASE_OUT="$OUT/use_cases" ./scripts/run_use_cases.sh
"$PY" -m retention_radar.cli.drift_check --strict --z-threshold 3.0 --out "$OUT/drift_report.json"
ok use-cases

# Launch through the console scripts, like the README and Streamlit Cloud do: unlike
# `python -m`, they do not put the repo root on sys.path, which is what exposes
# bundles that only load from the repo root.
BIN_DIR="$(dirname "$(command -v "$PY")")"
console() { if [[ -x "$BIN_DIR/$1" ]]; then echo "$BIN_DIR/$1"; else echo "$PY -m $1"; fi; }

step 7 "live thin API (uvicorn console script :$API_PORT)"
$(console uvicorn) retention_radar.serving.api:app --app-dir src --host 127.0.0.1 --port "$API_PORT" \
  > "$OUT/api.log" 2>&1 &
PIDS+=($!)
wait_http "http://127.0.0.1:$API_PORT/healthz" 60 || { cat "$OUT/api.log"; exit 1; }
"$PY" - "$API_PORT" <<'PY'
import csv, json, sys, urllib.error, urllib.request
base = f"http://127.0.0.1:{sys.argv[1]}"

def post(path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)

manifest = json.load(open("data/use_cases/personas.json"))
for p in manifest["personas"]:
    code, out = post("/v1/churn/score", p["record"])
    assert code == 200, (p["id"], out)
    assert (out["band"], out["hitl_action"]) == (p["expected"]["band"], p["expected"]["hitl_action"]), (p["id"], out)
    print(f"POST /v1/churn/score {p['id']:<24} → {out['band']:<6} {out['hitl_action']}")
for case in manifest["invalid_records"]:
    code, out = post("/v1/churn/score", case["record"])
    assert code == 422, (case["id"], code, out)
print(f"POST /v1/churn/score invalid records → 422 x{len(manifest['invalid_records'])}")
rows = list(csv.DictReader(open("data/use_cases/weekly_batch.csv")))
records = [{k: v for k, v in r.items() if v != ""} for r in rows]
code, out = post("/v1/churn/batch", {"records": records})
assert code == 200 and len(out["rejected"]) == manifest["weekly_batch"]["invalid_rows"], out
print(f"POST /v1/churn/batch → queue {len(out['queue'])}, rejected {len(out['rejected'])}, top {out['queue'][0]['hitl_action']}")
gone = next(p for p in manifest["personas"] if p["id"] == "gone_dark")["record"]["user_id"]
code, out = post("/v1/churn/reviews", {"user_id": gone, "reviewer": "local-e2e", "action_taken": "escalate"})
assert code == 200 and out["logged"]["action_suggested"] == "escalate", out
print("POST /v1/churn/reviews → logged against the service's own score")
PY
kill "${PIDS[-1]}" 2>/dev/null || true
ok api

step 8 "live Streamlit server (streamlit console script, headless :$UI_PORT)"
$(console streamlit) run app/streamlit_app.py --server.headless true \
  --server.port "$UI_PORT" --browser.gatherUsageStats false > "$OUT/streamlit.log" 2>&1 &
PIDS+=($!)
wait_http "http://127.0.0.1:$UI_PORT/_stcore/health" 90 || { cat "$OUT/streamlit.log"; exit 1; }
echo "Streamlit healthy at http://127.0.0.1:$UI_PORT (page render covered by tests/test_streamlit_app.py)"
kill "${PIDS[-1]}" 2>/dev/null || true
ok streamlit

step 9 "lakehouse gold E2E (optional)"
LAKE="${LAKEHOUSE_ROOT:-}"
if [[ -z "$LAKE" && -d "$ROOT/../local-data-lakehouse/scripts" ]]; then
  LAKE="$ROOT/../local-data-lakehouse"
fi
if [[ -n "$LAKE" && -d "$LAKE/scripts" ]]; then
  LAKEHOUSE_SUMMARY_PATH="$OUT/lakehouse-e2e-summary.json" \
    ./scripts/run_lakehouse_e2e.sh "$LAKE" > "$OUT/lakehouse.log" 2>&1 \
    || { tail -40 "$OUT/lakehouse.log"; exit 1; }
  if [[ -f "$LAKE/scripts/check_churn_export.py" ]]; then
    "$PY" "$LAKE/scripts/check_churn_export.py"
  fi
  "$PY" - "$OUT/lakehouse-e2e-summary.json" <<'PY'
import json, sys
new = json.load(open(sys.argv[1]))
ref = json.load(open("results/lakehouse-e2e-summary.json"))
keys = ["n_train", "n_test", "churn_rate_train", "best_optuna_auc_val",
        "calibrated_test_roc_auc", "best_f1_threshold", "santosh"]
diff = {k: (ref.get(k), new.get(k)) for k in keys if ref.get(k) != new.get(k)}
print("lakehouse summary:", {k: new.get(k) for k in keys})
if diff:
    sys.exit(f"lakehouse E2E differs from results/lakehouse-e2e-summary.json: {diff}")
print("lakehouse E2E matches the committed summary")
PY
  ok lakehouse
else
  echo "skipped — clone local-data-lakehouse beside this repo or set LAKEHOUSE_ROOT"
  SKIPPED+=(lakehouse)
fi

step 10 "committed files untouched"
if [[ "$HAS_GIT" == 1 ]]; then
  if [[ "$(tree_state | sha256sum)" != "$BEFORE" ]]; then
    git -C "$ROOT" status --short -- "${GUARDED[@]}"
    echo "committed outputs changed during the run (models/ docs/ results/ data/raw/ data/use_cases/)"
    exit 1
  fi
  echo "models/ docs/ results/ data/raw/ data/use_cases/ unchanged by the run"
  ok isolation
else
  SKIPPED+=(isolation)
fi

echo
echo "==> local E2E complete: passed [${PASSED[*]}]${SKIPPED:+ · skipped [${SKIPPED[*]}]}"
echo "    outputs: $OUT"
