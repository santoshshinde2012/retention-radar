#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  echo "Usage: $0 /path/to/local-data-lakehouse/data/export" >&2
  exit 2
fi
export PYTHONPATH="$ROOT"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
python - <<PY
from pathlib import Path
from src.retention_radar.data.ingest import sync_lakehouse_exports
csv_p, json_p = sync_lakehouse_exports(Path("$SRC"))
print(f"Synced → {csv_p}")
print(f"Synced → {json_p}")
PY
