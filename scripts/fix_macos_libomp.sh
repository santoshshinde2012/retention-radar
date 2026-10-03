#!/usr/bin/env bash
# macOS without `brew install libomp`: the XGBoost and LightGBM wheels only look for libomp under
# /opt/homebrew, but scikit-learn ships its own copy. Point both libraries at it (venv-local, ad-hoc
# re-signed). No-op on Linux, or when XGBoost and LightGBM already load. Called by `make setup`.
#   ./scripts/fix_macos_libomp.sh [venv_dir]
set -euo pipefail
VENV="${1:-.venv}"
PY="$VENV/bin/python"
[[ "$(uname)" == Darwin ]] || exit 0
if "$PY" -c 'import xgboost, lightgbm' >/dev/null 2>&1; then exit 0; fi
site="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ ! -f "$site/sklearn/.dylibs/libomp.dylib" ]]; then
  echo "XGBoost/LightGBM cannot load libomp and scikit-learn has no bundled copy: brew install libomp" >&2
  exit 1
fi
echo "XGBoost/LightGBM cannot load libomp; using scikit-learn's bundled copy (or: brew install libomp)"
for lib in "$site/xgboost/lib/libxgboost.dylib" "$site/lightgbm/lib/lib_lightgbm.dylib"; do
  [[ -f "$lib" ]] || continue
  install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$lib" 2>/dev/null || true
  codesign --force --sign - "$lib" >/dev/null 2>&1 || true
done
"$PY" -c 'import xgboost, lightgbm' || { echo "XGBoost/LightGBM still fail to load: brew install libomp" >&2; exit 1; }
echo "XGBoost and LightGBM load."
