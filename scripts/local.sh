#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${READMISSION_PYTHON:-.venv/bin/python}"
# Apple Silicon XGBoost needs OpenMP; reuse scikit-learn's bundled runtime.
if [[ "$(uname -s)" == "Darwin" ]]; then
  READMISSION_OMP_DIR="$($PYTHON_BIN -c 'from pathlib import Path; import sklearn; print(Path(sklearn.__file__).parent / ".dylibs")')"
  export DYLD_FALLBACK_LIBRARY_PATH="$READMISSION_OMP_DIR${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi
case "${1:-train}" in
  prepare) shift; exec "$PYTHON_BIN" -m readmission.prepare "$@" ;;
  train) [[ $# -eq 0 ]] || shift; exec "$PYTHON_BIN" -m readmission.train "$@" ;;
  serve) shift; exec "$PYTHON_BIN" -m uvicorn readmission.serve:app --host 127.0.0.1 --port 8080 "$@" ;;
  test) shift; exec "$PYTHON_BIN" -m unittest discover -s tests -v "$@" ;;
  *) printf 'Use prepare, train, serve, or test.\n' >&2; exit 2 ;;
esac

