#!/usr/bin/env bash
# One terminal command from a clean, dedicated compiler-research checkout.
# No system Python mutation, RL imports, background jobs or automatic Vitis.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ ! -f "$ROOT/experiments/compiler_ir/series.json" ]]; then
  echo "Missing compiler-IR experiment series config" >&2
  exit 2
fi

BOOTSTRAP=0
if [[ "$#" -gt 0 && "$1" == "--bootstrap" ]]; then
  BOOTSTRAP=1
  shift
fi

if [[ "$BOOTSTRAP" == 1 ]]; then
  PYTHON="$ROOT/.venv-compiler-ir/bin/python"
  if [[ -v FPGAI_SERIES_PYTHON ]]; then PYTHON="$FPGAI_SERIES_PYTHON"; fi
  if [[ ! -x "$PYTHON" ]]; then
    python3 -m venv "$ROOT/.venv-compiler-ir"
  fi
  # Only compiler-paper reference/test deps, not RL/Gym/Torch.
  # Isolated venv must not overwrite the user's running L0.
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -e "$ROOT[test,reference]"
else
  if [[ -v FPGAI_SERIES_PYTHON ]]; then
    PYTHON="$FPGAI_SERIES_PYTHON"
  elif [[ -x "$ROOT/.venv-compiler-ir/bin/python" ]]; then
    PYTHON="$ROOT/.venv-compiler-ir/bin/python"
  elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

if [[ ! -x "$PYTHON" ]] && ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "No Python found; use --bootstrap to prepare isolated dependencies" >&2
  exit 2
fi
exec "$PYTHON" "$ROOT/experiments/compiler_ir/series.py" "$@"
