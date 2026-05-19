#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

if [[ -x "$DEFAULT_PYTHON_BIN" && "$PYTHON_BIN" == "/usr/bin/python3" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON_BIN"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

export PYTHON_BIN
case ":${PYTHONPATH:-}:" in
  *":$REPO_ROOT:"*) ;;
  *) export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m v4.scripts.run_protocol156_ibkr_autostart_observability "$@"
