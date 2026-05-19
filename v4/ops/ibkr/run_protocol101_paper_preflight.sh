#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
if [[ -x "$DEFAULT_PYTHON_BIN" && "$PYTHON_BIN" == "/usr/bin/python3" ]]; then
  PYTHON_BIN="$DEFAULT_PYTHON_BIN"
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
WAIT_FOR_IBKR_API_SCRIPT="${WAIT_FOR_IBKR_API_SCRIPT:-$SCRIPT_DIR/wait_for_ibkr_api.py}"
IB_GATEWAY_API_PORT="${IB_GATEWAY_API_PORT:-4002}"
IB_GATEWAY_API_PORTS="${IB_GATEWAY_API_PORTS:-4002,4000,7497,7496,4001}"
IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS="${IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS:-600}"

if [[ ! -f "$WAIT_FOR_IBKR_API_SCRIPT" ]]; then
  WAIT_FOR_IBKR_API_SCRIPT="$REPO_ROOT/v4/ops/ibkr/wait_for_ibkr_api.py"
fi

export PYTHON_BIN
case ":${PYTHONPATH:-}:" in
  *":$REPO_ROOT:"*) ;;
  *) export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" ;;
esac
cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$WAIT_FOR_IBKR_API_SCRIPT" \
  --port "$IB_GATEWAY_API_PORT" \
  --auto-ports "$IB_GATEWAY_API_PORTS" \
  --timeout-seconds "$IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS" \
  --run-entitlement-probe \
  --repo-root "$REPO_ROOT"
