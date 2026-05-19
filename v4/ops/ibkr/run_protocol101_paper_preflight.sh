#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
WAIT_FOR_IBKR_API_SCRIPT="${WAIT_FOR_IBKR_API_SCRIPT:-$SCRIPT_DIR/wait_for_ibkr_api.py}"
IB_GATEWAY_API_PORT="${IB_GATEWAY_API_PORT:-4002}"
IB_GATEWAY_API_PORTS="${IB_GATEWAY_API_PORTS:-4002,4000,7497,7496,4001}"
IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS="${IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS:-600}"

if [[ ! -f "$WAIT_FOR_IBKR_API_SCRIPT" ]]; then
  WAIT_FOR_IBKR_API_SCRIPT="$REPO_ROOT/v4/ops/ibkr/wait_for_ibkr_api.py"
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$WAIT_FOR_IBKR_API_SCRIPT" \
  --port "$IB_GATEWAY_API_PORT" \
  --auto-ports "$IB_GATEWAY_API_PORTS" \
  --timeout-seconds "$IB_GATEWAY_PREFLIGHT_TIMEOUT_SECONDS" \
  --run-entitlement-probe \
  --repo-root "$REPO_ROOT"
