#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
INSTALL_IBC_SCRIPT="${INSTALL_IBC_SCRIPT:-$SCRIPT_DIR/install_ibc_macos.sh}"
WRITE_IBC_CONFIG_SCRIPT="${WRITE_IBC_CONFIG_SCRIPT:-$SCRIPT_DIR/write_ibc_runtime_config.py}"
PROBE_IBKR_API_SCRIPT="${PROBE_IBKR_API_SCRIPT:-$SCRIPT_DIR/probe_ibkr_api.py}"
IBC_VERSION="${IBC_VERSION:-3.23.0}"
IBC_PATH="${IBC_PATH:-$HOME/.autoresearch-trading/ibc/${IBC_VERSION}}"
IBC_INI="${IBC_INI:-$HOME/.autoresearch-trading/ibc/runtime/ibc-paper.ini}"
TWS_MAJOR_VRSN="${TWS_MAJOR_VRSN:-10.45}"
TWS_PATH="${TWS_PATH:-$HOME/Applications}"
IB_GATEWAY_API_PORTS="${IB_GATEWAY_API_PORTS:-4002,4000,7497,7496,4001}"
IB_GATEWAY_API_PORT="${IB_GATEWAY_API_PORT:-4002}"
IB_GATEWAY_WAIT_SECONDS="${IB_GATEWAY_WAIT_SECONDS:-240}"
IB_GATEWAY_STABLE_SECONDS="${IB_GATEWAY_STABLE_SECONDS:-10}"
IB_GATEWAY_KEEPALIVE_SECONDS="${IB_GATEWAY_KEEPALIVE_SECONDS:-0}"
LOG_PATH="${LOG_PATH:-$HOME/Library/Logs/autoresearch-trading/ibc}"

mkdir -p "$LOG_PATH"

if [[ ! -x "$IBC_PATH/scripts/ibcstart.sh" ]]; then
  if [[ ! -x "$INSTALL_IBC_SCRIPT" ]]; then
    INSTALL_IBC_SCRIPT="$REPO_ROOT/v4/ops/ibkr/install_ibc_macos.sh"
  fi
  "$INSTALL_IBC_SCRIPT" >/dev/null
fi

if [[ ! -f "$WRITE_IBC_CONFIG_SCRIPT" ]]; then
  WRITE_IBC_CONFIG_SCRIPT="$REPO_ROOT/v4/ops/ibkr/write_ibc_runtime_config.py"
fi

if ! "$PYTHON_BIN" "$WRITE_IBC_CONFIG_SCRIPT" \
  --out "$IBC_INI" \
  --api-port "$IB_GATEWAY_API_PORT" >/tmp/autoresearch-ibc-config.json; then
  echo "IBC runtime config was not created. Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally." >&2
  cat /tmp/autoresearch-ibc-config.json >&2 || true
  exit 2
fi

if [[ ! -f "$IBC_INI" ]]; then
  echo "IBC runtime config was not created. Run v4/ops/ibkr/store_ibkr_paper_credentials.sh locally." >&2
  cat /tmp/autoresearch-ibc-config.json >&2 || true
  exit 2
fi

if pgrep -f "java.*IBC.*${IBC_INI}" >/dev/null 2>&1; then
  echo "IBC already appears to be running for $IBC_INI"
else
  nohup "$IBC_PATH/scripts/ibcstart.sh" "$TWS_MAJOR_VRSN" \
    --gateway \
    --tws-path="$TWS_PATH" \
    --ibc-path="$IBC_PATH" \
    --ibc-ini="$IBC_INI" \
    --mode=paper \
    --on2fatimeout=exit \
    >"$LOG_PATH/ibc-gateway.out.log" \
    2>"$LOG_PATH/ibc-gateway.err.log" &
fi

if [[ ! -f "$PROBE_IBKR_API_SCRIPT" ]]; then
  PROBE_IBKR_API_SCRIPT="$REPO_ROOT/v4/ops/ibkr/probe_ibkr_api.py"
fi

if "$PYTHON_BIN" "$PROBE_IBKR_API_SCRIPT" \
  --ports "$IB_GATEWAY_API_PORTS" \
  --timeout-seconds "$IB_GATEWAY_WAIT_SECONDS" \
  --stable-seconds "$IB_GATEWAY_STABLE_SECONDS" \
  --hold-seconds "$IB_GATEWAY_KEEPALIVE_SECONDS" \
  --client-id 145 >/tmp/autoresearch-ibkr-api-probe.json; then
  cat /tmp/autoresearch-ibkr-api-probe.json
  exit 0
fi

echo "IBC started or attempted to start IB Gateway, but no stable IBKR API handshake was available after ${IB_GATEWAY_WAIT_SECONDS}s." >&2
echo "If IBKR Mobile/2FA is required, approve it; if credentials are missing, run store_ibkr_paper_credentials.sh." >&2
cat /tmp/autoresearch-ibkr-api-probe.json >&2 || true
exit 1
