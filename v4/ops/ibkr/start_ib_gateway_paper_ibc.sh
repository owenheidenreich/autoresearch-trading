#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/gduby/Documents/autoresearch-trading}"
IBC_VERSION="${IBC_VERSION:-3.23.0}"
IBC_PATH="${IBC_PATH:-$HOME/.autoresearch-trading/ibc/${IBC_VERSION}}"
IBC_INI="${IBC_INI:-$HOME/.autoresearch-trading/ibc/runtime/ibc-paper.ini}"
TWS_MAJOR_VRSN="${TWS_MAJOR_VRSN:-10.45}"
TWS_PATH="${TWS_PATH:-$HOME/Applications}"
IB_GATEWAY_API_PORTS="${IB_GATEWAY_API_PORTS:-4002,4000,7497,7496,4001}"
IB_GATEWAY_API_PORT="${IB_GATEWAY_API_PORT:-4002}"
IB_GATEWAY_WAIT_SECONDS="${IB_GATEWAY_WAIT_SECONDS:-240}"
LOG_PATH="${LOG_PATH:-$HOME/Library/Logs/autoresearch-trading/ibc}"

mkdir -p "$LOG_PATH"

if [[ ! -x "$IBC_PATH/scripts/ibcstart.sh" ]]; then
  "$REPO_ROOT/v4/ops/ibkr/install_ibc_macos.sh" >/dev/null
fi

if ! "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/v4/ops/ibkr/write_ibc_runtime_config.py" \
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

deadline=$((SECONDS + IB_GATEWAY_WAIT_SECONDS))
while [[ "$SECONDS" -lt "$deadline" ]]; do
  IFS=',' read -ra CANDIDATE_PORTS <<< "$IB_GATEWAY_API_PORTS"
  for port in "${CANDIDATE_PORTS[@]}"; do
    port="${port//[[:space:]]/}"
    if [[ -n "$port" ]] && lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
      echo "IB Gateway API listener is up on port $port via IBC"
      exit 0
    fi
  done
  sleep 5
done

echo "IBC started or attempted to start IB Gateway, but no API port from [$IB_GATEWAY_API_PORTS] was listening after ${IB_GATEWAY_WAIT_SECONDS}s." >&2
echo "If IBKR Mobile/2FA is required, approve it; if credentials are missing, run store_ibkr_paper_credentials.sh." >&2
exit 1
